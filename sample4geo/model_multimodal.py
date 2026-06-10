import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.autograd import Function

from sample4geo.models.disentangle_heads import DisentangleHeads
from sample4geo.models.cvgtext_openai_clip_text_encoder import CVGTextOpenAIClipTextEncoder


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class ResidualBasisAdapter(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, dropout=0.1, init_scale=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.scale = nn.Parameter(torch.ones([]) * init_scale)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        h = self.mlp(self.norm(x))
        y = x + self.scale * h
        y = self.bn(y)
        return F.normalize(y, dim=-1)


class TextSemanticAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, hidden_dim=1536, dropout=0.1, init_scale=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.scale = nn.Parameter(torch.ones([]) * init_scale)
        self.bn = nn.BatchNorm1d(out_dim)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.mlp(h)
        y = self.residual(x) + self.scale * h
        y = self.bn(y)
        return F.normalize(y, dim=-1)


class MultiModalGeoModel(nn.Module):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=384,
                 pretrained_path=None,
                 sem_dim=512,
                 sty_dim=512,
                 text_max_len=300,
                 text_model_name="openai/clip-vit-large-patch14-336",
                 text_checkpoint_path=None,
                 freeze_text_encoder=True,
                 strict_text_ckpt=False,
                 fusion_hidden_dim=None,
                 ebr_hidden_dim=None,
                 adv_hidden_dim=256):
        super().__init__()

        pretrained_cfg_overlay = None
        if pretrained and pretrained_path is not None:
            pretrained_cfg_overlay = {"file": pretrained_path}

        if "vit" in model_name:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, pretrained_cfg_overlay=pretrained_cfg_overlay)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, pretrained_cfg_overlay=pretrained_cfg_overlay)

        feat_dim = getattr(self.backbone, "num_features", sem_dim)
        self.heads = DisentangleHeads(in_dim=feat_dim, sem_dim=sem_dim, sty_dim=sty_dim)

        self.text_encoder = CVGTextOpenAIClipTextEncoder(
            checkpoint_path=text_checkpoint_path,
            model_name_for_tokenizer=text_model_name,
            context_length=text_max_len,
            vocab_size=49408,
            width=768,
            layers=12,
            heads=12,
            normalize=True,
            freeze_text_encoder=freeze_text_encoder,
        )
        self.text_adapter = TextSemanticAdapter(in_dim=768, out_dim=sem_dim, hidden_dim=sem_dim * 2, dropout=0.1, init_scale=1)

        ebr_hidden_dim = ebr_hidden_dim if ebr_hidden_dim is not None else sem_dim * 2
        self.image_basis_adapter = ResidualBasisAdapter(dim=sem_dim, hidden_dim=ebr_hidden_dim, dropout=0.1, init_scale=0.1)
        self.text_basis_adapter = ResidualBasisAdapter(dim=sem_dim, hidden_dim=ebr_hidden_dim, dropout=0.1, init_scale=0.1)
        self.image_proj_adapter = ResidualBasisAdapter(dim=sem_dim, hidden_dim=ebr_hidden_dim, dropout=0.1, init_scale=0.1)
        self.text_proj_adapter = ResidualBasisAdapter(dim=sem_dim, hidden_dim=ebr_hidden_dim, dropout=0.1, init_scale=0.1)

        self.modality_discriminator = nn.Sequential(
            nn.Linear(sem_dim, adv_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adv_hidden_dim, 2),
        )

        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(sem_dim * 4, fusion_hidden_dim if fusion_hidden_dim is not None else sem_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim if fusion_hidden_dim is not None else sem_dim, sem_dim),
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        return timm.data.resolve_model_data_config(self.backbone)

    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)

    def encode_semantic(self, img):
        feat = self.backbone(img)
        sem, _ = self.heads(feat)
        return F.normalize(sem, dim=-1)

    def _gate_fusion(self, sem_like, text_like):
        fusion_input = torch.cat([sem_like, text_like, sem_like * text_like, torch.abs(sem_like - text_like)], dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate_mlp(fusion_input))
        fused_q = F.normalize(sem_like + fusion_gate * text_like, dim=-1)
        return fused_q, fusion_gate

    def forward(self, query_img, reference_img=None, texts=None, adv_grl_lambda=1.0):
        if reference_img is None:
            return self.encode_semantic(query_img)

        feat_q = self.backbone(query_img)
        feat_g = self.backbone(reference_img)

        sem_q, sty_q = self.heads(feat_q)
        sem_g, sty_g = self.heads(feat_g)

        sem_q = F.normalize(sem_q, dim=-1)
        sty_q = F.normalize(sty_q, dim=-1)
        sem_g = F.normalize(sem_g, dim=-1)
        sty_g = F.normalize(sty_g, dim=-1)

        text_emb = None
        sem_q_ebr = sem_q
        text_emb_ebr = None
        fused_q = sem_q
        fusion_gate = None
        logits_disc = None

        if texts is not None:
            text_feat = self.text_encoder(texts)
            text_emb = self.text_adapter(text_feat)

            u_v = self.image_basis_adapter(sem_q)
            u_t = self.text_basis_adapter(text_emb)
            sem_q_ebr = self.image_proj_adapter(u_v)
            text_emb_ebr = self.text_proj_adapter(u_t)

            fused_q, fusion_gate = self._gate_fusion(sem_q_ebr, text_emb_ebr)

            disc_in_v = grad_reverse(u_v, adv_grl_lambda)
            disc_in_t = grad_reverse(u_t, adv_grl_lambda)
            logits_disc = self.modality_discriminator(torch.cat([disc_in_v, disc_in_t], dim=0))

        return {
            "sem_q": sem_q,
            "sty_q": sty_q,
            "sem_g": sem_g,
            "sty_g": sty_g,
            "text_emb": text_emb,
            "sem_q_ebr": sem_q_ebr,
            "text_emb_ebr": text_emb_ebr,
            "fused_q": fused_q,
            "fusion_gate": fusion_gate,
            "disc_logits": logits_disc,
        }
