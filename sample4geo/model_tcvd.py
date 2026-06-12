# sample4geo/model_tcvd.py

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from sample4geo.models.cvgtext_openai_clip_text_encoder import CVGTextOpenAIClipTextEncoder


class MLPHead(nn.Module):
    """
    简单 MLP 投影头。

    用于：
        backbone feature -> content feature
        backbone feature -> viewpoint feature
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextSemanticAdapter(nn.Module):
    """
    文本语义适配器。

    CVGTextOpenAIClipTextEncoder 的输出维度是 768。
    但是 T-CVD 的 content_dim 由 alpha 决定。

    所以需要：
        text_feat [B, 768] -> text_emb [B, content_dim]
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights(init_scale)

    def _init_weights(self, init_scale):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02 * init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class CrossViewFeatureDecoder(nn.Module):
    """
    跨视角特征重建解码器。

    论文思想：
        用 satellite content + ground viewpoint 重建 ground
        用 ground content + satellite viewpoint 重建 satellite

    这里不是图像级重建，而是特征级重建：
        decoder_q(c_s_raw, v_q) -> rec_q ≈ z_q
        decoder_s(c_q_raw, v_s) -> rec_s ≈ z_s
    """

    def __init__(
        self,
        content_dim: int,
        viewpoint_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        in_dim = content_dim + viewpoint_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, content_raw, viewpoint):
        x = torch.cat([content_raw, viewpoint], dim=-1)
        return self.net(x)


class TextGuidedCVDModel(nn.Module):
    """
    Text-Guided Content-Viewpoint Disentanglement Model.

    训练阶段输入：
        query_img      : ground image
        reference_img  : satellite image
        texts          : tokenized text

    训练阶段输出：
        z_q, z_s
        c_q_raw, c_s_raw
        c_q, c_s
        v_q, v_s
        text_emb
        rec_q, rec_s
        logit_scale

    推理阶段输入：
        img

    推理阶段输出：
        content feature

    关键点：
        1. 使用 alpha 拆分 content/viewpoint 维度。
        2. 文本只对齐 content，不对齐 viewpoint。
        3. 推理阶段不使用文本。
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: int = 384,
        pretrained_path: str = None,

        # ========================================================
        # CVD alpha split
        # content_dim = alpha * backbone_feature_dim
        # viewpoint_dim = (1 - alpha) * backbone_feature_dim
        # ========================================================
        split_alpha: float = 0.5,

        hidden_dim: int = 2048,
        dropout: float = 0.1,

        text_model_name: str = "openai/clip-vit-large-patch14-336",
        text_checkpoint_path: str = None,
        text_max_len: int = 300,
        freeze_text_encoder: bool = True,
        strict_text_ckpt: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.img_size = img_size
        self.split_alpha = split_alpha
        self.strict_text_ckpt = strict_text_ckpt

        if not (0.0 < split_alpha < 1.0):
            raise ValueError(
                f"split_alpha must be in (0, 1), but got {split_alpha}"
            )

        # ========================================================
        # Image backbone
        # ========================================================
        pretrained_cfg_overlay = None

        if pretrained and pretrained_path is not None:
            if os.path.exists(pretrained_path):
                pretrained_cfg_overlay = {"file": pretrained_path}
            else:
                print(
                    f"[T-CVD WARNING] pretrained_path not found: {pretrained_path}"
                )
                print("[T-CVD WARNING] timm will try default pretrained loading.")

        if "vit" in model_name.lower():
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                pretrained_cfg_overlay=pretrained_cfg_overlay,
            )

        feat_dim = getattr(self.backbone, "num_features", None)

        if feat_dim is None:
            raise ValueError(
                "Cannot get backbone num_features. Please check timm model."
            )

        self.feat_dim = int(feat_dim)

        # ========================================================
        # Alpha split
        # ========================================================
        content_dim = int(round(self.feat_dim * split_alpha))
        viewpoint_dim = self.feat_dim - content_dim

        if content_dim <= 0 or viewpoint_dim <= 0:
            raise ValueError(
                f"Invalid split result: content_dim={content_dim}, "
                f"viewpoint_dim={viewpoint_dim}. "
                f"Please check split_alpha={split_alpha}."
            )

        self.content_dim = content_dim
        self.viewpoint_dim = viewpoint_dim

        print("============================================================")
        print("[T-CVD] Backbone feature dim:", self.feat_dim)
        print("[T-CVD] split_alpha:", self.split_alpha)
        print("[T-CVD] content_dim = alpha * feat_dim:", self.content_dim)
        print("[T-CVD] viewpoint_dim = (1-alpha) * feat_dim:", self.viewpoint_dim)
        print("============================================================")

        # ========================================================
        # Content / viewpoint heads
        # ========================================================
        self.content_head = MLPHead(
            in_dim=self.feat_dim,
            out_dim=self.content_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.viewpoint_head = MLPHead(
            in_dim=self.feat_dim,
            out_dim=self.viewpoint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ========================================================
        # Text encoder
        # ========================================================
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

        # 文本编码器输出固定是 768。
        # 这里映射到 content_dim，使 text_emb 和 c_q/c_s 维度一致。
        self.text_adapter = TextSemanticAdapter(
            in_dim=768,
            out_dim=self.content_dim,
            hidden_dim=max(self.content_dim * 2, 512),
            dropout=dropout,
            init_scale=0.1,
        )

        # ========================================================
        # Cross-view feature reconstruction decoders
        # ========================================================
        self.decoder_q = CrossViewFeatureDecoder(
            content_dim=self.content_dim,
            viewpoint_dim=self.viewpoint_dim,
            out_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.decoder_s = CrossViewFeatureDecoder(
            content_dim=self.content_dim,
            viewpoint_dim=self.viewpoint_dim,
            out_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # InfoNCE temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07)
        )

    def get_config(self):
        """
        返回 timm backbone 的数据预处理配置。
        train_cvgtext_tcvd.py 里会用到 mean/std。
        """

        if hasattr(self.backbone, "pretrained_cfg"):
            cfg = self.backbone.pretrained_cfg
        else:
            cfg = {}

        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))

        return {
            "mean": mean,
            "std": std,
        }

    def encode_image_raw(self, img):
        """
        图像 backbone 原始特征。

        输入:
            img: [B, 3, H, W]

        输出:
            z: [B, feat_dim]
        """
        z = self.backbone(img)

        if z.dim() > 2:
            z = torch.flatten(z, start_dim=1)

        return z

    def disentangle(self, z):
        """
        将原始图像特征 z 拆成 content 和 viewpoint。

        输入:
            z: [B, feat_dim]

        输出:
            content_raw: [B, content_dim]
            content:     [B, content_dim]  normalized
            viewpoint:   [B, viewpoint_dim]
        """

        content_raw = self.content_head(z)
        viewpoint = self.viewpoint_head(z)

        content = F.normalize(content_raw, dim=-1)

        return content_raw, content, viewpoint

    def encode_content(self, img):
        """
        推理阶段使用。

        单张图像 -> content feature。

        输入:
            img: [B, 3, H, W]

        输出:
            content: [B, content_dim]
        """

        z = self.encode_image_raw(img)
        _, content, _ = self.disentangle(z)
        return content

    def encode_text_feature(self, texts):
        """
        文本编码。

        输入:
            texts 可以是：
                1. tokenized tensor: [B, text_len]
                2. list[str]

        输出:
            text_emb: [B, content_dim]
        """

        text_feat = self.text_encoder(texts)

        if text_feat.dim() > 2:
            text_feat = torch.flatten(text_feat, start_dim=1)

        text_emb = self.text_adapter(text_feat)

        return text_emb

    def forward(self, query_img, reference_img=None, texts=None):
        """
        两种模式：

        1. 推理模式:
            model(img)

            此时 reference_img is None，返回 content feature。

        2. 训练模式:
            model(query_img, reference_img, texts)

            返回 T-CVD 损失需要的所有中间变量。
        """

        # ========================================================
        # Inference mode
        # ========================================================
        if reference_img is None:
            return self.encode_content(query_img)

        # ========================================================
        # Training mode
        # ========================================================
        z_q = self.encode_image_raw(query_img)
        z_s = self.encode_image_raw(reference_img)

        c_q_raw, c_q, v_q = self.disentangle(z_q)
        c_s_raw, c_s, v_s = self.disentangle(z_s)

        # --------------------------------------------------------
        # Cross-view feature reconstruction
        #
        # rec_q = D_q(satellite content, ground viewpoint)
        # rec_s = D_s(ground content, satellite viewpoint)
        # --------------------------------------------------------
        rec_q = self.decoder_q(c_s_raw, v_q)
        rec_s = self.decoder_s(c_q_raw, v_s)

        # --------------------------------------------------------
        # Text semantic feature
        # --------------------------------------------------------
        text_emb = None

        if texts is not None:
            text_emb = self.encode_text_feature(texts)

        return {
            # Original image features
            "z_q": z_q,
            "z_s": z_s,

            # Content features
            "c_q_raw": c_q_raw,
            "c_s_raw": c_s_raw,
            "c_q": c_q,
            "c_s": c_s,

            # Viewpoint features
            "v_q": v_q,
            "v_s": v_s,

            # Text feature
            "text_emb": text_emb,

            # Reconstruction features
            "rec_q": rec_q,
            "rec_s": rec_s,

            # Temperature
            "logit_scale": self.logit_scale.exp(),
        }