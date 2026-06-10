import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):
    """OpenAI CLIP-style MLP with explicit key names: c_fc / c_proj."""

    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = CLIPMLP(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class CVGTextOpenAIClipTextEncoder(nn.Module):
    """
    OpenAI CLIP-style text tower compatible with CVG-Text/CrossText2Loc text keys.

    Expected text key names in checkpoint:
      - token_embedding.weight
      - positional_embedding
      - transformer.resblocks.*
      - ln_final.weight / ln_final.bias
      - text_projection

    Visual keys (visual.*) are skipped on load.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name_for_tokenizer: str = "openai/clip-vit-large-patch14-336",
        context_length: int = 300,
        vocab_size: int = 49408,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        normalize: bool = True,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.normalize = normalize

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_for_tokenizer)
        # Keep tokenizer warning-free for long-text setting; does not change model architecture.
        self.tokenizer.model_max_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(context_length),
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))

        self._init_parameters()

        # Keep explicit inspect API for training script / debugging consistency.
        self.inspect_info = self.inspect_checkpoint(checkpoint_path)

        load_info = self.load_text_from_checkpoint(checkpoint_path)
        self.load_info = load_info

        if freeze_text_encoder:
            self.freeze_text_encoder()

    @staticmethod
    def build_attention_mask(context_length: int) -> torch.Tensor:
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    def freeze_text_encoder(self):
        for p in self.parameters():
            p.requires_grad = False

    def tokenize(self, texts: Iterable[str], max_text_len: int = None) -> torch.Tensor:
        """
        OpenAI CLIP BPE-equivalent tokenization behavior:
        - pad token id: 0
        - keeps EOT token
        - truncation guarantees final token is EOT
        """
        max_len = max_text_len or self.context_length

        bos_id = self.tokenizer.bos_token_id
        eot_id = self.tokenizer.eos_token_id
        if bos_id is None:
            bos_id = 49406
        if eot_id is None:
            eot_id = 49407

        all_ids: List[List[int]] = []
        for t in texts:
            piece_ids = self.tokenizer.encode(t, add_special_tokens=False)
            ids = [bos_id] + piece_ids
            ids = ids[: max_len - 1]
            ids = ids + [eot_id]

            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))

            # safety: ensure last non-pad token is EOT
            last_non_pad = 0
            for i, x in enumerate(ids):
                if x != 0:
                    last_non_pad = i
            ids[last_non_pad] = eot_id

            all_ids.append(ids)

        return torch.tensor(all_ids, dtype=torch.long)

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens: [B, context_length]
        x = self.token_embedding(text_tokens)  # [B, L, C]
        x = x + self.positional_embedding.unsqueeze(0)  # [B, L, C]

        x = x.permute(1, 0, 2)  # [L, B, C]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, C]

        x = self.ln_final(x)

        # OpenAI CLIP-style EOT pooling
        eot_pos = text_tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_pos]
        x = x @ self.text_projection

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x

    def forward(self, texts):
        if torch.is_tensor(texts):
            text_tokens = texts
        else:
            text_tokens = self.tokenize(list(texts), max_text_len=self.context_length)

        device = self.positional_embedding.device
        text_tokens = text_tokens.to(device)
        return self.encode_text(text_tokens)

    def inspect_checkpoint(self, checkpoint_path: str) -> Dict[str, object]:
        """
        Inspect checkpoint text keys for explicit sanity checking.
        Returns a dict used by training/check scripts.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
        state, wrapped = self._extract_state_dict(ckpt_obj)

        print(f"[inspect] checkpoint: {checkpoint_path}")
        print(f"[inspect] wrapped_state_dict: {wrapped}")
        if isinstance(ckpt_obj, dict):
            print("[inspect] top-level keys:", list(ckpt_obj.keys())[:50])

        text_patterns = [
            "token_embedding",
            "positional_embedding",
            "transformer.resblocks",
            "ln_final",
            "text_projection",
        ]

        text_key_count = 0
        has_token_embedding = False
        has_positional_embedding = False
        has_ln_final_w = False
        has_ln_final_b = False
        has_text_projection = False
        hit_blocks = set()

        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            lk = k.lower()
            if lk.startswith("visual."):
                continue

            if any(p in lk for p in text_patterns):
                text_key_count += 1
                print(f"[inspect] {k}: {tuple(v.shape)}")

            if k == "token_embedding.weight":
                has_token_embedding = True
            elif k == "positional_embedding":
                has_positional_embedding = True
            elif k == "ln_final.weight":
                has_ln_final_w = True
            elif k == "ln_final.bias":
                has_ln_final_b = True
            elif k == "text_projection":
                has_text_projection = True

            if k.startswith("transformer.resblocks."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    hit_blocks.add(int(parts[2]))

        print(f"[inspect] checkpoint text keys: {text_key_count}")
        print(f"[inspect] transformer.resblocks hit: {len(hit_blocks)}/12")

        return {
            "wrapped_state_dict": wrapped,
            "checkpoint_text_keys": text_key_count,
            "has_token_embedding": has_token_embedding,
            "has_positional_embedding": has_positional_embedding,
            "has_ln_final_w": has_ln_final_w,
            "has_ln_final_b": has_ln_final_b,
            "has_text_projection": has_text_projection,
            "resblock_hit_count": len(hit_blocks),
            "resblock_ids": sorted(list(hit_blocks)),
        }

    def _extract_state_dict(self, ckpt_obj) -> Tuple[dict, bool]:
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"], True
        if isinstance(ckpt_obj, dict):
            return ckpt_obj, False
        raise ValueError("Unsupported checkpoint format.")

    def load_text_from_checkpoint(self, checkpoint_path: str) -> Dict[str, object]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
        src_state, wrapped = self._extract_state_dict(ckpt_obj)

        model_state = self.state_dict()
        filtered = {}
        text_key_count = 0

        for k, v in src_state.items():
            if not torch.is_tensor(v):
                continue
            if k.startswith("visual."):
                continue

            # Count text keys from checkpoint
            if (
                k == "token_embedding.weight"
                or k == "positional_embedding"
                or k.startswith("transformer.resblocks.")
                or k.startswith("ln_final.")
                or k == "text_projection"
            ):
                text_key_count += 1

            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                filtered[k] = v

        missing, unexpected = self.load_state_dict(filtered, strict=False)

        return {
            "wrapped_state_dict": wrapped,
            "checkpoint_text_key_count": text_key_count,
            "loaded_text_key_count": len(filtered),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "loaded_keys": list(filtered.keys()),
        }
