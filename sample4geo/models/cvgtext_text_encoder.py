import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


class CVGTextCLIPTextEncoder(nn.Module):
    """CLIP text tower wrapper with conditional positional expansion and checkpoint loading."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        max_text_len: int = 300,
        checkpoint_path: Optional[str] = None,
        freeze_text_encoder: bool = True,
        strict_load: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_text_len = max_text_len

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

        self.ckpt_positional_shape = None
        if checkpoint_path:
            self.ckpt_positional_shape = self.inspect_checkpoint(checkpoint_path)
            self.load_crosstext_checkpoint(checkpoint_path, strict=strict_load)

        # If checkpoint already provides len==300, this is a no-op.
        # If checkpoint/text model is len==77, this performs EPE 77->300.
        self._ensure_positional_length(max_text_len)

        if freeze_text_encoder:
            self.freeze_text_encoder()

    @property
    def output_dim(self) -> int:
        return self.text_model.config.hidden_size

    def freeze_text_encoder(self):
        for p in self.text_model.parameters():
            p.requires_grad = False

    def _resize_positional_embedding(self, old_weight: torch.Tensor, new_len: int) -> torch.Tensor:
        old_len, _ = old_weight.shape
        if old_len == new_len:
            return old_weight

        src = old_weight.detach()
        xs = torch.linspace(0, old_len - 1, steps=new_len, device=src.device, dtype=src.dtype)
        x0 = torch.floor(xs).long()
        x1 = torch.clamp(x0 + 1, max=old_len - 1)
        w = (xs - x0.to(src.dtype)).unsqueeze(1)
        return (1 - w) * src[x0] + w * src[x1]

    def _ensure_positional_length(self, max_text_len: int):
        emb = self.text_model.text_model.embeddings.position_embedding
        old_weight = emb.weight.data
        old_len, dim = old_weight.shape

        if old_len == max_text_len:
            self.text_model.config.max_position_embeddings = max_text_len
            return

        new_weight = self._resize_positional_embedding(old_weight, max_text_len)
        new_emb = nn.Embedding(max_text_len, dim)
        new_emb.weight.data.copy_(new_weight)

        self.text_model.text_model.embeddings.position_embedding = new_emb
        self.text_model.text_model.embeddings.register_buffer(
            "position_ids", torch.arange(max_text_len).expand((1, -1)), persistent=False
        )
        self.text_model.config.max_position_embeddings = max_text_len

    def tokenize(self, texts: Iterable[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

    def _extract_state_dict(self, ckpt_obj) -> Tuple[dict, bool]:
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"], True
        if isinstance(ckpt_obj, dict):
            return ckpt_obj, False
        raise ValueError("Unsupported checkpoint format.")

    def _normalize_key(self, key: str) -> str:
        prefixes = ["module.", "model.", "net.", "text_encoder.", "clip_model.", "clip."]
        out = key
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if out.startswith(p):
                    out = out[len(p):]
                    changed = True
        return out

    def inspect_checkpoint(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict, wrapped = self._extract_state_dict(ckpt)

        print(f"[inspect] checkpoint: {checkpoint_path}")
        print(f"[inspect] wrapped_state_dict: {wrapped}")
        if isinstance(ckpt, dict):
            print("[inspect] top-level keys:", list(ckpt.keys())[:50])

        patterns = ["token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection"]
        pos_shape = None
        for k, v in state_dict.items():
            lk = k.lower()
            is_text_key = (not lk.startswith("visual.")) and any(p in lk for p in patterns)
            if is_text_key:
                shape = tuple(v.shape) if torch.is_tensor(v) else "-"
                print(f"[inspect] {k}: {shape}")

            # Only use text positional embedding key, never visual positional embedding
            if lk == "positional_embedding" and torch.is_tensor(v) and len(v.shape) == 2:
                pos_shape = tuple(v.shape)

        if pos_shape is not None:
            print(f"[inspect] detected TEXT positional_embedding shape: {pos_shape}")
            if pos_shape[0] == 300:
                print("[inspect] text positional length is already 300.")
            elif pos_shape[0] == 77:
                print("[inspect] text positional length is 77 and will be resized to target length.")

        return pos_shape

    def load_crosstext_checkpoint(self, checkpoint_path: str, strict: bool = False):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        src_state, _ = self._extract_state_dict(ckpt)

        dst_state = self.text_model.state_dict()
        load_state = {}

        for k, v in src_state.items():
            if not torch.is_tensor(v):
                continue
            nk = self._normalize_key(k)
            lk = nk.lower()

            # Never load visual branch weights into text tower.
            if lk.startswith("visual."):
                continue

            if nk in dst_state and tuple(v.shape) == tuple(dst_state[nk].shape):
                load_state[nk] = v
                continue

            # Map OpenAI-CLIP text positional embedding ONLY from text key.
            if nk == "positional_embedding" and len(v.shape) == 2:
                target_key = "text_model.embeddings.position_embedding.weight"
                if target_key in dst_state:
                    target_shape = tuple(dst_state[target_key].shape)
                    # Skip incompatible dim (e.g., visual 1024 vs text 768)
                    if v.shape[1] != target_shape[1]:
                        continue
                    if tuple(v.shape) == target_shape:
                        load_state[target_key] = v
                    else:
                        resized = self._resize_positional_embedding(v, target_shape[0])
                        load_state[target_key] = resized

        missing, unexpected = self.text_model.load_state_dict(load_state, strict=False)
        print(f"[load_text_ckpt] loaded tensors: {len(load_state)}")
        print(f"[load_text_ckpt] missing_keys: {len(missing)}")
        print(f"[load_text_ckpt] unexpected_keys: {len(unexpected)}")

        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError("Strict checkpoint loading failed.")

    def forward(self, texts):
        if isinstance(texts, dict):
            tokenized = texts
        elif torch.is_tensor(texts):
            tokenized = {"input_ids": texts}
        else:
            tokenized = self.tokenize(texts)

        device = next(self.text_model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        if "attention_mask" not in tokenized:
            tokenized["attention_mask"] = (tokenized["input_ids"] != self.tokenizer.pad_token_id).long()

        outputs = self.text_model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask", None),
        )

        hidden = outputs.last_hidden_state
        eot_positions = tokenized["input_ids"].argmax(dim=-1)
        feat = hidden[torch.arange(hidden.shape[0], device=hidden.device), eot_positions]
        return F.normalize(feat, dim=-1)
