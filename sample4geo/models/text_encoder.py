import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPTextEncoder(nn.Module):
    """
    Minimal CLIP-like text encoder with Expanded Positional Embedding (EPE)
    via 1D linear interpolation from length 77 -> max_text_len.

    Note: this is a lightweight implementation to keep Sample4Geo changes minimal
    and runnable without introducing a heavy external dependency chain.
    """

    def __init__(self, vocab_size: int = 32000, width: int = 512, max_text_len: int = 300, base_len: int = 77):
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.max_text_len = max_text_len
        self.base_len = base_len

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Parameter(torch.randn(base_len, width) * 0.01)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=width, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.ln_final = nn.LayerNorm(width)

        epe = self._build_epe_position_embedding(base_len, max_text_len)
        self.register_buffer("position_embedding_epe", epe, persistent=False)

    def _build_epe_position_embedding(self, base_len: int, new_len: int) -> torch.Tensor:
        # 1D linear interpolation on positions
        # x in [0, base_len-1], sampled at new_len points
        if new_len == base_len:
            return self.position_embedding.detach().clone()

        device = self.position_embedding.device
        dtype = self.position_embedding.dtype

        src = self.position_embedding.detach()
        xs = torch.linspace(0, base_len - 1, steps=new_len, device=device, dtype=dtype)

        x0 = torch.floor(xs).long()
        x1 = torch.clamp(x0 + 1, max=base_len - 1)
        w = (xs - x0.to(dtype)).unsqueeze(1)

        p0 = src[x0]
        p1 = src[x1]
        return (1 - w) * p0 + w * p1

    def refresh_epe(self):
        self.position_embedding_epe = self._build_epe_position_embedding(self.base_len, self.max_text_len)

    def tokenize(self, texts):
        # minimal tokenizer: whitespace hash to vocab ids
        # keeps implementation dependency-free for minimal runnable version
        token_ids = []
        for t in texts:
            words = t.strip().split()
            ids = [1]  # BOS
            for w in words:
                ids.append((abs(hash(w)) % (self.vocab_size - 3)) + 3)
            ids.append(2)  # EOS
            ids = ids[: self.max_text_len]
            if len(ids) < self.max_text_len:
                ids = ids + [0] * (self.max_text_len - len(ids))
            token_ids.append(ids)
        return torch.tensor(token_ids, dtype=torch.long)

    def forward(self, texts):
        if torch.is_tensor(texts):
            token_ids = texts.to(self.position_embedding.device)
        else:
            token_ids = self.tokenize(texts).to(self.position_embedding.device)
        x = self.token_embedding(token_ids)

        if self.position_embedding_epe.device != x.device:
            self.refresh_epe()
            self.position_embedding_epe = self.position_embedding_epe.to(x.device)

        pos = self.position_embedding_epe[: x.shape[1]].unsqueeze(0)
        x = x + pos

        x = self.transformer(x)
        x = self.ln_final(x)

        # use EOS token position if present else mean pooling
        eos_id = 2
        eos_mask = token_ids.eq(eos_id)
        feats = []
        for i in range(x.shape[0]):
            idx = torch.where(eos_mask[i])[0]
            if len(idx) > 0:
                feats.append(x[i, idx[0]])
            else:
                feats.append(x[i].mean(dim=0))
        feats = torch.stack(feats, dim=0)
        return F.normalize(feats, dim=-1)
