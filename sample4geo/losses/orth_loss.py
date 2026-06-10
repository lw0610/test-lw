import torch
import torch.nn.functional as F


def orth_loss(semantic_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
    sem = F.normalize(semantic_emb, dim=-1)
    sty = F.normalize(style_emb, dim=-1)
    cos = (sem * sty).sum(dim=-1)
    return (cos ** 2).mean()
