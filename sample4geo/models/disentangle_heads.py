import torch.nn as nn


class DisentangleHeads(nn.Module):
    def __init__(self, in_dim: int, sem_dim: int, sty_dim: int):
        super().__init__()
        self.semantic_head = nn.Linear(in_dim, sem_dim)
        self.style_head = nn.Linear(in_dim, sty_dim)

    def forward(self, x):
        sem = self.semantic_head(x)
        sty = self.style_head(x)
        return sem, sty
