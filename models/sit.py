"""Surface Vision Transformer (SiT).

Adapted from the public SiT implementation by Dahan et al. (2022):
https://github.com/metrics-lab/surface-vision-transformers

SV-ViT changes the surface tokenization (SSVM), not the Transformer backbone.
The input is a tensor with shape ``(batch, channels, patches, vertices)``.
"""

from __future__ import annotations

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from vit_pytorch.vit import Transformer


class SiT(nn.Module):
    """Vision Transformer adapted to a sequence of cortical-surface patches."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        num_patches: int = 20,
        num_classes: int = 1,
        num_channels: int = 4,
        num_vertices: int = 2145,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be either 'cls' or 'mean'.")

        patch_dim = num_channels * num_vertices
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c n v -> b n (v c)"),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, surface: Tensor) -> Tensor:
        tokens = self.to_patch_embedding(surface)
        batch, num_patches, _ = tokens.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=batch)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embedding[:, : num_patches + 1]
        tokens = self.dropout(tokens)
        tokens = self.transformer(tokens)
        tokens = tokens.mean(dim=1) if self.pool == "mean" else tokens[:, 0]
        return self.mlp_head(self.to_latent(tokens))
