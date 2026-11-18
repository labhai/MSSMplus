"""Supervertex Vision Transformer (SV-ViT).

SV-ViT uses SiT as the backbone with SSVM tokenization. This module enforces
the SSVM input contract: 2,560 supervertices with 441 retained vertex
occurrences per supervertex. No patch-wise averaging is performed here.
"""

from __future__ import annotations

from torch import Tensor, nn

from .sit import SiT


class SVViT(nn.Module):
    """Validate SSVM tokens and delegate the Transformer computation to SiT."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        num_patches: int = 2560,
        num_classes: int = 1,
        num_channels: int = 1,
        num_vertices: int = 441,
        dim_head: int = 16,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pool != "cls":
            raise ValueError("SV-ViT requires CLS pooling; set pool='cls'.")
        if min(num_patches, num_channels, num_vertices) < 1:
            raise ValueError(
                "num_patches, num_channels, and num_vertices must be positive."
            )

        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_vertices = num_vertices
        self.sit = SiT(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            num_patches=num_patches,
            num_classes=num_classes,
            num_channels=num_channels,
            num_vertices=num_vertices,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

    def forward(self, surface: Tensor) -> Tensor:
        expected = (
            self.num_channels,
            self.num_patches,
            self.num_vertices,
        )
        if surface.ndim != 4 or tuple(surface.shape[1:]) != expected:
            raise ValueError(
                "Expected SV-ViT input shape "
                f"(batch, {expected[0]}, {expected[1]}, {expected[2]}), "
                f"got {tuple(surface.shape)}."
            )
        return self.sit(surface)
