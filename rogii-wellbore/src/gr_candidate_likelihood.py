"""Learned horizontal-GR to typewell-candidate likelihood utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def robust_normalize(x: np.ndarray) -> np.ndarray:
    median = np.nanmedian(x, axis=-1, keepdims=True)
    scale = np.nanmedian(np.abs(x - median), axis=-1, keepdims=True)
    return np.clip((x - median) / (1.4826 * scale + 3.0), -5.0, 5.0)


class CandidateScorer(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def horizontal_windows(gr: np.ndarray, width: int = 17) -> np.ndarray:
    """Centered windows with edge padding; non-finite values are interpolated."""
    gr = np.asarray(gr, dtype=float)
    finite = np.isfinite(gr)
    if not finite.any():
        gr = np.zeros_like(gr)
    elif not finite.all():
        at = np.arange(len(gr))
        gr = np.interp(at, at[finite], gr[finite])
    half = width // 2
    padded = np.pad(gr, (half, half), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, width).copy()


def score_likelihood_grid(model: nn.Module, horizontal_gr: np.ndarray,
                          type_tvt: np.ndarray, type_gr: np.ndarray,
                          spacing: float = 2.0, width: int = 17,
                          time_batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate TVT grid, centered logits[T,K]) on CPU."""
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    tvt = np.asarray(type_tvt, float)[finite]
    gr = np.asarray(type_gr, float)[finite]
    order = np.argsort(tvt)
    tvt, gr = tvt[order], gr[order]
    half_span = 8.0
    grid = np.arange(np.ceil((tvt.min() + half_span) / spacing) * spacing,
                     np.floor((tvt.max() - half_span) / spacing) * spacing + 0.1,
                     spacing)
    patch_offsets = np.linspace(-half_span, half_span, width)
    type_patch = np.stack([np.interp(center + patch_offsets, tvt, gr) for center in grid])
    type_norm = robust_normalize(type_patch).astype(np.float32)
    horizontal_norm = robust_normalize(horizontal_windows(horizontal_gr, width)).astype(np.float32)
    output = np.empty((len(horizontal_norm), len(grid)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(horizontal_norm), time_batch):
            h = horizontal_norm[start:start + time_batch]
            repeated_h = np.broadcast_to(h[:, None, :],
                                         (len(h), len(grid), width))
            repeated_t = np.broadcast_to(type_norm[None, :, :],
                                         (len(h), len(grid), width))
            correlation = np.mean(repeated_h * repeated_t, axis=2, keepdims=True)
            features = np.concatenate(
                [repeated_h, repeated_t, repeated_h - repeated_t, correlation], axis=2
            ).astype(np.float32)
            logits = model(torch.from_numpy(features.reshape(-1, features.shape[-1])))
            block = logits.numpy().reshape(len(h), len(grid))
            output[start:start + len(h)] = block - block.max(axis=1, keepdims=True)
    return grid, output
