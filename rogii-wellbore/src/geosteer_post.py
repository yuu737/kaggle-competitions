"""Robust low-order PROJECTION post-processing (ported from 7.159 cell_20).

Per well, the structural position U = TVT + Z is a smooth function of along-lateral
position. We fit a robust (IRLS) low-degree polynomial of dU = (pred + Z) - anchor
against normalized MD, then blend the fit back with the raw prediction. This
denoises tracker jitter and down-weights wrong-branch outliers. Deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def robfit(s, y, deg=4, iters=4):
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


def project(known: pd.DataFrame, hw_lat: pd.DataFrame, pred: np.ndarray,
            deg: int = 4, blend: float = 0.75) -> np.ndarray:
    """Return projected prediction for the lateral. `blend` = weight on the fit."""
    n = len(hw_lat)
    if n < deg + 2:
        return pred
    anchor = float(known["TVT"].values[-1]) + float(known["Z"].values[-1])
    ps = float(known["MD"].values[-1])
    md = hw_lat["MD"].values.astype(float)
    Z = hw_lat["Z"].values.astype(float)
    end = float(md[-1])
    s = (md - ps) / max(end - ps, 1e-6)
    fit = robfit(s, (pred + Z) - anchor, deg)
    proj_full = (anchor + fit) - Z
    out = (1.0 - blend) * pred + blend * proj_full
    if not np.isfinite(out).all():
        return pred
    return out
