"""Anisotropic spatial geological-surface predictor.

Replaces the isotropic IDW in geosteer_spatial.py with an anisotropic kernel
that respects the dominant geological structure direction (strike).

Validated parameters (from eval_anisotropic_spatial.py LOO on 20 locked wells):
  theta_deg = 45.0   (strike direction from east, ~N45E)
  ratio     = 1.5    (along-strike / across-strike length scale)
  locked RMSE improvement: 10.863 -> 10.256  (+0.607)

Full 773-well OOF standalone RMSE: 12.90 (vs isotropic 12.77 at step=8 cloud)
Blend eval (stack+pf+aniso_sp, 154 wells): 7.685 -> 7.482 (+0.203), weight=0.30

Public interface:
  build_surface_cloud(well_ids, load_well_fn, formation, step)
  predict(known, hw_lat, XY, VAL, WID, exclude_wid, theta_deg, ratio, k)
  predict_with_defaults(known, hw_lat, XY, VAL, WID, exclude_wid)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

THETA_DEG = 45.0
RATIO = 1.5
IDW_K = 12
SEARCH_K_EXTRA = 128
TREND_N = 2000
GATE_INNER_FT = 1500.0
GATE_WIDTH_FT = 1200.0
FORM_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def _metric_xy(xy: np.ndarray, theta_deg: float, ratio: float) -> np.ndarray:
    angle = math.radians(theta_deg)
    c, s = math.cos(angle), math.sin(angle)
    parallel = xy[:, 0] * c + xy[:, 1] * s
    perp = (-xy[:, 0] * s + xy[:, 1] * c) * ratio
    return np.c_[parallel, perp]


def build_surface_cloud(load_well, well_ids, form: str = "EGFDU", step: int = 24):
    """Return (XY, values, well_tags) for a formation surface cloud."""
    xy_parts, val_parts, tag_parts = [], [], []
    for wid in well_ids:
        hw, _ = load_well(wid)
        if form not in hw.columns:
            continue
        use = np.isfinite(hw["X"].values) & np.isfinite(hw["Y"].values) & np.isfinite(hw[form].values)
        pts = np.c_[hw["X"].values[use], hw["Y"].values[use]][::step]
        vals = hw[form].values[use][::step]
        xy_parts.append(pts)
        val_parts.append(vals)
        tag_parts.append(np.full(len(pts), wid, dtype=object))
    XY = np.vstack(xy_parts)
    VAL = np.concatenate(val_parts)
    WID = np.concatenate(tag_parts)
    return XY, VAL, WID


def _other_neighbors(
    tree: cKDTree,
    query_m: np.ndarray,
    tags: np.ndarray,
    excluded: str,
    wanted: int,
    search_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(search_k, len(tags))
    dist, idx = tree.query(query_m, k=k)
    if dist.ndim == 1:
        dist, idx = dist[None, :], idx[None, :]
    dist = np.where(tags[idx] == excluded, np.inf, dist)
    chosen = np.argpartition(dist, min(wanted - 1, k - 1), axis=1)[:, :wanted]
    sel_d = np.take_along_axis(dist, chosen, axis=1)
    sel_i = np.take_along_axis(idx, chosen, axis=1)
    return sel_d, sel_i


def predict(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    XY: np.ndarray,
    VAL: np.ndarray,
    WID: np.ndarray,
    exclude_wid: str,
    theta_deg: float = THETA_DEG,
    ratio: float = RATIO,
    k: int = IDW_K,
) -> np.ndarray:
    """Anisotropic PS-anchored spatial prediction for a lateral.

    Args:
        known:       pre-PS rows with columns X, Y, Z, TVT
        hw_lat:      lateral rows with columns X, Y, Z (no TVT)
        XY:          (N, 2) cloud of formation-surface sample coordinates
        VAL:         (N,) cloud values (e.g. EGFDU depths)
        WID:         (N,) string tags identifying which well each point belongs to
        exclude_wid: well id to exclude from the cloud (the prediction well)
        theta_deg:   strike direction in degrees from east (default 45)
        ratio:       along-strike / across-strike length scale (default 1.5)
        k:           number of IDW neighbors (default 12)

    Returns:
        (L,) TVT predictions for the lateral.
    """
    n = len(hw_lat)
    if n == 0:
        return np.zeros(0)
    anchor = float(known["TVT"].values[-1])
    z_ps = float(known["Z"].values[-1])
    psxy = known[["X", "Y"]].values[-1:].astype(float)

    # Build anisotropic tree
    Xm = _metric_xy(XY, theta_deg, ratio)
    tree = cKDTree(Xm)

    self_n = int(np.count_nonzero(WID == exclude_wid))
    search_k = self_n + k + SEARCH_K_EXTRA

    # Detrend using TREND_N nearest-to-center non-self neighbors
    qxy = hw_lat[["X", "Y"]].values.astype(float)
    center = qxy.mean(axis=0, keepdims=True)
    center_m = _metric_xy(center, theta_deg, ratio)
    trend_wanted = min(TREND_N, len(XY) - self_n - 1)
    _, trend_idx = _other_neighbors(tree, center_m, WID, exclude_wid, trend_wanted, self_n + trend_wanted + SEARCH_K_EXTRA)
    trend_idx = trend_idx[0]
    design = np.c_[XY[trend_idx], np.ones(len(trend_idx))]
    coef = np.linalg.lstsq(design, VAL[trend_idx], rcond=None)[0]
    trend_all = XY @ coef[:2] + coef[2]
    residual = VAL - trend_all

    # IDW on residuals
    qm = _metric_xy(qxy, theta_deg, ratio)
    psm = _metric_xy(psxy, theta_deg, ratio)
    dist_lat, idx_lat = _other_neighbors(tree, qm, WID, exclude_wid, k, search_k)
    dist_ps, idx_ps = _other_neighbors(tree, psm, WID, exclude_wid, k, search_k)
    w_lat = 1.0 / (dist_lat ** 2 + 1.0)
    w_ps = 1.0 / (dist_ps ** 2 + 1.0)
    surf_lat = (w_lat * residual[idx_lat]).sum(1) / w_lat.sum(1) + qxy @ coef[:2] + coef[2]
    surf_ps = float((w_ps * residual[idx_ps]).sum() / w_ps.sum()) + float((psxy @ coef[:2] + coef[2])[0])

    raw = anchor - (hw_lat["Z"].values.astype(float) - z_ps) + (surf_lat - surf_ps)

    # Gate: fall back to anchor when nearest well > GATE_INNER_FT + GATE_WIDTH_FT in physical space
    physical_nearest = np.linalg.norm(XY[idx_lat] - qxy[:, None, :], axis=2).min(axis=1)
    gate = np.clip((GATE_INNER_FT - physical_nearest) / GATE_WIDTH_FT, 0.0, 1.0)
    pred = gate * raw + (1.0 - gate) * anchor

    return np.where(np.isfinite(pred), pred, anchor)


def predict_with_defaults(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    XY: np.ndarray,
    VAL: np.ndarray,
    WID: np.ndarray,
    exclude_wid: str,
) -> np.ndarray:
    """Convenience wrapper using validated default parameters."""
    return predict(known, hw_lat, XY, VAL, WID, exclude_wid, THETA_DEG, RATIO, IDW_K)
