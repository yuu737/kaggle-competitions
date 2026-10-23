"""Spatial geological-surface predictor (offset-well structural model).

Key fact (verified on train): TVT is *exactly* determined by the formation-top
surface depth at the well location:  TVT = ref_tvt - (Z - formation_topZ) + const
(reconstruction RMSE 0.005 ft with the true tops).

The formation surfaces vary smoothly in (X,Y) and are densely sampled by the other
wells. So we predict the surface change along the lateral by spatial interpolation
of OTHER wells' formation-top samples, anchored at the PS point:

    TVT_lat = TVT_PS - (Z_lat - Z_PS) + (surf_lat - surf_PS)

This is independent of (orthogonal to) the GR-based particle filter, which makes it
a powerful blend partner. Leak-free: uses other wells' surfaces + this well's known
anchor only. In the real test the surface cloud is built from the 773 train wells
(which DO carry formation tops); test wells are queried by (X,Y).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FORM_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def build_surface_cloud(load_well, well_ids, form="EGFDU", step=8):
    """Return (KDTree on XY, values, well_tag) for a formation surface."""
    XY, VAL, WID = [], [], []
    for wid in well_ids:
        hw, _ = load_well(wid)
        if form not in hw:
            continue
        x = hw["X"].values; y = hw["Y"].values; v = hw[form].values
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        XY.append(np.c_[x[m], y[m]][::step])
        VAL.append(v[m][::step])
        WID.append(np.full(((m.sum()) + step - 1) // step, wid, dtype=object))
    XY = np.vstack(XY); VAL = np.concatenate(VAL); WID = np.concatenate(WID)
    return XY, VAL, WID


def _idw(tree, vals, qxy, k=12):
    d, idx = tree.query(qxy, k=k)
    if k == 1:
        return vals[idx]
    w = 1.0 / (d ** 2 + 1.0)
    return (w * vals[idx]).sum(1) / w.sum(1)


def predict(known: pd.DataFrame, hw_lat: pd.DataFrame, XY, VAL, WID, exclude_wid,
            k=12) -> np.ndarray:
    """PS-anchored spatial surface prediction for the lateral."""
    n = len(hw_lat)
    anchor = float(known["TVT"].values[-1])
    if n == 0:
        return np.zeros(0)
    mask = WID != exclude_wid
    tree = cKDTree(XY[mask]); vals = VAL[mask]
    qlat = np.c_[hw_lat["X"].values, hw_lat["Y"].values]
    surf_lat = _idw(tree, vals, qlat, k=k)
    surf_ps = _idw(tree, vals, np.c_[[known["X"].values[-1]], [known["Y"].values[-1]]], k=k)[0]
    z_ps = float(known["Z"].values[-1])
    pred = anchor - (hw_lat["Z"].values.astype(float) - z_ps) + (surf_lat - surf_ps)
    if not np.isfinite(pred).all():
        pred = np.where(np.isfinite(pred), pred, anchor)
    return pred
