"""Neighbor-lateral wiggle transfer component (H2, 2026-07-05).

Mechanism: train laterals carry true TVT; U = TVT + Z detrended by a per-well OLS
line (vs MD) leaves the within-well structural residual ("wiggle"). That wiggle is
spatially coherent across wells (corr 0.78 @100-250 ft, 0.62 @250-500, 0.52 @500-750,
0.47 @750-1000; artifacts/wiggle_coherence_gate/report.json) and is DATUM-FREE, so it
transfers without the per-well registration constants that cap level interpolation.

Deployment form: pred' = pred + alpha * predict_wiggle(bank, x, y, exclude=wid).
The bank is built from train wells only; at submission rerun the same 773 train wells
are available, so local LOO evaluation matches deployment exactly.
"""
from __future__ import annotations

import numpy as np

MIN_LAT_ROWS = 300
MAX_PAIR_DIST = 2000.0
IDW_DIST = 1500.0
IDW_K = 8
DUP_DIST = 60.0
IDW_SOFT = 100.0


def build_wiggle_bank(load_well, split_known_lateral, wids):
    """Build per-well wiggle arrays + KD trees. Callables injected for portability."""
    from scipy.spatial import cKDTree

    bank = []
    for wid in wids:
        try:
            hw, _ = load_well(wid)
            known, lat, ps = split_known_lateral(hw)
            if len(lat) < MIN_LAT_ROWS:
                continue
            md = lat["MD"].to_numpy(float)
            x = lat["X"].to_numpy(float)
            y = lat["Y"].to_numpy(float)
            z = lat["Z"].to_numpy(float)
            tvt = lat["TVT"].to_numpy(float)
            m = (np.isfinite(md) & np.isfinite(x) & np.isfinite(y)
                 & np.isfinite(z) & np.isfinite(tvt))
            if m.sum() < MIN_LAT_ROWS:
                continue
            md, x, y, z, tvt = md[m], x[m], y[m], z[m], tvt[m]
            u = tvt + z
            c = np.polyfit(md - md[0], u, 1)
            resid = u - np.polyval(c, md - md[0])
            bank.append({
                "wid": wid,
                "tree": cKDTree(np.column_stack([x, y])),
                "resid": resid,
                "bbox": (float(x.min()), float(x.max()), float(y.min()), float(y.max())),
            })
        except Exception:
            continue
    return bank


def predict_wiggle(bank, x, y, exclude_wid=None):
    """IDW neighbor wiggle at query points. Returns (wig, nn_dist); wig=0 where no support."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    ax0, ax1 = pts[:, 0].min(), pts[:, 0].max()
    ay0, ay1 = pts[:, 1].min(), pts[:, 1].max()
    cand = []
    for b in bank:
        if b["wid"] == exclude_wid:
            continue
        bx0, bx1, by0, by1 = b["bbox"]
        if bx0 - MAX_PAIR_DIST > ax1 or bx1 + MAX_PAIR_DIST < ax0:
            continue
        if by0 - MAX_PAIR_DIST > ay1 or by1 + MAX_PAIR_DIST < ay0:
            continue
        cand.append(b)
    n = len(pts)
    if not cand:
        return np.zeros(n), np.full(n, np.inf)
    dmat = np.full((n, len(cand)), np.inf)
    rmat = np.zeros((n, len(cand)))
    for jc, b in enumerate(cand):
        d, idx = b["tree"].query(pts, k=1)
        ok = d <= MAX_PAIR_DIST
        if ok.sum() >= 30 and np.median(d[ok]) < DUP_DIST:
            continue  # suspected duplicate well: skip entirely
        dmat[:, jc] = d
        rmat[:, jc] = b["resid"][idx]
    order = np.argsort(dmat, axis=1)[:, :IDW_K]
    ds = np.take_along_axis(dmat, order, axis=1)
    rs = np.take_along_axis(rmat, order, axis=1)
    wgt = 1.0 / (ds ** 2 + IDW_SOFT)
    wgt[ds > IDW_DIST] = 0.0
    sw = wgt.sum(axis=1)
    wig = np.zeros(n)
    has = sw > 0
    wig[has] = (wgt[has] * rs[has]).sum(axis=1) / sw[has]
    nn = np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.inf)
    return wig, nn
