"""Beam-search geosteering tracker (ported from the existing LB 7.159 solution).

A second, methodologically different tracker to pair with the particle filter:
it walks an index into the typewell TVT axis, at each lateral step moving by
{-2,-1,0,+1,+2} cells under a GR-match cost plus a move cost, keeping a beam of
the lowest-cost paths. An ensemble of 14 (beam_size, move_cost, err_scale, smooth)
configs is averaged. Adapted to the local CV harness (known / lateral / typewell)
and leak-free: only the anchor (last known TVT) and lateral GR are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# (beam_size, move_cost, err_scale, savgol_radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2., 1., 0., 1., 2.])

    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1
    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)

        ni_f = ni.flatten()[valid.flatten()]
        tot_f = tot.flatten()[valid.flatten()]
        order = np.argsort(tot_f)
        ni_s = ni_f[order]
        tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


def solve(known: pd.DataFrame, hw_lat: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    n = len(hw_lat)
    anchor = float(known["TVT"].values[-1])
    if n == 0:
        return np.zeros(0)
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    if tw_tvt.size < 5 or not np.isfinite(tw_gr).any():
        return np.full(n, anchor)

    # interpolate lateral GR
    gr = pd.Series(hw_lat["GR"].values.astype(float)).interpolate(limit_direction="both")
    hgr = gr.fillna(np.nanmean(tw_gr)).values.astype(float)

    results = [beam_search(hgr, tw_tvt, tw_gr, anchor, bs, mc, es, r)
               for (bs, mc, es, r) in BEAM_CONFIGS]
    return np.stack(results, 0).mean(0)


def make_predictor():
    def _p(known, hw_lat, tw):
        return solve(known, hw_lat, tw)
    return _p
