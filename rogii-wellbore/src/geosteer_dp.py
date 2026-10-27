"""In-house GR -> TVT geosteering solver (Viterbi DP).

Idea (competition physics): horizontal GR(MD) projected onto the TVT axis must
match the typewell GR(TVT) curve. We recover TVT(MD) over the lateral as a slow,
smooth path that minimizes GR mismatch against the typewell, anchored on the last
known TVT at the PS point.

Calibration: horizontal and typewell GR have different baseline/scale (slide 9).
We standardize both to z-scores. When the known pre-PS region overlaps the
typewell TVT range we could refine calibration there; v1 uses global z-scores.

Honest by construction: the solver only sees test-visible lateral columns plus
the pre-PS anchor; it never reads the lateral's true TVT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-9:
        s = 1.0
    return (x - m) / s


def _build_typewell_interp(tw: pd.DataFrame):
    """Return (grid_tvt, grid_gr_z) sorted/deduplicated for fast lookup."""
    t = tw["TVT"].values.astype(float)
    g = tw["GR"].values.astype(float)
    ok = np.isfinite(t) & np.isfinite(g)
    t, g = t[ok], g[ok]
    order = np.argsort(t)
    t, g = t[order], g[order]
    # collapse duplicate TVT by averaging
    uniq_t, inv = np.unique(t, return_inverse=True)
    uniq_g = np.zeros_like(uniq_t)
    cnt = np.zeros_like(uniq_t)
    np.add.at(uniq_g, inv, g)
    np.add.at(cnt, inv, 1.0)
    uniq_g /= np.maximum(cnt, 1)
    gz = _zscore(uniq_g)
    return uniq_t, gz


def solve(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    decim: int = 5,          # decimation step (ft) for the DP sequence
    grid_radius: float = 80, # +/- TVT search range around anchor (ft)
    grid_dt: float = 0.5,    # TVT grid resolution (ft)
    vmax: float = 0.12,      # max |dTVT/dMD| (ft/ft); from data p99 ~0.07
    smooth: float = 0.15,    # transition penalty weight on (delta TVT)^2
    match_w: float = 1.0,    # GR mismatch weight
    anchor_pull: float = 0.0,  # per-block prior pulling TVT toward anchor (x offset^2)
) -> np.ndarray:
    n = len(hw_lat)
    anchor = float(known["TVT"].values[-1])
    if n == 0:
        return np.zeros(0)

    tw_t, tw_gz = _build_typewell_interp(tw)
    if tw_t.size < 5:
        return np.full(n, anchor)

    md = hw_lat["MD"].values.astype(float)
    gr = hw_lat["GR"].values.astype(float)
    # Calibrate horizontal GR onto the typewell z-scale using the KNOWN pre-PS
    # region, where both horizontal GR and true TVT are available:
    #   fit a, b so that a*GR_h + b ~= typewell_gz(true TVT).
    # This makes the emission cost (calibrated GR_h - typewell_gz(state))^2 meaningful.
    grk = known["GR"].values.astype(float)
    tvk = known["TVT"].values.astype(float)
    mk = np.isfinite(grk) & np.isfinite(tvk)
    if mk.sum() >= 20:
        target = np.interp(tvk[mk], tw_t, tw_gz, left=tw_gz[0], right=tw_gz[-1])
        A = np.vstack([grk[mk], np.ones(mk.sum())]).T
        a, b = np.linalg.lstsq(A, target, rcond=None)[0]
    else:
        gr_ref = np.concatenate([grk, gr])
        m, s = np.nanmean(gr_ref), np.nanstd(gr_ref)
        s = s if (np.isfinite(s) and s > 1e-9) else 1.0
        a, b = 1.0 / s, -m / s
    gr_z = a * gr + b

    # Decimate the lateral into blocks; robust GR per block.
    idx_blocks = [np.arange(i, min(i + decim, n)) for i in range(0, n, decim)]
    blk_md = np.array([md[b].mean() for b in idx_blocks])
    blk_gr = np.array([np.nanmedian(gr_z[b]) if np.isfinite(gr_z[b]).any() else np.nan for b in idx_blocks])
    nb = len(idx_blocks)

    # TVT state grid around anchor.
    offsets = np.arange(-grid_radius, grid_radius + grid_dt, grid_dt)
    states = anchor + offsets
    ns = states.size
    # Precompute typewell GR z at each state via interpolation.
    g_at_state = np.interp(states, tw_t, tw_gz, left=tw_gz[0], right=tw_gz[-1])

    # Emission cost per block x state.
    emit = np.zeros((nb, ns))
    prior = anchor_pull * (offsets ** 2)
    for j in range(nb):
        if np.isfinite(blk_gr[j]):
            emit[j] = match_w * (blk_gr[j] - g_at_state) ** 2 + prior
        else:
            emit[j] = prior  # missing GR -> rely on smoothness + prior only

    # Transition band per block (depends on MD gap).
    dmd = np.diff(blk_md, prepend=blk_md[0])
    # Viterbi forward.
    INF = 1e18
    dp = np.full(ns, INF)
    # start near anchor (offset 0), soft prior
    start_pen = 0.02 * (offsets ** 2)
    dp = emit[0] + start_pen
    back = np.zeros((nb, ns), dtype=np.int32)
    for j in range(1, nb):
        max_jump = max(grid_dt, vmax * abs(dmd[j]))
        band = int(np.ceil(max_jump / grid_dt))
        prev = dp
        cur = np.full(ns, INF)
        bk = np.zeros(ns, dtype=np.int32)
        for d in range(-band, band + 1):
            shifted = np.full(ns, INF)
            if d >= 0:
                shifted[d:] = prev[: ns - d] if d > 0 else prev
                src_idx = np.arange(ns) - d
            else:
                shifted[:d] = prev[-d:]
                src_idx = np.arange(ns) - d
            cand = shifted + smooth * (d * grid_dt) ** 2
            take = cand < cur
            cur[take] = cand[take]
            bk[take] = src_idx[take]
        dp = cur + emit[j]
        back[j] = bk

    # Backtrack.
    path = np.zeros(nb, dtype=np.int32)
    path[-1] = int(np.argmin(dp))
    for j in range(nb - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    blk_tvt = states[path]

    # Interpolate block TVT back to full lateral, anchored at start.
    full = np.interp(md, blk_md, blk_tvt)
    return full


def make_predictor(**kw):
    def _p(known, hw_lat, tw):
        return solve(known, hw_lat, tw, **kw)
    return _p
