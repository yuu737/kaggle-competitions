"""Leak-free slope-integration geosteering solver.

Geosteering inversion: as the well advances in MD, a change in (calibrated)
horizontal GR implies a TVT move scaled by the LOCAL typewell GR gradient:

    dTVT/dMD  ~=  (dGR_h/dMD) / (dGR_tw/dTVT | at current TVT)

We integrate this from the known PS anchor. The typewell gradient is evaluated
at the RUNNING TVT ESTIMATE (never at the true TVT) -- so this uses no label
information from the lateral. Calibration of horizontal GR onto the typewell
scale is fit on the pre-PS region, where true TVT is legitimately known.

This is a foundation that can keep growing: better GR calibration, offset-well
dip priors, residual ML, confidence-weighted blending all attach on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from geosteer_dp import _build_typewell_interp


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(np.pad(x, w // 2, mode="edge"), k, mode="same")[w // 2 : w // 2 + len(x)]


def _calibrate(known: pd.DataFrame, tw_t, tw_gz):
    grk = known["GR"].values.astype(float)
    tvk = known["TVT"].values.astype(float)
    mk = np.isfinite(grk) & np.isfinite(tvk)
    if mk.sum() >= 20:
        target = np.interp(tvk[mk], tw_t, tw_gz, left=tw_gz[0], right=tw_gz[-1])
        A = np.vstack([grk[mk], np.ones(mk.sum())]).T
        a, b = np.linalg.lstsq(A, target, rcond=None)[0]
        return a, b
    s = np.nanstd(grk)
    s = s if (np.isfinite(s) and s > 1e-9) else 1.0
    return 1.0 / s, -np.nanmean(grk) / s


def solve(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    gr_smooth: float = 31,     # GR smoothing window (ft) before differentiating
    grad_floor: float = 0.02,  # ignore TVT moves where |typewell gradient| < floor
    vel_clip: float = 0.1,     # clamp |dTVT/dMD| (ft/ft); data p99 ~0.07
    vel_smooth: float = 101,   # smoothing window on the velocity estimate (ft)
    n_iter: int = 3,           # self-consistent passes (gradient at running estimate)
    shrink: float = 1.0,       # global shrink of the integrated deviation toward anchor
) -> np.ndarray:
    n = len(hw_lat)
    anchor = float(known["TVT"].values[-1])
    if n == 0:
        return np.zeros(0)

    tw_t, tw_gz = _build_typewell_interp(tw)
    if tw_t.size < 5:
        return np.full(n, anchor)
    tw_grad_curve = np.gradient(tw_gz, tw_t)

    a, b = _calibrate(known, tw_t, tw_gz)
    md = hw_lat["MD"].values.astype(float)
    gr = a * hw_lat["GR"].values.astype(float) + b
    # interpolate missing GR over index so gradients are defined
    fin = np.isfinite(gr)
    if fin.sum() < 5:
        return np.full(n, anchor)
    gr = np.interp(np.arange(n), np.where(fin)[0], gr[fin])
    gr_s = _smooth(gr, gr_smooth)
    dgr = np.gradient(gr_s, md)
    dmd = np.gradient(md)

    tvt = np.full(n, anchor)
    for _ in range(max(1, n_iter)):
        grad_tw = np.interp(tvt, tw_t, tw_grad_curve)
        grad_tw = np.where(np.abs(grad_tw) < grad_floor, np.nan, grad_tw)
        vel = dgr / grad_tw
        vel = np.clip(vel, -vel_clip, vel_clip)
        vel = np.where(np.isfinite(vel), vel, 0.0)
        vel = _smooth(vel, vel_smooth)
        dev = np.cumsum(vel * dmd)
        tvt = anchor + shrink * dev
    return tvt


def make_predictor(**kw):
    def _p(known, hw_lat, tw):
        return solve(known, hw_lat, tw, **kw)
    return _p
