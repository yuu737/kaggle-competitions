"""Offline HMM smoother for geosteering.

Unlike the forward-only particle filter, this uses the whole lateral GR sequence
at prediction time.  The hidden state is structural position U = TVT + Z on a
fixed grid, with a smooth random-walk transition.  Observations are typewell-GR
match likelihoods at TVT = U - Z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gr_scale(known: pd.DataFrame, tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    tw_at_k = np.interp(known["TVT"].values.astype(float), tw_tvt, tw_gr)
    return float(np.clip(np.nanstd(known["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))


def _initial_rate(known: pd.DataFrame) -> float:
    tail = known.tail(30)
    dt = np.diff(tail["TVT"].values.astype(float))
    dz = np.diff(tail["Z"].values.astype(float))
    dm = np.diff(tail["MD"].values.astype(float))
    m = dm > 0
    return float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0


def _kernel(sigma_cells: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma_cells, 1e-6)) ** 2)
    return k / k.sum()


def _shift_convolve(p: np.ndarray, shift_cells: int, kernel: np.ndarray) -> np.ndarray:
    """Apply p_next[j] = sum_i p[i] K[j - i - shift]."""
    conv = np.convolve(p, kernel, mode="same")
    out = np.zeros_like(conv)
    if shift_cells > 0:
        out[shift_cells:] = conv[:-shift_cells]
    elif shift_cells < 0:
        out[:shift_cells] = conv[-shift_cells:]
    else:
        out = conv
    s = out.sum()
    return out / s if s > 0 else np.ones_like(out) / len(out)


def _posterior_quantile(grid: np.ndarray, probs: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    idx = (cdf < q).sum(axis=1)
    idx = np.clip(idx, 0, len(grid) - 1)
    return grid[idx]


def hmm_smoother(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    tw: pd.DataFrame,
    grid_radius: float = 90.0,
    grid_step: float = 0.5,
    trans_sigma: float = 0.35,
    trans_radius: float = 4.0,
    init_sigma: float = 4.5,
    obs_temp: float = 1.0,
    output: str = "mean",
):
    """Return an offline-smoothed lateral TVT prediction.

    output:
      mean   : posterior mean of U, converted to TVT
      median : posterior median of U, converted to TVT
      map    : per-step posterior mode of U, converted to TVT
    """
    if len(hw_lat) == 0:
        return np.zeros(0)

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    last = known.iloc[-1]
    anchor_u = float(last["TVT"]) + float(last["Z"])
    last_md = float(last["MD"])
    ir = _initial_rate(known)
    gs = _gr_scale(known, tw_tvt, tw_gr)

    grid = np.arange(anchor_u - grid_radius, anchor_u + grid_radius + grid_step * 0.5, grid_step)
    n_state = len(grid)
    md_v = hw_lat["MD"].values.astype(float)
    z_v = hw_lat["Z"].values.astype(float)
    gr_v = hw_lat["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    n = len(hw_lat)

    obs = np.empty((n, n_state), dtype=np.float64)
    for i in range(n):
        tvt_grid = np.clip(grid - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        eg = np.interp(tvt_grid, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / (gs * max(obs_temp, 1e-6))
        row = np.exp(-0.5 * np.minimum(d * d, 600.0))
        row = np.maximum(row, 1e-300)
        obs[i] = row / row.sum()

    prior = np.exp(-0.5 * ((grid - anchor_u) / init_sigma) ** 2)
    prior /= prior.sum()

    alpha = np.empty_like(obs)
    shifts = np.empty(n, dtype=np.int32)
    kernels = []
    prev_md = last_md
    p = prior
    for i in range(n):
        dm = max(md_v[i] - prev_md, 1.0)
        shift = int(round((ir * dm) / grid_step))
        sigma_cells = max((trans_sigma * np.sqrt(dm)) / grid_step, 0.75)
        radius = max(2, int(round(trans_radius * sigma_cells)))
        k = _kernel(sigma_cells, radius)
        p = _shift_convolve(p, shift, k)
        p = p * obs[i]
        p = p / p.sum() if p.sum() > 0 else np.ones(n_state) / n_state
        alpha[i] = p
        shifts[i] = shift
        kernels.append(k)
        prev_md = md_v[i]

    beta = np.ones_like(obs)
    b = np.ones(n_state) / n_state
    for i in range(n - 1, -1, -1):
        gamma = obs[i] * b
        gamma = gamma / gamma.sum() if gamma.sum() > 0 else np.ones(n_state) / n_state
        b = _shift_convolve(gamma, -int(shifts[i]), kernels[i])
        beta[i] = b

    post = alpha * beta
    post /= post.sum(axis=1, keepdims=True)

    if output == "median":
        u = _posterior_quantile(grid, post, 0.5)
    elif output == "map":
        u = grid[np.argmax(post, axis=1)]
    else:
        u = post @ grid
    pred = u - z_v
    if not np.isfinite(pred).all():
        return np.full(n, float(last["TVT"]))
    return pred


def make_predictor(**kw):
    def _p(known, hw_lat, tw):
        return hmm_smoother(known, hw_lat, tw, **kw)

    return _p
