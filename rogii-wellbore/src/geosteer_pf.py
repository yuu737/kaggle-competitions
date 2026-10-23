"""Particle-filter geosteering (ported from the existing LB 7.159 solution).

This is the principled, growable engine: a sequential Monte Carlo tracker whose
state is structural position `pos = TVT + Z` plus a structural rate. At each MD
step it predicts forward, then weights particles by how well the horizontal GR
matches the typewell GR at the particle's TVT. The particle cloud naturally
handles the multi-modal typewell-GR ambiguity that defeats a single-path DP.

Ported faithfully from `inspect_cell_04.py::run_particle_filter` and adapted to
the local CV harness interface (known / lateral / typewell DataFrames). No label
information from the lateral is used: only horizontal GR, MD, Z and the typewell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_particle_filter(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 500,
    seed: int = 42,
    mom: float = 0.998,
    vn: float = 0.002,
    pn: float = 0.005,
    rp: float = 0.1,
    rr: float = 0.001,
    resamp: float = 0.5,
    init_pos_std: float = 4.5,
    init_rate_std: float = 0.01,
    adaptive_noise: bool = False,
    adaptive_floor: float = 0.25,
    adaptive_window: int = 41,
    proposal_frac: float = 0.0,
    proposal_band: float = 90.0,
    proposal_prior: float = 20.0,
    proposal_rate_jitter: float = 0.004,
    aux_tvt_grid: np.ndarray | None = None,
    aux_loglik: np.ndarray | None = None,
    aux_alpha: float = 0.0,
    return_modes: bool = False,
    mode_k: int = 3,
    mode_bin_ft: float = 2.0,
):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    if len(hw_lat) == 0:
        return np.zeros(0), 0.0

    last = known.iloc[-1]
    last_tvt = float(last["TVT"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    # GR-mismatch scale from the known region.
    tw_at_k = np.interp(known["TVT"].values.astype(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(known["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    # Initial structural rate from the last 30 known points.
    tail = known.tail(30)
    dt = np.diff(tail["TVT"].values.astype(float))
    dz = np.diff(tail["Z"].values.astype(float))
    dm = np.diff(tail["MD"].values.astype(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + init_pos_std * rng.standard_normal(N)
    rate = ir + init_rate_std * rng.standard_normal(N)
    w = np.ones(N) / N

    MOM = mom; VN = vn; PN = pn; RP = rp; RR = rr; RESAMP = resamp

    md_v = hw_lat["MD"].values.astype(float)
    z_v = hw_lat["Z"].values.astype(float)
    # interpolate GR gaps across the full well order, then take lateral values
    gr_v = hw_lat["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    if aux_alpha != 0.0:
        if aux_tvt_grid is None or aux_loglik is None:
            raise ValueError("aux_tvt_grid and aux_loglik are required when aux_alpha != 0")
        aux_tvt_grid = np.asarray(aux_tvt_grid, dtype=float)
        aux_loglik = np.asarray(aux_loglik, dtype=float)
        if aux_loglik.shape != (len(hw_lat), len(aux_tvt_grid)):
            raise ValueError(
                f"aux_loglik shape {aux_loglik.shape} != "
                f"({len(hw_lat)}, {len(aux_tvt_grid)})"
            )
    if adaptive_noise:
        gr_s = pd.Series(gr_v)
        half = max(2, int(adaptive_window) // 2)
        local_amp = (
            gr_s.rolling(2 * half + 1, center=True, min_periods=3)
            .std()
            .interpolate(limit_direction="both")
            .fillna(float(np.nanstd(gr_v)))
            .values.astype(float)
        )
        denom = max(float(np.nanpercentile(local_amp, 75)), 1e-6)
        noise_scale_v = np.clip(local_amp / denom, adaptive_floor, 1.0)
    else:
        noise_scale_v = None

    res = np.empty(len(hw_lat))
    mode_trace = np.empty((len(hw_lat), mode_k), dtype=float) if return_modes else None
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(hw_lat)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        ns = float(noise_scale_v[i]) if noise_scale_v is not None else 1.0
        rate = MOM * rate + (VN * ns) * rng.standard_normal(N)
        pos = pos + rate * dm_step + (PN * ns) * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        if proposal_frac > 0:
            n_prop = int(round(N * proposal_frac))
            if n_prop > 0:
                center = float(np.dot(w, tvt_p))
                band = np.abs(tw_tvt - center) <= proposal_band
                if band.sum() >= 3:
                    cand_tvt = tw_tvt[band]
                    cand_gr = tw_gr[band]
                    d_cand = (gr_v[i] - cand_gr) / gs
                    p_cand = np.exp(-0.5 * np.minimum(d_cand ** 2, 600.0))
                    p_cand *= np.exp(-0.5 * ((cand_tvt - center) / max(proposal_prior, 1e-6)) ** 2)
                    p_sum = p_cand.sum()
                    if p_sum > 0:
                        p_cand /= p_sum
                        repl = rng.choice(N, size=n_prop, replace=False)
                        samp = rng.choice(len(cand_tvt), size=n_prop, replace=True, p=p_cand)
                        tvt_p[repl] = cand_tvt[samp] + 0.25 * rng.standard_normal(n_prop)
                        pos[repl] = tvt_p[repl] + z_v[i]
                        rate[repl] = rate[repl] + proposal_rate_jitter * rng.standard_normal(n_prop)

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d ** 2, 600.0))
        if aux_alpha != 0.0:
            learned = np.interp(tvt_p, aux_tvt_grid, aux_loglik[i], left=0.0, right=0.0)
            lk *= np.exp(np.clip(aux_alpha * learned, -80.0, 0.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        if return_modes:
            lo = float(tw_tvt[0] - 100.0)
            bins = np.floor((tvt_p - lo) / mode_bin_ft).astype(int)
            mass = np.bincount(bins, weights=w)
            ranked = np.argsort(mass)[::-1]
            selected = []
            for candidate_bin in ranked:
                if mass[candidate_bin] <= 0:
                    break
                if all(abs(int(candidate_bin) - old) >= 2 for old in selected):
                    selected.append(int(candidate_bin))
                if len(selected) == mode_k:
                    break
            mean_tvt = float(np.dot(w, tvt_p))
            for j in range(mode_k):
                if j >= len(selected):
                    mode_trace[i, j] = mean_tvt
                    continue
                use = bins == selected[j]
                local_weight = w[use]
                mode_trace[i, j] = (
                    float(np.dot(local_weight, tvt_p[use]) / local_weight.sum())
                    if local_weight.sum() > 0 else mean_tvt
                )

        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + (RP * ns) * rng.standard_normal(N)
            rate = rate[idx] + (RR * ns) * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    if return_modes:
        return res, log_lik, mode_trace
    return res, log_lik


def pf_lik_ensemble(
    known,
    hw_lat,
    tw,
    n_particles: int = 500,
    n_seeds: int = 32,
    scale: float = 5.0,
    **pf_kw,
):
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = run_particle_filter(known, hw_lat, tw, n_particles=n_particles, seed=s, **pf_kw)
        preds.append(p)
        liks.append(ll)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def pf_ensemble_with_diag(
    known,
    hw_lat,
    tw,
    n_particles: int = 500,
    n_seeds: int = 16,
    scale: float = 5.0,
    **pf_kw,
):
    """Return (pred, diag) where diag holds leak-free confidence signals.

    diag keys:
      ens_std_mean : mean across-seed std of the prediction (disagreement)
      dev_absmean  : mean |pred - anchor| (how far the tracker wandered)
      gs           : known-region GR-mismatch scale (per-well noise level)
      loglik_mean  : mean per-seed total log-likelihood (fit quality)
    """
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = run_particle_filter(known, hw_lat, tw, n_particles=n_particles, seed=s, **pf_kw)
        preds.append(p)
        liks.append(ll)
    P = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()
    pred = (weights[:, None] * P).sum(0)

    anchor = float(known["TVT"].values[-1])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    tw_at_k = np.interp(known["TVT"].values.astype(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(known["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    diag = {
        "ens_std_mean": float(np.mean(P.std(0))),
        "dev_absmean": float(np.mean(np.abs(pred - anchor))),
        "gs": gs,
        "loglik_mean": float(liks.mean()),
        "n_lat": len(hw_lat),
    }
    return pred, diag


def run_particle_filter_smoother(
    known: pd.DataFrame,
    hw_lat: pd.DataFrame,
    tw: pd.DataFrame,
    n_particles: int = 500,
    seed: int = 42,
    output: str = "mean",
    **pf_kw,
):
    """Particle trajectory smoother.

    The forward filter stores each particle's full TVT path.  Resampling copies
    whole ancestor paths, so after the last observation the surviving trajectory
    cloud reflects information from the entire lateral sequence.  This is a
    lightweight fixed-interval smoother that keeps the PF's multi-hypothesis
    behavior instead of collapsing to a grid observation model.
    """
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    if len(hw_lat) == 0:
        return np.zeros(0), 0.0

    last = known.iloc[-1]
    last_tvt = float(last["TVT"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    tw_at_k = np.interp(known["TVT"].values.astype(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(known["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = known.tail(30)
    dt = np.diff(tail["TVT"].values.astype(float))
    dz = np.diff(tail["Z"].values.astype(float))
    dm = np.diff(tail["MD"].values.astype(float))
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    mom = pf_kw.get("mom", 0.998)
    vn = pf_kw.get("vn", 0.002)
    pn = pf_kw.get("pn", 0.005)
    rp = pf_kw.get("rp", 0.1)
    rr = pf_kw.get("rr", 0.001)
    resamp = pf_kw.get("resamp", 0.5)
    init_pos_std = pf_kw.get("init_pos_std", 4.5)
    init_rate_std = pf_kw.get("init_rate_std", 0.01)

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + init_pos_std * rng.standard_normal(N)
    rate = ir + init_rate_std * rng.standard_normal(N)
    w = np.ones(N) / N

    md_v = hw_lat["MD"].values.astype(float)
    z_v = hw_lat["Z"].values.astype(float)
    gr_v = hw_lat["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    paths = np.empty((len(hw_lat), N), dtype=np.float32)
    filt = np.empty(len(hw_lat), dtype=np.float64)

    prev_MD = last_MD
    log_lik = 0.0
    for i in range(len(hw_lat)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = mom * rate + vn * rng.standard_normal(N)
        pos = pos + rate * dm_step + pn * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d ** 2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        tvt_now = pos - z_v[i]
        filt[i] = float(np.dot(w, tvt_now))
        paths[i] = tvt_now.astype(np.float32)

        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < resamp * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            paths[: i + 1] = paths[: i + 1, idx]
            pos = pos[idx] + rp * rng.standard_normal(N)
            rate = rate[idx] + rr * rng.standard_normal(N)
            w = np.ones(N) / N
        prev_MD = md_v[i]

    if output == "median":
        pred = np.median(paths, axis=1).astype(float)
    elif output == "trimmed":
        lo = np.quantile(paths, 0.1, axis=1)
        hi = np.quantile(paths, 0.9, axis=1)
        pred = np.empty(len(hw_lat), dtype=float)
        for i in range(len(hw_lat)):
            m = (paths[i] >= lo[i]) & (paths[i] <= hi[i])
            pred[i] = float(paths[i, m].mean()) if m.any() else float(paths[i].mean())
    elif output == "filter":
        pred = filt
    else:
        pred = paths.mean(axis=1).astype(float)
    return pred, log_lik


def pf_smoother_ensemble(
    known,
    hw_lat,
    tw,
    n_particles: int = 500,
    n_seeds: int = 16,
    scale: float = 5.0,
    output: str = "mean",
    **pf_kw,
):
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = run_particle_filter_smoother(
            known, hw_lat, tw, n_particles=n_particles, seed=s, output=output, **pf_kw
        )
        preds.append(p)
        liks.append(ll)
    liks = np.array(liks)
    weights = np.exp((liks - liks.max()) / scale)
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def make_predictor(mode: str = "single", **kw):
    if mode == "single":
        def _p(known, hw_lat, tw):
            return run_particle_filter(known, hw_lat, tw, **kw)[0]
    elif mode == "smooth":
        def _p(known, hw_lat, tw):
            return pf_smoother_ensemble(known, hw_lat, tw, **kw)
    else:
        def _p(known, hw_lat, tw):
            return pf_lik_ensemble(known, hw_lat, tw, **kw)
    return _p
