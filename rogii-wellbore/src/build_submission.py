"""Build a Kaggle submission with the current best model: particle-filter
geosteering tracker + robust blend toward the last-known TVT (hold).

Validated leak-free CV (773-well harness): hold 15.37 -> PF 10.63 -> blend(0.8) 10.12.

At TEST time the known TVT lives in `TVT_input` (there is no `TVT` column), and the
lateral rows are exactly those with `TVT_input` NaN. We predict TVT for every row
listed in sample_submission.csv, so this works unchanged when Kaggle reruns the
notebook on the hidden test set.

This file is self-contained (no project imports) so it can be pasted into a
Kaggle notebook verbatim.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# ---- model hyperparameters (frozen from CV) ----
N_PARTICLES = 500
N_SEEDS = 16
LIK_SCALE = 5.0
BLEND_W = 0.80  # weight on PF; (1-w) on hold
PROJ_DEG = 4     # projection post-processing polynomial degree
PROJ_BLEND = 0.75  # weight on the robust fit vs raw PF in the projection step


def run_particle_filter(known_md, known_tvt, known_z, known_gr,
                        lat_md, lat_z, lat_gr, tw_tvt, tw_gr,
                        n_particles=N_PARTICLES, seed=42):
    """One particle-filter pass. State = structural position pos=TVT+Z plus rate.
    Returns (pred_tvt_for_lateral, total_log_likelihood). Uses only known TVT for
    the anchor/calibration -- no lateral labels.
    """
    n = len(lat_md)
    if n == 0:
        return np.zeros(0), 0.0

    last_tvt = float(known_tvt[-1])
    last_Z = float(known_z[-1])
    last_MD = float(known_md[-1])

    tw_at_k = np.interp(known_tvt, tw_tvt, tw_gr)
    gk = np.where(np.isfinite(known_gr), known_gr, 0.0)
    gs = float(np.clip(np.nanstd(gk - tw_at_k), 10.0, 60.0))

    # initial structural rate from last 30 known points
    k = min(30, len(known_md))
    dt = np.diff(known_tvt[-k:])
    dz = np.diff(known_z[-k:])
    dm = np.diff(known_md[-k:])
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    res = np.empty(n)
    prev_MD = last_MD
    log_lik = 0.0
    for i in range(n):
        dm_step = max(lat_md[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - lat_z[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + lat_z[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (lat_gr[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d ** 2, 600.0)), 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        if 1.0 / (w ** 2).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - lat_z[i]))
        prev_MD = lat_md[i]
    return res, log_lik


def predict_well(hw: pd.DataFrame, tw: pd.DataFrame):
    """Return a full-length TVT array for hw (known rows keep TVT_input, lateral
    rows get the blended PF prediction). Robust fallbacks to hold-constant."""
    known_mask = hw["TVT_input"].notna().values
    lat_mask = ~known_mask
    out = hw["TVT_input"].values.astype(float).copy()
    if lat_mask.sum() == 0:
        return out
    if known_mask.sum() < 5:
        # no usable anchor; fall back to nearest finite TVT_input or 0
        anchor = float(np.nanmedian(hw["TVT_input"].values)) if np.isfinite(hw["TVT_input"].values).any() else 0.0
        out[lat_mask] = anchor
        return out

    anchor = float(hw["TVT_input"].values[known_mask][-1])

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    if tw_tvt.size < 5 or not np.isfinite(tw_gr).any():
        out[lat_mask] = anchor
        return out

    # GR interpolated across full well order, then split
    gr_full = hw["GR"].interpolate(limit_direction="both")
    gr_full = gr_full.fillna(np.nanmean(tw_gr)).values.astype(float)

    known_md = hw["MD"].values[known_mask].astype(float)
    known_tvt = hw["TVT_input"].values[known_mask].astype(float)
    known_z = hw["Z"].values[known_mask].astype(float)
    known_gr = gr_full[known_mask]

    lat_md = hw["MD"].values[lat_mask].astype(float)
    lat_z = hw["Z"].values[lat_mask].astype(float)
    lat_gr = gr_full[lat_mask]

    preds, liks = [], []
    for s in range(N_SEEDS):
        p, ll = run_particle_filter(known_md, known_tvt, known_z, known_gr,
                                    lat_md, lat_z, lat_gr, tw_tvt, tw_gr,
                                    n_particles=N_PARTICLES, seed=s)
        preds.append(p); liks.append(ll)
    liks = np.array(liks)
    wts = np.exp((liks - liks.max()) / LIK_SCALE)
    wts /= wts.sum()
    pf_pred = (wts[:, None] * np.stack(preds, 0)).sum(0)

    # robust projection post-processing on the structural position U = TVT + Z
    pf_pred = project_curve(pf_pred, lat_md, lat_z, anchor + float(known_z[-1]),
                            float(known_md[-1]))

    blended = BLEND_W * pf_pred + (1.0 - BLEND_W) * anchor
    if not np.isfinite(blended).all():
        blended = np.where(np.isfinite(blended), blended, anchor)
    out[lat_mask] = blended
    return out


def _robfit(s, y, deg=PROJ_DEG, iters=4):
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


def project_curve(pred, lat_md, lat_z, anchor_u, ps_md):
    """Denoise pred via a robust low-degree fit of U=pred+Z vs normalized MD."""
    n = len(pred)
    if n < PROJ_DEG + 2:
        return pred
    end = float(lat_md[-1])
    s = (lat_md - ps_md) / max(end - ps_md, 1e-6)
    fit = _robfit(s, (pred + lat_z) - anchor_u, PROJ_DEG)
    proj_full = (anchor_u + fit) - lat_z
    out = (1.0 - PROJ_BLEND) * pred + PROJ_BLEND * proj_full
    return out if np.isfinite(out).all() else pred


def find_data_dir(explicit: str | None) -> str:
    if explicit and os.path.isdir(explicit):
        return explicit
    for c in [
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        "/kaggle/input/rogii-wellbore-geology-prediction",
        "data/raw",
    ]:
        if os.path.isdir(c):
            return c
    # fall back: any dir containing sample_submission.csv
    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    if not hits:
        hits = glob.glob("**/sample_submission.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("Could not locate competition data directory")


def build_submission(data_dir: str | None = None, output: str = "submission.csv") -> pd.DataFrame:
    data_dir = find_data_dir(data_dir)
    ss = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    ss["well"] = ss["id"].str[:8]
    ss["row"] = ss["id"].str[9:].astype(int)

    test_dir = os.path.join(data_dir, "test")
    pred_map: dict[str, float] = {}
    wells = sorted(ss["well"].unique())
    for j, wid in enumerate(wells):
        hw_path = os.path.join(test_dir, f"{wid}__horizontal_well.csv")
        tw_path = os.path.join(test_dir, f"{wid}__typewell.csv")
        try:
            hw = pd.read_csv(hw_path)
            tw = pd.read_csv(tw_path)
            full = predict_well(hw, tw)
        except Exception as e:  # never let one well break the submission
            print(f"  WARN {wid}: {e}; using hold fallback")
            hw = pd.read_csv(hw_path)
            full = hw["TVT_input"].ffill().bfill().fillna(0.0).values.astype(float)
        sub = ss[ss["well"] == wid]
        for r in sub["row"].values:
            pred_map[f"{wid}_{r}"] = float(full[r]) if r < len(full) else float(np.nanmedian(full))
        print(f"[{j+1}/{len(wells)}] {wid}: {len(sub)} rows")

    out = ss[["id"]].copy()
    out["tvt"] = out["id"].map(pred_map)
    # final safety: no NaN/inf
    if not np.isfinite(out["tvt"].values).all():
        med = float(np.nanmedian(out["tvt"].values))
        out["tvt"] = out["tvt"].fillna(med)
    out.to_csv(output, index=False)
    print(f"wrote {output}: {len(out)} rows, "
          f"tvt range {out['tvt'].min():.2f}..{out['tvt'].max():.2f}, "
          f"NaN={int(out['tvt'].isna().sum())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output", default="submission.csv")
    args = ap.parse_args()
    build_submission(args.data_dir, args.output)


if __name__ == "__main__":
    main()
