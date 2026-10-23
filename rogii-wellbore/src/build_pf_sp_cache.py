"""Build a reusable PF + spatial cache keyed by submission-style id.

This is intentionally checkpointed. PF is CPU-heavy and occasionally slow for a
well, so each completed well is written to disk before moving on.
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import cv_geosteer as cv
import geosteer_pf as pf
import geosteer_post as post


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neural", default="output/neural_oof.csv")
    p.add_argument("--out", default="output/pf_sp_cache.csv")
    p.add_argument("--max-wells", type=int, default=80)
    p.add_argument("--particles", type=int, default=250)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--formation", default="EGFDU")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_surface_cloud(formation):
    xy, val, wid_arr = [], [], []
    for wid in cv.well_ids("train"):
        hw, _ = cv.load_well(wid)
        if formation not in hw:
            continue
        x = hw["X"].values
        y = hw["Y"].values
        v = hw[formation].values
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        sample = np.c_[x[m], y[m]][::8]
        xy.append(sample)
        val.append(v[m][::8])
        wid_arr.append(np.full(len(sample), wid, dtype=object))
    return np.vstack(xy), np.concatenate(val), np.concatenate(wid_arr)


def spatial_prediction(k, lat, wid, xy, val, wid_arr):
    anchor = k["TVT"].values[-1]
    z_ps = k["Z"].values[-1]
    mask = wid_arr != wid
    tree = cKDTree(xy[mask])
    ov = val[mask]
    q = np.c_[lat["X"].values, lat["Y"].values]
    ps = np.array([[k["X"].values[-1], k["Y"].values[-1]]])

    center = q.mean(0)
    _, idx_near = tree.query(center, k=min(2000, len(ov)))
    a = np.c_[xy[mask][idx_near, 0], xy[mask][idx_near, 1], np.ones(len(idx_near))]
    coef = np.linalg.lstsq(a, ov[idx_near], rcond=None)[0]
    trend = lambda p: p @ coef[:2] + coef[2]

    residual = ov - trend(xy[mask])
    dist, idx = tree.query(q, k=12)
    weight = 1 / (dist**2 + 1)
    surf = (weight * residual[idx]).sum(1) / weight.sum(1) + trend(q)

    dist_ps, idx_ps = tree.query(ps, k=12)
    weight_ps = 1 / (dist_ps**2 + 1)
    surf_ps = (weight_ps * residual[idx_ps]).sum() / weight_ps.sum() + trend(ps)[0]

    pred = anchor - (lat["Z"].values - z_ps) + (surf - surf_ps)
    gate = np.clip((1500 - dist[:, 0]) / 1200, 0, 1)
    return gate * pred + (1 - gate) * anchor


def save_rows(rows, out_path):
    tmp = out_path + ".tmp"
    pd.DataFrame(rows, columns=["id", "well", "pf_dev", "sp_dev", "target"]).to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def main():
    args = parse_args()
    neural = pd.read_csv(args.neural, dtype={"id": str, "well": str})
    valid = set(cv.well_ids("train"))
    wells = [w for w in dict.fromkeys(neural["well"].tolist()) if w in valid][: args.max_wells]

    rows = []
    done = set()
    if args.resume and os.path.exists(args.out):
        old = pd.read_csv(args.out, dtype={"id": str, "well": str})
        rows = list(old[["id", "well", "pf_dev", "sp_dev", "target"]].itertuples(index=False, name=None))
        done = set(old["well"].astype(str).unique())
        print(f"resuming from {args.out}: {len(done)} wells, {len(rows)} rows", flush=True)

    print(
        f"caching PF+spatial for {len(wells)} wells "
        f"(particles={args.particles}, steps={args.steps})",
        flush=True,
    )
    xy, val, wid_arr = build_surface_cloud(args.formation)
    feat = cv.TEST_FEATURE_COLS
    t0 = time.time()

    for i, wid in enumerate(wells):
        if wid in done:
            continue
        tw0 = time.time()
        hw, tw = cv.load_well(wid)
        k, lat, ps = cv.split_known_lateral(hw)
        print(f"well {i + 1}/{len(wells)} {wid} latlen={len(lat)}", flush=True)
        if len(lat) < 5:
            done.add(wid)
            continue

        anchor = k["TVT"].values[-1]
        pf_pred, _ = pf.pf_ensemble_with_diag(k, lat[feat], tw, args.particles, args.steps)
        pf_pred = post.project(k, lat[feat], pf_pred)
        sp_pred = spatial_prediction(k, lat, wid, xy, val, wid_arr)
        truth = lat["TVT"].values

        for j, row_idx in enumerate(np.arange(ps, ps + len(lat))):
            rows.append(
                (
                    f"{wid}_{row_idx}",
                    wid,
                    pf_pred[j] - anchor,
                    sp_pred[j] - anchor,
                    truth[j] - anchor,
                )
            )
        done.add(wid)
        save_rows(rows, args.out)
        print(f"   done {time.time() - tw0:.1f}s, saved {len(rows)} rows, total {time.time() - t0:.0f}s", flush=True)

    save_rows(rows, args.out)
    print(f"saved {args.out}: {len(done)} wells, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
