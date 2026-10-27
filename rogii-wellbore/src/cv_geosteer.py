"""Local CV harness for ROGII Wellbore Geology Prediction.

Each train well already encodes the exact competition setup:
  - pre-PS region: TVT_input is finite and equals the true TVT (known to the model).
  - post-PS region (lateral): TVT_input is NaN; true TVT is the label to predict.

This harness hides the lateral TVT, runs a predictor, and scores it with the
competition metric: RMSE pooled over every predicted lateral point across wells.

A predictor is a callable:
    predict(hw_known, hw_lat, typewell) -> np.ndarray  # length == len(hw_lat)
where:
    hw_known : DataFrame of pre-PS rows (has true TVT available)
    hw_lat   : DataFrame of lateral rows WITHOUT TVT/TVT_input (only MD,X,Y,Z,GR)
    typewell : DataFrame with columns TVT, GR (reference vertical well)

Nothing here fabricates results: every RMSE is computed from real train labels.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Callable

import numpy as np
import pandas as pd

DATA_DIR = "data/raw"
TEST_FEATURE_COLS = ["MD", "X", "Y", "Z", "GR"]  # columns visible at test time in the lateral


def well_ids(split: str = "train") -> list[str]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, split, "*__horizontal_well.csv")))
    return [os.path.basename(f).split("__")[0] for f in files]


def load_well(wid: str, split: str = "train"):
    hw = pd.read_csv(os.path.join(DATA_DIR, split, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(DATA_DIR, split, f"{wid}__typewell.csv"))
    return hw, tw


def split_known_lateral(hw: pd.DataFrame):
    """Split a train horizontal well into (known pre-PS, lateral) by TVT_input NaN onset."""
    nan = hw["TVT_input"].isna().values
    if not nan.any():
        return hw.copy(), hw.iloc[0:0].copy(), None
    ps = int(np.argmax(nan))
    known = hw.iloc[:ps].copy()
    lateral = hw.iloc[ps:].copy()
    return known, lateral, ps


def evaluate(
    predict: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray],
    wids: list[str] | None = None,
    split: str = "train",
    verbose: bool = False,
):
    """Run a predictor over wells and return pooled RMSE plus per-well RMSEs.

    Returns dict: {pooled_rmse, per_well_rmse_mean, per_well_rmse_median,
                   n_wells, n_points, per_well (DataFrame)}.
    """
    if wids is None:
        wids = well_ids(split)

    sq_err_sum = 0.0
    n_points = 0
    rows = []
    for wid in wids:
        hw, tw = load_well(wid, split)
        known, lateral, ps = split_known_lateral(hw)
        if len(lateral) == 0:
            continue
        truth = lateral["TVT"].values.astype(float)
        # Hide labels from the predictor: lateral gets only test-visible columns.
        hw_lat = lateral[TEST_FEATURE_COLS].copy()
        pred = np.asarray(predict(known, hw_lat, tw), dtype=float)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError(f"{wid}: predictor returned {pred.shape[0]} values, expected {truth.shape[0]}")
        if not np.isfinite(pred).all():
            raise ValueError(f"{wid}: predictor returned non-finite values")
        err = truth - pred
        sq_err_sum += float(np.sum(err ** 2))
        n_points += err.shape[0]
        rmse_w = float(np.sqrt(np.mean(err ** 2)))
        rows.append({"well": wid, "rmse": rmse_w, "n": err.shape[0]})
        if verbose:
            print(f"  {wid}: rmse={rmse_w:.3f} n={err.shape[0]}")

    per_well = pd.DataFrame(rows)
    pooled = float(np.sqrt(sq_err_sum / n_points)) if n_points else float("nan")
    return {
        "pooled_rmse": pooled,
        "per_well_rmse_mean": float(per_well["rmse"].mean()),
        "per_well_rmse_median": float(per_well["rmse"].median()),
        "n_wells": len(per_well),
        "n_points": n_points,
        "per_well": per_well,
    }


# --------------------------------------------------------------------------- #
# Baseline predictors
# --------------------------------------------------------------------------- #
def predict_hold_constant(known: pd.DataFrame, hw_lat: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    """Hold the last known TVT constant over the whole lateral."""
    last = float(known["TVT"].values[-1])
    return np.full(len(hw_lat), last)


def predict_linear_extrap(known: pd.DataFrame, hw_lat: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    """Extrapolate TVT from the slope of the last 50 known points (vs MD)."""
    k = min(50, len(known) - 1)
    if k < 2:
        return predict_hold_constant(known, hw_lat, tw)
    md_k = known["MD"].values[-k:].astype(float)
    tvt_k = known["TVT"].values[-k:].astype(float)
    slope = np.polyfit(md_k - md_k[0], tvt_k, 1)[0]
    md_lat = hw_lat["MD"].values.astype(float)
    return tvt_k[-1] + slope * (md_lat - md_k[-1])


BASELINES = {
    "hold_constant": predict_hold_constant,
    "linear_extrap": predict_linear_extrap,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="hold_constant", choices=list(BASELINES))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N wells (0=all)")
    args = ap.parse_args()

    wids = well_ids("train")
    if args.limit:
        wids = wids[: args.limit]
    res = evaluate(BASELINES[args.predictor], wids)
    print(f"predictor      : {args.predictor}")
    print(f"wells          : {res['n_wells']}")
    print(f"lateral points : {res['n_points']}")
    print(f"POOLED RMSE    : {res['pooled_rmse']:.4f}   <-- competition-style metric")
    print(f"per-well mean   : {res['per_well_rmse_mean']:.4f}")
    print(f"per-well median : {res['per_well_rmse_median']:.4f}")


if __name__ == "__main__":
    main()
