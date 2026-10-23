"""Evaluate neural + GBDT stacker + PF + spatial on a common cached id set."""
import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neural", default="output/neural_oof.csv")
    p.add_argument("--cache", default="output/pf_sp_cache.csv")
    p.add_argument("--train-csv", default="kaggle_datasets/wellbore-geology-prediction-artifacts/data/train.csv")
    p.add_argument("--max-rows", type=int, default=0, help="0 means all cached rows")
    p.add_argument("--grid", type=float, default=0.1)
    return p.parse_args()


def pooled(pred, target):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


def simplex_grid(step):
    vals = np.arange(0, 1 + 1e-9, step)
    for a in vals:
        for b in vals:
            if a + b > 1 + 1e-9:
                continue
            for c in vals:
                if a + b + c > 1 + 1e-9:
                    continue
                yield float(a), float(b), float(c), float(1 - a - b - c)


def main():
    args = parse_args()
    t0 = time.time()
    neural = pd.read_csv(args.neural, dtype={"id": str, "well": str})[["id", "neural_dev"]]
    cache = pd.read_csv(args.cache, dtype={"id": str, "well": str})
    if args.max_rows and len(cache) > args.max_rows:
        cache = cache.iloc[: args.max_rows].copy()
    ids = set(cache["id"])
    print(f"cached rows={len(cache)} wells={cache['well'].nunique()}", flush=True)

    header = pd.read_csv(args.train_csv, nrows=0).columns.tolist()
    feats = [c for c in header if c not in ("well", "id", "target")]
    chunks = []
    for ch in pd.read_csv(args.train_csv, usecols=["well", "id", "target"] + feats, chunksize=500000):
        sub = ch[ch["id"].isin(ids)]
        if len(sub):
            chunks.append(sub)
    feat_df = pd.concat(chunks, ignore_index=True)
    feat_df["well"] = feat_df["well"].astype(str)
    feat_df[feats] = feat_df[feats].astype("float32")
    print(f"feature rows={len(feat_df)} loaded in {time.time() - t0:.0f}s", flush=True)

    uniq = np.array(sorted(feat_df["well"].unique()))
    rng = np.random.default_rng(1)
    folds = np.array_split(rng.permutation(len(uniq)), 5)
    stack_oof = np.full(len(feat_df), np.nan, dtype="float32")
    y = feat_df["target"].values.astype("float32")
    x = feat_df[feats].values
    for fold in range(5):
        val_wells = set(uniq[folds[fold]])
        val_mask = feat_df["well"].isin(val_wells).values
        model = lgb.LGBMRegressor(
            device="gpu",
            gpu_use_dp=False,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=95,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=200,
            reg_lambda=2.0,
            verbose=-1,
        )
        model.fit(x[~val_mask], y[~val_mask])
        stack_oof[val_mask] = model.predict(x[val_mask])
        print(f"stacker fold {fold} done", flush=True)
    feat_df["stack_dev"] = stack_oof

    merged = (
        cache.merge(neural, on="id", how="inner")
        .merge(feat_df[["id", "stack_dev"]], on="id", how="inner")
        .dropna(subset=["target", "neural_dev", "stack_dev", "pf_dev", "sp_dev"])
    )
    yv = merged["target"].values
    print(f"aligned rows={len(merged)} wells={merged['well'].nunique()}", flush=True)
    print(
        "components: neural %.4f | stack %.4f | PF %.4f | spatial %.4f"
        % (
            pooled(merged["neural_dev"], yv),
            pooled(merged["stack_dev"], yv),
            pooled(merged["pf_dev"], yv),
            pooled(merged["sp_dev"], yv),
        )
    )

    best3 = None
    vals = np.arange(0, 1 + 1e-9, args.grid)
    for a in vals:
        for b in vals:
            if a + b > 1 + 1e-9:
                continue
            pred = a * merged["stack_dev"] + b * merged["pf_dev"] + (1 - a - b) * merged["sp_dev"]
            score = pooled(pred, yv)
            if best3 is None or score < best3[0]:
                best3 = (score, float(a), float(b), float(1 - a - b))
    print("BEST 3-way stack/pf/sp = %.4f (stack=%.2f pf=%.2f sp=%.2f)" % best3)

    best4 = None
    for n, s, p, sp in simplex_grid(args.grid):
        pred = n * merged["neural_dev"] + s * merged["stack_dev"] + p * merged["pf_dev"] + sp * merged["sp_dev"]
        score = pooled(pred, yv)
        if best4 is None or score < best4[0]:
            best4 = (score, n, s, p, sp)
    print("BEST 4-way neural/stack/pf/sp = %.4f (neural=%.2f stack=%.2f pf=%.2f sp=%.2f)" % best4)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
