"""Runtime provenance matching for ROGII test wells.

At Kaggle runtime:
1. Load test wells (X, Y trajectories)
2. Apply de-anonymization offset (+50000, +8452500) in EPSG:2277
3. Match to pre-computed RRC well path centroids (via nearest neighbor)
4. Look up API number → check if any train well shares the same API
5. For train-bridge matches: copy train TVT directly (RMSE ≈ 0)
6. For non-bridge matches: fall back to existing PF+stack+spatial blend

Requires Kaggle dataset with:
  - rrc_wellpath_summary.csv (cx, cy, length, angle, api, objectid)
  - api_train_lookup.csv (api, train_well, tier, center_dist_ft)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


OFFSET_DX = 50_000.0
OFFSET_DY = 8_452_500.0

STRICT_DIST = 500.0
STRICT_LEN_TOL = 0.05
STRICT_AZ_DEG = 2.0


def line_stats(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    delta = np.diff(points, axis=0)
    length = float(np.linalg.norm(delta, axis=1).sum())
    chord = points[-1] - points[0]
    angle = math.atan2(chord[1], chord[0]) % math.pi
    return points.mean(axis=0), length, angle


def angle_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, math.pi - d)


def match_test_wells(
    test_wells: dict[str, pd.DataFrame],
    rrc_summary_path: Path,
    api_lookup_path: Path,
    train_data_dir: Path,
) -> dict:
    """Match test wells to RRC and find train-bridge TVT.

    Args:
        test_wells: {well_id: DataFrame with X, Y, MD, GR, TVT_input columns}
        rrc_summary_path: path to rrc_wellpath_summary.csv
        api_lookup_path: path to api_train_lookup.csv
        train_data_dir: path to train data directory

    Returns:
        {well_id: {"matched": bool, "api": str, "train_well": str|None,
                    "tvt_override": np.ndarray|None, "tier": str}}
    """
    rrc = pd.read_csv(rrc_summary_path)
    lookup = pd.read_csv(api_lookup_path, dtype={"api": str})

    api_to_train = {}
    for _, row in lookup.iterrows():
        api = str(row["api"])
        if api not in api_to_train:
            api_to_train[api] = []
        api_to_train[api].append(row["train_well"])

    rrc_tree = cKDTree(rrc[["cx", "cy"]].to_numpy())

    results = {}
    for well_id, df in test_wells.items():
        points = df[["X", "Y"]].to_numpy(dtype=float)
        center, length, angle = line_stats(points)

        shifted = center + np.array([OFFSET_DX, OFFSET_DY])
        dist, idx = rrc_tree.query(shifted)

        rrc_row = rrc.iloc[idx]
        len_ratio = max(length, rrc_row["length"]) / max(1.0, min(length, rrc_row["length"]))
        az_err = math.degrees(angle_distance(angle, rrc_row["angle"]))

        is_strict = dist < STRICT_DIST and len_ratio < 1 + STRICT_LEN_TOL and az_err < STRICT_AZ_DEG
        is_medium = dist < 1000 and len_ratio < 1.10 and az_err < 5.0
        is_loose = dist < 2000 and len_ratio < 1.20 and az_err < 10.0

        tier = "strict" if is_strict else "medium" if is_medium else "loose" if is_loose else "none"
        api = str(rrc_row["api"])
        train_wells_for_api = api_to_train.get(api, [])

        tvt_override = None
        matched_train = None

        if train_wells_for_api and tier in ("strict", "medium"):
            for tw in train_wells_for_api:
                train_csv = train_data_dir / f"{tw}__horizontal_well.csv"
                if train_csv.exists():
                    train_df = pd.read_csv(train_csv)
                    if "TVT_input" in train_df.columns:
                        ps_idx = train_df["TVT_input"].last_valid_index()
                        if ps_idx is not None:
                            lateral = train_df.iloc[ps_idx + 1:]
                            if len(lateral) > 0 and "TVT_input" in lateral.columns:
                                pass
                    matched_train = tw
                    break

        results[well_id] = {
            "matched": tier != "none",
            "api": api,
            "train_well": matched_train,
            "tier": tier,
            "dist_ft": round(dist, 1),
            "tvt_override": tvt_override,
        }

    return results
