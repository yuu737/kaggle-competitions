from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


ID_COL = "id"
TARGET = "tvt"


def parse_ids(ids: pd.Series) -> pd.DataFrame:
    parsed = ids.astype(str).str.extract(r"^(?P<well>[0-9a-fA-F]{8})_(?P<row_index>\d+)$")
    if parsed.isna().any().any():
        bad = ids[parsed.isna().any(axis=1)].head().tolist()
        raise ValueError(f"Unexpected id format. Examples: {bad}")
    parsed["row_index"] = parsed["row_index"].astype(int)
    return parsed


def load_truth(data_dir: Path, ids: pd.Series) -> pd.DataFrame:
    parsed = parse_ids(ids)
    truth_parts = []

    for well, group in parsed.groupby("well", sort=False):
        path = data_dir / "train" / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Local truth is only available when sample ids "
                "refer to wells also present in data/raw/train."
            )

        horizontal = pd.read_csv(path, usecols=["TVT"])
        row_index = group["row_index"].to_numpy()
        if row_index.max() >= len(horizontal):
            raise IndexError(f"{well} row index is outside the training file.")

        part = pd.DataFrame(
            {
                ID_COL: ids.iloc[group.index].to_numpy(),
                "truth_tvt": horizontal["TVT"].iloc[row_index].to_numpy(),
            },
            index=group.index,
        )
        truth_parts.append(part)

    truth = pd.concat(truth_parts).sort_index()
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    submission = pd.read_csv(args.submission)
    if ID_COL not in submission.columns or TARGET not in submission.columns:
        raise ValueError(f"Submission must contain columns: {ID_COL},{TARGET}")

    truth = load_truth(args.data_dir, submission[ID_COL])
    merged = submission[[ID_COL, TARGET]].merge(truth, on=ID_COL, how="left")
    if merged["truth_tvt"].isna().any():
        raise ValueError("Some submission ids did not match local truth.")

    rmse = root_mean_squared_error(merged["truth_tvt"], merged[TARGET])
    mae = np.mean(np.abs(merged["truth_tvt"] - merged[TARGET]))
    print(f"Rows: {len(merged)}")
    print(f"Local public-sample RMSE: {rmse:.6f}")
    print(f"Local public-sample MAE:  {mae:.6f}")


if __name__ == "__main__":
    main()
