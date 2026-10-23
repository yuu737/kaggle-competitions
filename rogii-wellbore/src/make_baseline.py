from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


TARGET = "tvt"
ID_COL = "id"
WELL_COL = "well"
ROW_COL = "row_index"


def find_sample_submission(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("**/sample_submission.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find sample_submission.csv under {data_dir}")
    return candidates[0]


def read_horizontal_files(split_dir: Path, require_target: bool) -> pd.DataFrame:
    paths = sorted(split_dir.glob("*__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No *__horizontal_well.csv files found in {split_dir}")

    frames = []
    for path in paths:
        well = path.name.split("__", 1)[0]
        df = pd.read_csv(path)
        df.columns = [col.lower() for col in df.columns]
        if require_target and TARGET not in df.columns:
            raise ValueError(f"{path} does not contain target column {TARGET!r}")

        df[WELL_COL] = well
        df[ROW_COL] = np.arange(len(df), dtype=np.int32)
        df[ID_COL] = df[WELL_COL] + "_" + df[ROW_COL].astype(str)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {test_dir}. "
            "Download and extract the Kaggle data into data/raw."
        )

    train = read_horizontal_files(train_dir, require_target=True)
    test = read_horizontal_files(test_dir, require_target=False)
    sample = pd.read_csv(find_sample_submission(data_dir))
    return train, test, sample


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"x", "y"}.issubset(out.columns):
        out["xy_dist"] = np.hypot(
            out["x"] - out.groupby(WELL_COL)["x"].transform("first"),
            out["y"] - out.groupby(WELL_COL)["y"].transform("first"),
        )

    if {"md", "z"}.issubset(out.columns):
        out["md_minus_z"] = out["md"] - out["z"]

    if "gr" in out.columns:
        grouped = out.groupby(WELL_COL)["gr"]
        out["gr_roll_11"] = grouped.transform(
            lambda s: s.rolling(11, center=True, min_periods=1).mean()
        )
        out["gr_roll_51"] = grouped.transform(
            lambda s: s.rolling(51, center=True, min_periods=1).mean()
        )

    return out


def parse_submission_ids(sample: pd.DataFrame) -> pd.DataFrame:
    parsed = sample[ID_COL].astype(str).str.extract(r"^(?P<well>[0-9a-fA-F]{8})_(?P<row_index>\d+)$")
    if parsed.isna().any().any():
        raise ValueError("sample_submission.csv has unexpected id format.")
    parsed[ROW_COL] = parsed[ROW_COL].astype(np.int32)
    return parsed


def build_matrices(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = add_basic_features(train)
    test = add_basic_features(test)

    labeled = train[train[TARGET].notna()].copy()
    y = labeled[TARGET].astype(float)
    groups = labeled[WELL_COL]

    sample_index = parse_submission_ids(sample)
    sample_test = sample_index.merge(test, on=[WELL_COL, ROW_COL], how="left")
    missing = sample_test["md"].isna().sum() if "md" in sample_test.columns else sample_test.isna().all(axis=1).sum()
    if missing:
        print(f"Warning: {missing} sample_submission rows did not match visible test rows.")

    drop_cols = {
        TARGET,
        "tvt_input",
        ID_COL,
        WELL_COL,
        ROW_COL,
    }
    surface_cols = {"ancc", "astnu", "astnl", "egfdu", "egfdl", "buda"}
    drop_cols.update(surface_cols)

    feature_cols = [
        col
        for col in labeled.columns
        if col in sample_test.columns and col not in drop_cols
    ]

    x_train = labeled[feature_cols]
    x_test = sample_test[feature_cols]
    return x_train, y, x_test, groups


def make_model(x_train: pd.DataFrame, model_name: str) -> Pipeline:
    numeric_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_name == "hist":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            l2_regularization=0.05,
            random_state=42,
        )
    elif model_name == "lgbm":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed. Run: python -m pip install lightgbm")
        model = LGBMRegressor(
            objective="regression",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--model", choices=["hist", "lgbm"], default="hist")
    args = parser.parse_args()

    train, test, sample = load_data(args.data_dir)
    x_train, y, x_test, groups = build_matrices(train, test, sample)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit_idx, valid_idx = next(splitter.split(x_train, y, groups=groups))

    model = make_model(x_train, args.model)
    model.fit(x_train.iloc[fit_idx], y.iloc[fit_idx])
    valid_pred = model.predict(x_train.iloc[valid_idx])
    rmse = root_mean_squared_error(y.iloc[valid_idx], valid_pred)
    print(f"Well-group validation RMSE: {rmse:.6f}")

    model.fit(x_train, y)
    submission = sample.copy()
    submission[TARGET] = model.predict(x_test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COL, TARGET]].to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
