"""Project paths and process-local Kaggle authentication."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
COMPETITION = "rsna-knee-abnormality-detection"
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
METADATA_FILES = [
    "train.csv", "train_series.csv", "test.csv", "test_series.csv",
    "sample_submission.csv",
]


def kaggle_api():
    """Read the supplied token without copying it to another file."""
    if not os.environ.get("KAGGLE_API_TOKEN"):
        token_path = ROOT / "Kaggle_API.txt"
        if not token_path.is_file():
            raise RuntimeError("Provide Kaggle_API.txt or set KAGGLE_API_TOKEN.")
        token = token_path.read_text(encoding="utf-8-sig").strip()
        if not token.startswith("KGAT_") or any(c.isspace() for c in token):
            raise RuntimeError("Kaggle_API.txt must contain one KGAT API token.")
        os.environ["KAGGLE_API_TOKEN"] = token
    config_dir = ROOT / ".cache" / "kaggle"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def safe_error(exc: Exception) -> str:
    """Remove credentials from SDK errors before writing diagnostics."""
    message = str(exc)
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if token:
        message = message.replace(token, "[REDACTED]")
    return message
