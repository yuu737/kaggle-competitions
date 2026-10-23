#!/usr/bin/env python3
"""Verify the Fable5 NeuroGolf handoff before expensive model work begins."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


CURRENT_SHA = "a342c2fdbc664012fc1a52b83b45a01bec28f7423b002d4cc1f529a1750dcdcd"
PREVIOUS_SHA = "d091040aea012d3573cb408d23d2b928f86e013b97ee7033a677e86bfa8d44df"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(root: Path, bundle_path: str, workspace_path: str) -> Path:
    bundled = root / bundle_path
    return bundled if bundled.exists() else root / workspace_path


def check_zip(path: Path) -> dict[str, object]:
    expected = [f"task{task:03d}.onnx" for task in range(1, 401)]
    with zipfile.ZipFile(path) as archive:
        names = [Path(name).name for name in archive.namelist()]
        bad = archive.testzip()
    return {
        "exists": True,
        "entries": len(names),
        "canonical": names == expected,
        "bad_entry": bad,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("numpy", "onnx", "onnxruntime", "kagglehub"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    current = locate(
        root,
        "baselines/submission_7210_063948_current_main.zip",
        (
            "submissions/"
            "submission_7210.06395_20260704-024805_"
            "public_refresh_verified.zip"
        ),
    )
    previous = locate(
        root,
        "references/submission_7191_093048_previous_main.zip",
        (
            "submissions/"
            "submission_7191.09305_20260703-223512_"
            "public_refresh_verified.zip"
        ),
    )
    data_dir = (
        root / "data" / "kagglehub" / "competitions" / "neurogolf-2026"
    )
    tasks = sorted(data_dir.glob("task[0-9][0-9][0-9].json"))
    utils = data_dir / "neurogolf_utils" / "neurogolf_utils.py"

    checks: dict[str, object] = {
        "root": str(root),
        "packages": package_versions(),
        "task_data": {
            "count": len(tasks),
            "canonical": [path.name for path in tasks]
            == [f"task{task:03d}.json" for task in range(1, 401)],
        },
        "official_utils": utils.exists(),
        "current_main": {
            "path": str(current),
            "sha256": sha256(current) if current.exists() else None,
            "sha_match": current.exists() and sha256(current) == CURRENT_SHA,
            "zip": check_zip(current) if current.exists() else {"exists": False},
        },
        "previous_main": {
            "path": str(previous),
            "sha256": sha256(previous) if previous.exists() else None,
            "sha_match": previous.exists() and sha256(previous) == PREVIOUS_SHA,
            "zip": check_zip(previous) if previous.exists() else {"exists": False},
        },
    }

    required_ok = bool(
        checks["task_data"]["canonical"]
        and checks["official_utils"]
        and checks["current_main"]["sha_match"]
        and checks["current_main"]["zip"]["canonical"]
        and checks["previous_main"]["sha_match"]
        and checks["previous_main"]["zip"]["canonical"]
        and checks["packages"]["onnx"]
        and checks["packages"]["onnxruntime"]
        and checks["packages"]["numpy"]
    )

    full_results = []
    if args.full and required_ok:
        commands = [
            [sys.executable, "scripts/audit_onnx_bundle.py", str(current)],
            [sys.executable, "scripts/check_overfit.py", str(current)],
        ]
        for command in commands:
            process = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            full_results.append(
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout.strip(),
                    "stderr": process.stderr.strip(),
                }
            )
        required_ok = required_ok and all(
            item["returncode"] == 0 for item in full_results
        )

    checks["full_checks"] = full_results
    checks["ok"] = required_ok
    output = json.dumps(checks, indent=2) + "\n"
    print(output)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output, encoding="utf-8")
    raise SystemExit(0 if required_ok else 1)


if __name__ == "__main__":
    main()
