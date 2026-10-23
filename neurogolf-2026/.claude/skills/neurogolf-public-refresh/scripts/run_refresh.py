#!/usr/bin/env python3
"""Run the project NeuroGolf refresh script from a reusable Claude Code skill."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.project_root.resolve()
    script = root / "scripts" / "neurogolf_public_refresh.py"
    if not script.exists():
        raise FileNotFoundError(f"Refresh script not found: {script}")
    raise SystemExit(
        subprocess.call([sys.executable, str(script), *args.arguments], cwd=root)
    )


if __name__ == "__main__":
    main()
