from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UTILS = (
    ROOT
    / "data"
    / "kagglehub"
    / "competitions"
    / "neurogolf-2026"
    / "neurogolf_utils"
    / "neurogolf_utils.py"
)
DEFAULT_DATA = ROOT / "data" / "kagglehub" / "competitions" / "neurogolf-2026"
GRID_SHAPE = (1, 10, 30, 30)


def load_bundle(path: Path) -> dict[str, bytes]:
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {
                Path(name).name: archive.read(name)
                for name in archive.namelist()
                if Path(name).name.endswith(".onnx")
            }
    if path.is_dir():
        return {
            model_path.name: model_path.read_bytes()
            for model_path in path.rglob("task*.onnx")
        }
    raise FileNotFoundError(path)


def load_utils(path: Path):
    spec = importlib.util.spec_from_file_location("local_neurogolf_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_grid(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros(GRID_SHAPE, dtype=np.float32)
    array = np.asarray(grid, dtype=np.intp)
    rows, columns = np.indices(array.shape)
    result[0, array, rows, columns] = 1.0
    return result


def profile_model(
    payload: bytes, sample_input: np.ndarray, profile_prefix: Path, utils
) -> tuple[int, int]:
    model = utils.sanitize_model(onnx.load_model(io.BytesIO(payload)))
    if model is None:
        raise ValueError("model sanitization failed")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(profile_prefix)
    options.log_severity_level = 4
    session = ort.InferenceSession(model.SerializeToString(), options)
    input_name = session.get_inputs()[0].name
    session.run(None, {input_name: sample_input})
    trace_path = session.end_profiling()
    memory, parameters = utils.score_network(model, trace_path)
    if memory is None or parameters is None:
        raise ValueError("official score calculation rejected model")
    return int(memory), int(parameters)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate a bundle with the official NeuroGolf cost functions."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--coverage-report",
        type=Path,
        help="Validation JSON; only fully solved tasks receive points.",
    )
    parser.add_argument("--utils", type=Path, default=DEFAULT_UTILS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    models = load_bundle(args.bundle)
    utils = load_utils(args.utils)
    solved = set(range(1, 401))
    if args.coverage_report:
        coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
        solved = {
            int(task["task"])
            for task in coverage["tasks"]
            if task.get("arc_agi_solved") and task.get("arc_gen_solved")
        }

    records: list[dict[str, int | float | bool | str | None]] = []
    estimated_score = 0.0
    invalid_tasks: list[dict[str, int | str]] = []
    temp_root = ROOT / "artifacts"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="neurogolf-score-", dir=temp_root, ignore_cleanup_errors=True
    ) as temp_dir:
        profile_dir = Path(temp_dir)
        for task_id in range(1, 401):
            name = f"task{task_id:03d}.onnx"
            task = json.loads(
                (args.data_dir / f"task{task_id:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
            sample_input = encode_grid(task["train"][0]["input"])
            try:
                memory, parameters = profile_model(
                    models[name],
                    sample_input,
                    profile_dir / f"task{task_id:03d}",
                    utils,
                )
            except Exception as exc:
                invalid_tasks.append({"task": task_id, "error": str(exc)})
                records.append(
                    {
                        "task": task_id,
                        "solved": False,
                        "memory": None,
                        "parameters": None,
                        "cost": None,
                        "points_if_correct": 0.0,
                        "error": str(exc),
                    }
                )
                continue
            cost = memory + parameters
            points = max(1.0, 25.0 - math.log(max(1.0, cost)))
            is_solved = task_id in solved
            estimated_score += points if is_solved else 0.0
            records.append(
                {
                    "task": task_id,
                    "solved": is_solved,
                    "memory": memory,
                    "parameters": parameters,
                    "cost": cost,
                    "points_if_correct": points,
                }
            )

    report = {
        "bundle": str(args.bundle.resolve()),
        "solved_tasks": len(solved),
        "officially_scorable_tasks": 400 - len(invalid_tasks),
        "invalid_tasks": invalid_tasks,
        "estimated_score": estimated_score,
        "total_memory": sum(
            int(item["memory"]) for item in records if item["memory"] is not None
        ),
        "total_parameters": sum(
            int(item["parameters"])
            for item in records
            if item["parameters"] is not None
        ),
        "tasks": records,
    }
    print(
        f"Estimated score: {estimated_score:.2f} from {len(solved)}/400 solved tasks"
    )
    print(
        f"Officially scorable models: {400 - len(invalid_tasks)}/400; "
        f"rejected tasks: {[item['task'] for item in invalid_tasks]}"
    )
    print(
        f"Total memory: {report['total_memory']:,}; "
        f"total parameters: {report['total_parameters']:,}"
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_output.resolve()}")


if __name__ == "__main__":
    main()
