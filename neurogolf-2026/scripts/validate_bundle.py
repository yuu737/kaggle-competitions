from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

from score_bundle import DEFAULT_UTILS, load_utils
from validate_rewrite_candidates import sanitized_session


ROOT = Path(__file__).resolve().parents[1]
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
    raise FileNotFoundError(f"Not a model directory or zip archive: {path}")


def encode_grid(grid: list[list[int]]) -> np.ndarray | None:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if max(height, width) > 30:
        return None
    encoded = np.zeros(GRID_SHAPE, dtype=np.float32)
    array = np.asarray(grid, dtype=np.intp)
    rows, columns = np.indices(array.shape)
    encoded[0, array, rows, columns] = 1.0
    return encoded


def session_for(payload: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(io.BytesIO(payload).read(), options)


def verify_examples(
    session: ort.InferenceSession,
    examples: list[dict[str, list[list[int]]]],
    *,
    official_threshold: bool = False,
) -> tuple[int, int, int]:
    correct = 0
    skipped = 0
    runtime_errors = 0
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for example in examples:
        input_tensor = encode_grid(example["input"])
        expected = encode_grid(example["output"])
        if input_tensor is None or expected is None:
            skipped += 1
            continue
        try:
            actual = session.run([output_name], {input_name: input_tensor})[0]
            if official_threshold:
                actual = (actual > 0).astype(float)
            correct += int(np.array_equal(actual, expected))
        except Exception:
            runtime_errors += 1
    return correct, skipped, runtime_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NeuroGolf models against local task examples."
    )
    parser.add_argument("bundle", type=Path, help="Directory or submission zip.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--arc-gen-limit",
        type=int,
        default=0,
        help="ARC-GEN examples per task (0 disables, -1 checks all).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable per-task report.",
    )
    parser.add_argument(
        "--official-threshold",
        action="store_true",
        help="Sanitize models and compare official binary outputs (raw > 0).",
    )
    parser.add_argument("--utils", type=Path, default=DEFAULT_UTILS)
    args = parser.parse_args()

    models = load_bundle(args.bundle)
    utils = load_utils(args.utils) if args.official_threshold else None
    totals = {"arc_agi_checked": 0, "arc_agi_correct": 0, "arc_gen_checked": 0,
              "arc_gen_correct": 0, "skipped": 0, "session_failures": 0}
    failed_tasks: list[str] = []
    arc_agi_failed_tasks: set[int] = set()
    arc_gen_failed_tasks: set[int] = set()
    task_results: list[dict[str, int | bool | str]] = []

    for task_id in range(1, 401):
        name = f"task{task_id:03d}.onnx"
        task = json.loads(
            (args.data_dir / f"task{task_id:03d}.json").read_text(encoding="utf-8")
        )
        try:
            session = (
                sanitized_session(models[name], utils)[0]
                if args.official_threshold
                else session_for(models[name])
            )
        except Exception as exc:
            totals["session_failures"] += 1
            failed_tasks.append(f"{name}: session error: {exc}")
            task_results.append(
                {"task": task_id, "name": name, "session_ok": False,
                 "arc_agi_solved": False, "arc_gen_solved": False}
            )
            continue

        arc_agi = task["train"] + task["test"]
        correct, skipped, runtime_errors = verify_examples(
            session, arc_agi, official_threshold=args.official_threshold
        )
        checked = len(arc_agi) - skipped
        totals["arc_agi_checked"] += checked
        totals["arc_agi_correct"] += correct
        totals["skipped"] += skipped
        if correct != checked:
            arc_agi_failed_tasks.add(task_id)
            failed_tasks.append(
                f"{name}: ARC-AGI {correct}/{checked}, runtime_errors={runtime_errors}"
            )
        result: dict[str, int | bool | str] = {
            "task": task_id,
            "name": name,
            "session_ok": True,
            "arc_agi_correct": correct,
            "arc_agi_checked": checked,
            "arc_agi_runtime_errors": runtime_errors,
            "arc_agi_solved": correct == checked,
            "arc_gen_solved": False,
        }

        if args.arc_gen_limit:
            arc_gen = task["arc-gen"]
            if args.arc_gen_limit > 0:
                arc_gen = arc_gen[: args.arc_gen_limit]
            correct, skipped, runtime_errors = verify_examples(
                session, arc_gen, official_threshold=args.official_threshold
            )
            checked = len(arc_gen) - skipped
            totals["arc_gen_checked"] += checked
            totals["arc_gen_correct"] += correct
            totals["skipped"] += skipped
            if correct != checked:
                arc_gen_failed_tasks.add(task_id)
                failed_tasks.append(
                    f"{name}: ARC-GEN {correct}/{checked}, "
                    f"runtime_errors={runtime_errors}"
                )
            result.update(
                {
                    "arc_gen_correct": correct,
                    "arc_gen_checked": checked,
                    "arc_gen_runtime_errors": runtime_errors,
                    "arc_gen_solved": correct == checked,
                }
            )
        task_results.append(result)

    print(f"Bundle: {args.bundle.resolve()}")
    print(
        f"ARC-AGI: {totals['arc_agi_correct']}/{totals['arc_agi_checked']}; "
        f"ARC-GEN: {totals['arc_gen_correct']}/{totals['arc_gen_checked']}"
    )
    print(
        f"Skipped oversized examples: {totals['skipped']}; "
        f"session failures: {totals['session_failures']}; "
        f"ARC-AGI solved tasks: {400 - len(arc_agi_failed_tasks)}/400"
    )
    if args.arc_gen_limit:
        print(
            "ARC-GEN sampled solved tasks: "
            f"{400 - len(arc_gen_failed_tasks)}/400"
        )
    for failure in failed_tasks[:30]:
        print(f"- {failure}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "bundle": str(args.bundle.resolve()),
            "arc_gen_limit": args.arc_gen_limit,
            "totals": totals,
            "arc_agi_solved_tasks": 400 - len(arc_agi_failed_tasks),
            "arc_gen_solved_tasks": (
                400 - len(arc_gen_failed_tasks) if args.arc_gen_limit else None
            ),
            "tasks": task_results,
        }
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_output.resolve()}")
    if failed_tasks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
