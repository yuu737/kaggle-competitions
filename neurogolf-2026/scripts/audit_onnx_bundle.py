from __future__ import annotations

import argparse
import io
import math
import zipfile
from collections import Counter
from pathlib import Path

import onnx


TASK_COUNT = 400
FILE_SIZE_LIMIT = int(1.44 * 1024 * 1024)
EXCLUDED_OPS = {
    "LOOP",
    "SCAN",
    "NONZERO",
    "UNIQUE",
    "SCRIPT",
    "FUNCTION",
    "COMPRESS",
}


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


def parameter_count(model: onnx.ModelProto) -> int:
    total = sum(math.prod(initializer.dims) for initializer in model.graph.initializer)
    total += sum(
        math.prod(initializer.values.dims)
        for initializer in model.graph.sparse_initializer
    )
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                total += math.prod(attribute.t.dims) if attribute.t.dims else 1
            elif attribute.name == "sparse_value":
                dims = attribute.sparse_tensor.values.dims
                total += math.prod(dims) if dims else 1
            elif attribute.name == "value_floats":
                total += len(attribute.floats)
            elif attribute.name == "value_ints":
                total += len(attribute.ints)
            elif attribute.name == "value_strings":
                total += len(attribute.strings)
    return total


def validate_model(name: str, payload: bytes) -> tuple[Counter[str], int, list[str]]:
    errors: list[str] = []
    if len(payload) > FILE_SIZE_LIMIT:
        errors.append(f"file exceeds {FILE_SIZE_LIMIT} bytes")

    try:
        model = onnx.load_model(io.BytesIO(payload))
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        return Counter(), 0, [f"ONNX checker failed: {exc}"]

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        errors.append("model must have exactly one input and one output")
    if model.functions:
        errors.append("local ONNX functions are forbidden")
    if any(opset.domain not in {"", "ai.onnx"} for opset in model.opset_import):
        errors.append("custom operator domain found")

    operations = Counter(node.op_type for node in model.graph.node)
    forbidden = sorted(op for op in operations if op.upper() in EXCLUDED_OPS)
    if forbidden:
        errors.append(f"forbidden operators: {', '.join(forbidden)}")
    if any("Sequence" in op for op in operations):
        errors.append("sequence operator found")
    if any(
        attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in model.graph.node
        for attribute in node.attribute
    ):
        errors.append("subgraph found")

    return operations, parameter_count(model), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a NeuroGolf ONNX bundle.")
    parser.add_argument("bundle", type=Path, help="Directory or submission zip.")
    args = parser.parse_args()

    tasks = load_bundle(args.bundle)
    expected = {f"task{task_id:03d}.onnx" for task_id in range(1, TASK_COUNT + 1)}
    missing = sorted(expected - tasks.keys())
    extras = sorted(tasks.keys() - expected)

    operations: Counter[str] = Counter()
    total_parameters = 0
    failures: list[tuple[str, str]] = []
    for name in sorted(expected & tasks.keys()):
        model_ops, parameters, errors = validate_model(name, tasks[name])
        operations.update(model_ops)
        total_parameters += parameters
        failures.extend((name, error) for error in errors)

    print(f"Bundle: {args.bundle.resolve()}")
    print(f"Canonical tasks: {len(expected & tasks.keys())}/{TASK_COUNT}")
    print(f"Model bytes: {sum(map(len, tasks.values())):,}")
    print(f"Initializer/constant parameters: {total_parameters:,}")
    print(f"Nodes: {sum(operations.values()):,}")
    print("Top operators: " + ", ".join(
        f"{name}={count}" for name, count in operations.most_common(12)
    ))
    print(f"Missing: {len(missing)}; extras: {len(extras)}; failures: {len(failures)}")
    for name, error in failures[:20]:
        print(f"- {name}: {error}")
    if missing:
        print("Missing tasks: " + ", ".join(missing[:20]))
    if extras:
        print("Unexpected files: " + ", ".join(extras[:20]))
    if missing or extras or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

