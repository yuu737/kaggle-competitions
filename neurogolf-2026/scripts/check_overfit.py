"""Flag memorization-style ('overfit') task graphs in a submission bundle.

A task that replaces a real algorithm with a tiny graph plus a large constant
lookup table fed into Gather/GatherND typically memorizes the ARC-GEN answers:
it passes local train+test+ARC-GEN but fails the official private benchmark.
This read-only check flags such tasks so they are never submitted.

Heuristic: a task is suspicious if it has a Gather/GatherND whose data operand is
an initializer/Constant with >= TABLE_MIN elements AND the graph is small
(node_count <= NODE_MAX). Tune via flags.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]


def table_elems(model: onnx.ModelProto) -> dict[str, int]:
    elems = {x.name: int(np.prod(x.dims)) for x in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    elems[n.output[0]] = int(np.prod(a.t.dims))
    return elems


def scan(bundle: Path, table_min: int, node_max: int) -> list[tuple]:
    flagged = []
    with zipfile.ZipFile(bundle) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".onnx"):
                continue
            m = onnx.load_model_from_string(z.read(name))
            elems = table_elems(m)
            nodes = len(m.graph.node)
            for n in m.graph.node:
                if n.op_type in ("Gather", "GatherND") and n.input:
                    tbl = elems.get(n.input[0], 0)
                    if tbl >= table_min and nodes <= node_max:
                        flagged.append((name, nodes, tbl, n.op_type, n.input[0]))
                        break
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--table-min", type=int, default=512)
    ap.add_argument("--node-max", type=int, default=40)
    args = ap.parse_args()
    flagged = scan(args.bundle, args.table_min, args.node_max)
    if not flagged:
        print(f"OK: no memorization-style tasks in {args.bundle} "
              f"(table>={args.table_min} elems AND nodes<={args.node_max})")
        return
    print(f"SUSPICIOUS ({len(flagged)}) in {args.bundle}:")
    for name, nodes, tbl, op, tname in flagged:
        print(f"  {name}: nodes={nodes} {op} table='{tname}' elems={tbl}")


if __name__ == "__main__":
    main()
