"""Builder for task029 candidate (C-11 delegated implementation).

Design reference:
  FABLE5_WORK/codex-delegations/20260704-042706-c8-design-audit-batch1-105ee391/
    workspace/FABLE5_WORK/C8_design_batch1/task029_design.md

Approach:
  Keep the baseline's front-end (color-validity + boundary-scalar detection,
  nodes cntf..rtop / ccnt..ctop) byte-for-byte identical -- it is already
  cheap (all outputs are [1,10] or [1,1,1,1] scalars) and is proven correct
  (matches the baseline's own 266/266 pass on task029, verified separately
  with an independent NumPy re-implementation of the same bbox+full-border
  test).  Replace the expensive GridSample-based crop tail (rspan..output)
  with a Gather/Pad-based crop using the same rtop/rbot/ctop/cbot scalars.

NOTE (reported to Fable 5): a Gather/OneHot-based tail was measured to cost
far more than the GridSample tail it replaces, because any ONNX op that is
not the terminal "output" node and touches the full 10-channel 30x30 data
is charged in full (9000-36000 bytes) by the official memory formula,
whereas GridSample's own output is exempt (named "output" directly) and
only the small sampling-grid construction is charged.  This build is kept
for evidence/measurement purposes; see C11 final report for the abort
recommendation.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BASELINE = Path(__file__).resolve().parents[2] / "FABLE5_WORK" / "baseline_models" / "task029.onnx"
OUT = Path(__file__).resolve().parent / "task029_candidate.onnx"

KEEP_OUTPUTS = {
    "cntf", "cnti", "cntc", "cnt2",
    "m2rf", "m2ri", "m2cf", "m2ci",
    "msum", "msum4", "c16n", "q1", "q2", "P", "lhs",
    "d", "d2", "rhs", "eq1", "ge8", "valid", "vu8",
    "cstar", "cstar3", "e4",
    "rcnt", "rpb", "rpu", "rbot", "rtop",
    "ccnt", "cpb", "cpu", "cbot", "ctop",
}


def i64(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def u8(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.uint8), name=name)


def f32(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def assemble():
    base = onnx.load(str(BASELINE))
    kept_nodes = [n for n in base.graph.node if n.output[0] in KEEP_OUTPUTS]
    kept_init_names = {
        inp for n in kept_nodes for inp in n.input
    }
    kept_inits = [init for init in base.graph.initializer if init.name in kept_init_names]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])]

    nodes = list(kept_nodes)
    extra_inits = [
        i64("one_scalar", np.ones((1, 1, 1, 1), dtype=np.int64)),
        i64("c29", np.full((1, 1, 1, 1), 29, dtype=np.int64)),
        i64("iota28", np.arange(28, dtype=np.int64)),
        i64("axis0_u", [0]),
        i64("axis1_u", [1]),
        i64("shape_flat28", [28]),
        f32("color_weight10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        u8("zero_u8", 0),
        u8("sentinel_u8", 255),
        i64("pads_28to30", [0, 0, 0, 0, 0, 0, 2, 2]),
        i64("shape_1_30_30", [1, 30, 30]),
    ]

    def N(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    # ---- boundary scalars (reuse rtop/rbot/ctop/cbot from the kept front-end) ----
    # rtop/rbot/ctop/cbot are all [1,1,1,1] INT64.
    N("Add", ["rtop", "one_scalar"], ["inner_r0"])
    N("Sub", ["rbot", "one_scalar"], ["inner_r1"])
    N("Add", ["ctop", "one_scalar"], ["inner_c0"])
    N("Sub", ["cbot", "one_scalar"], ["inner_c1"])
    N("Sub", ["inner_r1", "inner_r0"], ["h_minus1"])
    N("Add", ["h_minus1", "one_scalar"], ["h_val"])  # [1,1,1,1]
    N("Sub", ["inner_c1", "inner_c0"], ["w_minus1"])
    N("Add", ["w_minus1", "one_scalar"], ["w_val"])  # [1,1,1,1]

    # row_idx = min(iota28 + inner_r0, 29), shape [28]
    N("Add", ["iota28", "inner_r0"], ["row_idx_raw"])  # broadcasts [28] x [1,1,1,1] -> [1,1,1,28]
    N("Min", ["row_idx_raw", "c29"], ["row_idx_clip"])
    N("Reshape", ["row_idx_clip", "shape_flat28"], ["row_idx"])

    N("Add", ["iota28", "inner_c0"], ["col_idx_raw"])
    N("Min", ["col_idx_raw", "c29"], ["col_idx_clip"])
    N("Reshape", ["col_idx_clip", "shape_flat28"], ["col_idx"])

    # validity masks, shape [28]
    N("Less", ["iota28", "h_val"], ["valid_row_pre"])  # broadcasts -> [1,1,1,28]
    N("Reshape", ["valid_row_pre", "shape_flat28"], ["valid_row"])
    N("Less", ["iota28", "w_val"], ["valid_col_pre"])
    N("Reshape", ["valid_col_pre", "shape_flat28"], ["valid_col"])
    N("Unsqueeze", ["valid_row", "axis1_u"], ["valid_row_col"])  # [28,1]
    N("Unsqueeze", ["valid_col", "axis0_u"], ["valid_col_row"])  # [1,28]
    N("And", ["valid_row_col", "valid_col_row"], ["valid_cell"])  # [28,28] bool

    # ---- collapse to compact color index (single channel) ----
    N("Conv", ["input", "color_weight10"], ["color_val_f"], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1])
    N("Cast", ["color_val_f"], ["color_val_u8"], to=TensorProto.UINT8)

    N("Gather", ["color_val_u8", "row_idx"], ["gather_rows"], axis=2)
    N("Gather", ["gather_rows", "col_idx"], ["gather_cols"], axis=3)

    # sentinel_u8 (255) is used (instead of 0) to mark "no color" positions
    # (masked-out crop overflow and the trailing pad-back region), so that
    # after Cast to INT64 the OneHot index is out-of-range ([0,9]) and every
    # channel is emitted as 0 -- matching encode_grid's all-zero-pad
    # convention exactly (a real color-0 cell must still map to channel 0).
    N("Where", ["valid_cell", "gather_cols", "sentinel_u8"], ["cropped_u8"])
    N("Pad", ["cropped_u8", "pads_28to30", "sentinel_u8"], ["padded_u8"], mode="constant")

    N("Cast", ["padded_u8"], ["padded_i64"], to=TensorProto.INT64)
    N("Reshape", ["padded_i64", "shape_1_30_30"], ["padded_i64_3d"])
    N("OneHot", ["padded_i64_3d", "depth", "onoff"], ["output"], axis=1)

    return inputs, outputs, kept_inits + extra_inits, nodes


def build():
    inputs, outputs, initializers, nodes = assemble()
    graph = helper.make_graph(nodes, "task029_gather_crop", inputs, outputs, initializers)
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 21)],
    )
    return model


if __name__ == "__main__":
    model = build()
    onnx.checker.check_model(model, full_check=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))
    print("saved", OUT)
