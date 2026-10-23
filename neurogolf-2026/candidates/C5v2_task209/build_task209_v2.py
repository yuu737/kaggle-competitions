from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def init(name: str, value, dtype=None) -> str:
        array = np.asarray(value, dtype=dtype)
        inits.append(numpy_helper.from_array(array, name))
        return name

    def node(op: str, inputs: list[str], output: str, **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    zero_i = init("zero_i", 0, np.int32)
    one_i = init("one_i", 1, np.int32)
    two_i = init("two_i", 2, np.int32)
    three_i = init("three_i", 3, np.int32)
    four_i = init("four_i", 4, np.int32)
    nineteen_i = init("nineteen_i", 19, np.int32)
    big_i = init("big_i", 9999, np.int32)
    neg_i = init("neg_i", -1, np.int32)
    zero_u8 = init("zero_u8", 0, np.uint8)
    one_u8 = init("one_u8", 1, np.uint8)
    four_u8 = init("four_u8", 4, np.uint8)
    neg_u8 = init("neg_u8", 255, np.uint8)
    zero_f = init("zero_f", 0.0, np.float32)
    coords = init("coords", np.arange(20, dtype=np.int32))
    row4 = init("row4", np.arange(20, dtype=np.int32).reshape(1, 1, 20, 1))
    col4 = init("col4", np.arange(20, dtype=np.int32).reshape(1, 1, 1, 20))
    dr = init("dr", np.arange(3, dtype=np.int32).reshape(1, 1, 3, 1))
    dc = init("dc", np.arange(5, dtype=np.int32).reshape(1, 1, 1, 5))
    score15 = init("score15", np.arange(15, dtype=np.int32).reshape(1, 1, 3, 5))
    shape15 = init("shape15", [15], np.int64)
    shape3 = init("shape3", [3], np.int64)
    shape5 = init("shape5", [5], np.int64)
    shape20 = init("shape20", [20], np.int64)
    pads30 = init("pads30", [0, 0, 0, 0, 0, 0, 10, 10], np.int64)
    qscale = init("qscale", 1.0, np.float32)
    out_colors = init("out_colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))

    # A valid 20x20 label plane directly from the upper-left 20x20 input.
    # A 2x2 kernel with dilation 10 has effective size 11, while charging only
    # 40 parameters instead of a dense 1x10x11x11 mostly-zero kernel.
    weights = np.zeros((1, 10, 2, 2), dtype=np.float32)
    weights[0, :, 0, 0] = np.arange(10, dtype=np.float32)
    init("label_w", weights)
    label_f = node("Conv", ["input", "label_w"], "label_f", dilations=[10, 10])
    label = node("Cast", [label_f], "label", to=TensorProto.UINT8)

    mask4 = node("Equal", [label, four_u8], "mask4")
    mask4_u8 = node("Cast", [mask4], "mask4_u8", to=TensorProto.UINT8)
    row4_count = node("ReduceMax", [mask4_u8], "row4_count", axes=[3], keepdims=0)
    col4_count = node("ReduceMax", [mask4_u8], "col4_count", axes=[2], keepdims=0)
    row4_has = node("Cast", [row4_count], "row4_has", to=TensorProto.BOOL)
    col4_has = node("Cast", [col4_count], "col4_has", to=TensorProto.BOOL)
    rmin_vec = node("Where", [row4_has, coords, big_i], "rmin_vec")
    rmax_vec = node("Where", [row4_has, coords, neg_i], "rmax_vec")
    cmin_vec = node("Where", [col4_has, coords, big_i], "cmin_vec")
    cmax_vec = node("Where", [col4_has, coords, neg_i], "cmax_vec")
    r_top = node("ReduceMin", [rmin_vec], "r_top", keepdims=0)
    r_bot = node("ReduceMax", [rmax_vec], "r_bot", keepdims=0)
    c_left = node("ReduceMin", [cmin_vec], "c_left", keepdims=0)
    c_right = node("ReduceMax", [cmax_vec], "c_right", keepdims=0)
    ohm1 = node("Sub", [r_bot, r_top], "ohm1")
    owm1 = node("Sub", [c_right, c_left], "owm1")
    row_in = node("And", [
        node("GreaterOrEqual", [coords, r_top], "row_ge_top"),
        node("LessOrEqual", [coords, r_bot], "row_le_bot"),
    ], "row_in")
    col_in = node("And", [
        node("GreaterOrEqual", [coords, c_left], "col_ge_left"),
        node("LessOrEqual", [coords, c_right], "col_le_right"),
    ], "col_in")
    # NOTE: an earlier v2 draft tried UINT8 here to shrink these Einsum
    # operands, but ORT 1.22's Einsum kernel only implements
    # float/double/int32/int64 (measured directly; see validation notes).
    # UINT8/INT8/INT16 all fail with NOT_IMPLEMENTED at session-creation time,
    # so this stays FLOAT to match v1 exactly (no byte savings available here
    # without an unsupported dtype).
    row_in_u8 = node("Cast", [row_in], "row_in_u8", to=TensorProto.FLOAT)
    col_in_u8 = node("Cast", [col_in], "col_in_u8", to=TensorProto.FLOAT)

    non4_u8 = node("Where", [mask4, zero_f, label_f], "non4_u8")
    total_rows = node("ReduceMax", [non4_u8], "total_rows", axes=[3], keepdims=0)
    total_cols = node("ReduceMax", [non4_u8], "total_cols", axes=[2], keepdims=0)
    tmpl_rows0 = node(
        "Einsum", [non4_u8, col_in_u8], "tmpl_rows0", equation="abcd,d->abc"
    )
    tmpl_cols0 = node(
        "Einsum", [non4_u8, row_in_u8], "tmpl_cols0", equation="abcd,c->abd"
    )
    tmpl_rows = node("Mul", [tmpl_rows0, row_in_u8], "tmpl_rows")
    tmpl_cols = node("Mul", [tmpl_cols0, col_in_u8], "tmpl_cols")
    tmpl_row_has = node("Cast", [tmpl_rows], "tmpl_row_has", to=TensorProto.BOOL)
    tmpl_col_has = node("Cast", [tmpl_cols], "tmpl_col_has", to=TensorProto.BOOL)
    tr_vec = node("Where", [tmpl_row_has, coords, big_i], "tr_vec")
    tc_vec = node("Where", [tmpl_col_has, coords, big_i], "tc_vec")
    tr_anchor = node("ReduceMin", [tr_vec], "tr_anchor", keepdims=0)
    tc_anchor = node("ReduceMin", [tc_vec], "tc_anchor", keepdims=0)

    row_out = node("Not", [row_in], "row_out")
    col_out = node("Not", [col_in], "col_out")
    row_out_u8 = node("Cast", [row_out], "row_out_u8", to=TensorProto.FLOAT)
    col_out_u8 = node("Cast", [col_out], "col_out_u8", to=TensorProto.FLOAT)
    side_rows = node(
        "Einsum", [non4_u8, col_out_u8], "side_rows", equation="abcd,d->abc"
    )
    side_cols = node(
        "Einsum", [non4_u8, row_out_u8], "side_cols", equation="abcd,c->abd"
    )
    total_row_has = node("Cast", [total_rows], "total_row_has", to=TensorProto.BOOL)
    total_col_has = node("Cast", [total_cols], "total_col_has", to=TensorProto.BOOL)
    side_row_has = node("Cast", [side_rows], "side_row_has", to=TensorProto.BOOL)
    side_col_has = node("Cast", [side_cols], "side_col_has", to=TensorProto.BOOL)
    key_row_has = node("Or", [
        node("And", [row_out, total_row_has], "key_row_outside"),
        node("And", [row_in, side_row_has], "key_row_side"),
    ], "key_row_has")
    key_col_has = node("Or", [
        node("And", [col_out, total_col_has], "key_col_outside"),
        node("And", [col_in, side_col_has], "key_col_side"),
    ], "key_col_has")
    kr_min_vec = node("Where", [key_row_has, coords, big_i], "kr_min_vec")
    kr_max_vec = node("Where", [key_row_has, coords, neg_i], "kr_max_vec")
    kc_min_vec = node("Where", [key_col_has, coords, big_i], "kc_min_vec")
    kc_max_vec = node("Where", [key_col_has, coords, neg_i], "kc_max_vec")
    kr_top = node("ReduceMin", [kr_min_vec], "kr_top", keepdims=0)
    kr_bot = node("ReduceMax", [kr_max_vec], "kr_bot", keepdims=0)
    kc_left = node("ReduceMin", [kc_min_vec], "kc_left", keepdims=0)
    kc_right = node("ReduceMax", [kc_max_vec], "kc_right", keepdims=0)

    # Scale from the gcd class of the first horizontal and vertical color runs.
    # The run vectors are only length 20; no additional image plane is created.
    row_at_tr_raw = node("Gather", [label, tr_anchor], "row_at_tr_raw", axis=2)
    col_at_tc_raw = node("Gather", [label, tc_anchor], "col_at_tc_raw", axis=3)
    row_at_tr = node("Where", [col_in, row_at_tr_raw, zero_u8], "row_at_tr")
    col_at_tc = node("Where", [row_in, col_at_tc_raw, zero_u8], "col_at_tc")

    def run_length(vector: str, prefix: str) -> str:
        nz = node("Greater", [vector, zero_u8], f"{prefix}_nz")
        is4 = node("Equal", [vector, four_u8], f"{prefix}_is4")
        valid = node("Xor", [nz, is4], f"{prefix}_valid")
        starts = node("Where", [valid, coords, big_i], f"{prefix}_starts")
        start = node("ReduceMin", [starts], f"{prefix}_start", keepdims=0)
        color = node("Gather", [vector, start], f"{prefix}_color", axis=2)
        same = node("Equal", [vector, color], f"{prefix}_same")
        before = node("Less", [coords, start], f"{prefix}_before")
        prefix_ok = node("Or", [same, before], f"{prefix}_prefix_ok")
        mismatch = node("Not", [prefix_ok], f"{prefix}_mismatch")
        mismatch_i = node("Cast", [mismatch], f"{prefix}_mismatch_i", to=TensorProto.INT32)
        failures = node("CumSum", [mismatch_i, zero_i], f"{prefix}_failures")
        in_run = node("Equal", [failures, zero_i], f"{prefix}_in_run")
        in_run_i = node("Cast", [in_run], f"{prefix}_in_run_i", to=TensorProto.INT32)
        prefix_count = node("ReduceSum", [in_run_i], f"{prefix}_prefix_count", keepdims=0)
        return node("Sub", [prefix_count, start], f"{prefix}_run")

    hrun = run_length(row_at_tr, "h")
    vrun = run_length(col_at_tc, "v")
    hmod4 = node("Mod", [hrun, four_i], "hmod4")
    vmod4 = node("Mod", [vrun, four_i], "vmod4")
    both4 = node("And", [
        node("Equal", [hmod4, zero_i], "hdiv4"),
        node("Equal", [vmod4, zero_i], "vdiv4"),
    ], "both4")
    hmod3 = node("Mod", [hrun, three_i], "hmod3")
    vmod3 = node("Mod", [vrun, three_i], "vmod3")
    both3 = node("And", [
        node("Equal", [hmod3, zero_i], "hdiv3"),
        node("Equal", [vmod3, zero_i], "vdiv3"),
    ], "both3")
    scale_pre = node("Where", [both3, three_i, two_i], "scale_pre")
    scale = node("Where", [both4, four_i, scale_pre], "scale")

    # v2: drop the 400-byte flattened label plane and the 1,600-byte 20x20
    # flat-index plane. Row and column offsets in this task are separable
    # (sr depends only on dr, sc only on dc), so a row-axis Gather followed
    # by a col-axis Gather reproduces the exact flat-index Gather with two
    # small vectors instead of one big INT32 plane.
    sr = node("Add", [tr_anchor, node("Mul", [scale, dr], "sdr")], "sr")
    sc = node("Add", [tc_anchor, node("Mul", [scale, dc], "sdc")], "sc")
    sc_clip = node("Clip", [sc, zero_i, nineteen_i], "sc_clip")
    sr_vec = node("Reshape", [sr, shape3], "sr_vec")
    sc_clip_vec = node("Reshape", [sc_clip, shape5], "sc_clip_vec")
    row_pick = node("Gather", [label, sr_vec], "row_pick", axis=2)
    kernel_label = node("Gather", [row_pick, sc_clip_vec], "kernel_label", axis=3)
    kernel_nz = node("Greater", [kernel_label, zero_u8], "kernel_nz")
    kernel_4 = node("Equal", [kernel_label, four_u8], "kernel_4")
    kernel_valid = node("Xor", [kernel_nz, kernel_4], "kernel_valid")
    kernel_scores = node("Where", [kernel_valid, score15, big_i], "kernel_scores")
    pick = node("ReduceMin", [kernel_scores], "pick", axes=[2, 3], keepdims=0)
    kernel_flat = node("Reshape", [kernel_label, shape15], "kernel_flat")
    anchor_color = node("Gather", [kernel_flat, pick], "anchor_color", axis=0)
    indicator = node("Equal", [label, anchor_color], "indicator")
    indicator_u8 = node("Cast", [indicator], "indicator_u8", to=TensorProto.UINT8)
    kernel_anchor = node("Equal", [kernel_label, anchor_color], "kernel_anchor")
    kernel_u8 = node("Cast", [kernel_anchor], "kernel_u8", to=TensorProto.UINT8)
    vote = node(
        "QLinearConv",
        [indicator_u8, qscale, zero_u8, kernel_u8, qscale, zero_u8, qscale, zero_u8],
        "vote",
        kernel_shape=[3, 5],
        pads=[0, 0, 2, 4],
        strides=[1, 1],
    )

    tr_rel = node("Sub", [tr_anchor, r_top], "tr_rel")
    tc_rel = node("Sub", [tc_anchor, c_left], "tc_rel")
    row_rel = node("Sub", [coords, kr_top], "row_rel")
    col_rel = node("Sub", [coords, kc_left], "col_rel")
    row_fit = node("And", [
        node("GreaterOrEqual", [row_rel, zero_i], "row_fit_ge"),
        node("LessOrEqual", [row_rel, node("Div", [tr_rel, scale], "row_fit_max")], "row_fit_le"),
    ], "row_fit")
    col_fit = node("And", [
        node("GreaterOrEqual", [col_rel, zero_i], "col_fit_ge"),
        node("LessOrEqual", [col_rel, node("Div", [tc_rel, scale], "col_fit_max")], "col_fit_le"),
    ], "col_fit")
    row_fit4 = node(
        "Reshape",
        [row_fit, init("shape_row", [1, 1, 20, 1], np.int64)],
        "row_fit4",
    )
    col_fit4 = node(
        "Reshape",
        [col_fit, init("shape_col", [1, 1, 1, 20], np.int64)],
        "col_fit4",
    )
    fit2d = node("And", [row_fit4, col_fit4], "fit2d")
    vote_masked = node("Where", [fit2d, vote, zero_u8], "vote_masked")
    best_cols64 = node("ArgMax", [vote_masked], "best_cols64", axis=3, keepdims=1)
    row_scores = node("ReduceMax", [vote_masked], "row_scores", axes=[3], keepdims=1)
    best_r64 = node("ArgMax", [row_scores], "best_r64", axis=2, keepdims=1)
    best_c64 = node(
        "GatherElements", [best_cols64, best_r64], "best_c64", axis=2
    )
    best_r = node("Cast", [best_r64], "best_r", to=TensorProto.INT32)
    best_c = node("Cast", [best_c64], "best_c", to=TensorProto.INT32)
    kor = node("Sub", [
        node("Sub", [tr_anchor, node("Mul", [scale, node("Sub", [best_r, kr_top], "best_r_rel")], "s_best_r_rel")], "tr_shift"),
        r_top,
    ], "kor")
    koc = node("Sub", [
        node("Sub", [tc_anchor, node("Mul", [scale, node("Sub", [best_c, kc_left], "best_c_rel")], "s_best_c_rel")], "tc_shift"),
        c_left,
    ], "koc")

    row_off = node("Sub", [row4, kor], "row_off")
    col_off = node("Sub", [col4, koc], "col_off")
    key_h = node("Add", [node("Sub", [kr_bot, kr_top], "key_hm1"), one_i], "key_h")
    key_w = node("Add", [node("Sub", [kc_right, kc_left], "key_wm1"), one_i], "key_w")
    row_valid = node("And", [
        node("GreaterOrEqual", [row_off, zero_i], "row_off_ge"),
        node("Less", [row_off, node("Mul", [key_h, scale], "scaled_key_h")], "row_off_lt"),
    ], "row_valid")
    col_valid = node("And", [
        node("GreaterOrEqual", [col_off, zero_i], "col_off_ge"),
        node("Less", [col_off, node("Mul", [key_w, scale], "scaled_key_w")], "col_off_lt"),
    ], "col_valid")
    stamp_valid = node("And", [row_valid, col_valid], "stamp_valid")
    src_r = node("Add", [
        node("Div", [node("Max", [row_off, zero_i], "row_off_pos"), scale], "src_r_rel"),
        kr_top,
    ], "src_r")
    src_c = node("Add", [
        node("Div", [node("Max", [col_off, zero_i], "col_off_pos"), scale], "src_c_rel"),
        kc_left,
    ], "src_c")
    src_r_clip = node("Clip", [src_r, zero_i, nineteen_i], "src_r_clip")
    src_c_clip = node("Clip", [src_c, zero_i, nineteen_i], "src_c_clip")
    src_r_vec = node("Reshape", [src_r_clip, shape20], "src_r_vec")
    src_c_vec = node("Reshape", [src_c_clip, shape20], "src_c_vec")
    row_gathered = node("Gather", [label, src_r_vec], "row_gathered", axis=2)
    stamp = node("Gather", [row_gathered, src_c_vec], "stamp", axis=3)
    stamp_color = node("Where", [stamp_valid, stamp, zero_u8], "stamp_color")

    row_edge = node("Or", [
        node("Equal", [row4, zero_i], "row_zero"),
        node("Equal", [row4, ohm1], "row_last"),
    ], "row_edge")
    col_edge = node("Or", [
        node("Equal", [col4, zero_i], "col_zero"),
        node("Equal", [col4, owm1], "col_last"),
    ], "col_edge")
    corners = node("And", [row_edge, col_edge], "corners")
    corner_stamp = node("Where", [corners, four_u8, stamp_color], "corner_stamp")
    out_rows = node("LessOrEqual", [row4, ohm1], "out_rows")
    out_cols = node("LessOrEqual", [col4, owm1], "out_cols")
    out_bbox = node("And", [out_rows, out_cols], "out_bbox")
    out_label = node("Where", [out_bbox, corner_stamp, neg_u8], "out_label")
    out_label30 = node("Pad", [out_label, pads30, neg_u8], "out_label30", mode="constant")
    node("Equal", [out_label30, out_colors], "output")

    graph = helper.make_graph(
        nodes,
        "task209_summary_first_v2",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
        producer_name="codex-task209-summary-first-v2",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = build()
    args.output.write_bytes(model.SerializeToString())
    print(f"nodes={len(model.graph.node)} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
