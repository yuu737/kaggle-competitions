# task029: actual measured cost vs C-8 design budget

## Result: ABORT (over budget by a wide margin)

| item | design estimate | actually measured |
|---|---:|---:|
| old cost (baseline, official) | 5,340 | 5,525 (memory 5,436 + params 89) |
| executable floor (design) | 2,300 | -- |
| corrected estimate (design, floor x1.4) | 3,220 | -- |
| acceptance ceiling | ≤3,956 | -- |
| abort threshold (build-time estimate) | >4,600 | -- |
| **candidate cost (measured)** | -- | **25,840** (memory 25,776 + params 64) |
| gain = ln(old/new) | design claimed +0.506 | **-1.543** (cost got 4.7x worse) |

## Root cause

The official memory formula (`neurogolf_utils.calculate_memory`) charges the
declared byte size of **every** named intermediate tensor in the graph except
the two tensors literally named `input` and `output`. This is the mechanism
that makes the baseline's `GridSample`-based crop comparatively cheap: the
`GridSample` node's own output **is** the graph's `output` tensor (free), so
only the small sampling-grid construction (`[1,30,30,2]` float16, plus a
handful of `[1,1,1,1]`/`[1,10]`-sized front-end scalars) is charged.

Any Gather/Slice/Pad/OneHot-based replacement that (a) must keep the final
`[1,10,30,30]` one-hot contract and (b) must handle a genuinely
data-dependent crop position/size, unavoidably needs at least one
intermediate tensor that touches the full 10-channel spatial data before the
terminal node, or an equivalent large one-hot re-expansion step
(`OneHot`/`Equal`-vs-class-ids) whose *input* index tensor must first be cast
to `INT64` (the only index dtype ONNX Runtime actually implements for
`OneHot`, verified empirically — `UINT8`/`INT32`/`FLOAT` indices all raise
`NOT_IMPLEMENTED`). That `INT64` cast alone costs `30*30*8 = 7,200` bytes.

Concretely, in the implemented candidate (`build_task029_v1.py`):

- `color_val_f` (Conv 1x1 channel-collapse, FLOAT `[1,1,30,30]`): 3,600 B
- `color_val_u8` (Cast to UINT8): 900 B
- `gather_rows` `[1,1,28,30]` UINT8: 840 B
- `gather_cols` `[1,1,28,28]` UINT8: 784 B
- `cropped_u8` (Where-masked) `[1,1,28,28]` UINT8: 784 B
- `padded_u8` `[1,1,30,30]` UINT8: 900 B
- `padded_i64` (forced by OneHot's ORT-implemented dtype) `[1,1,30,30]` INT64: 7,200 B
- `padded_i64_3d` (reshape, same bytes again): 7,200 B
- plus front-end scalars (kept identical to baseline, cheap) and small `[28]`/`[28,28]` index/validity vectors (~1,400 B)

Sum ≈ 25.8 KB, matching the measured `memory=25,776`.

## Conclusion

A Gather/Slice-based tail cannot beat the existing `GridSample`-based
baseline for this task under the official cost formula, because `GridSample`
already exploits the one structural discount the formula offers (terminal
node == graph output is free) about as efficiently as is achievable, while
any Gather/OneHot/Equal-based substitute pays for at least one full-size
intermediate before reaching the terminal node. The C-8 design's assumption
that a "4 scalars → dynamic Slice → payload directly to output" construction
would land near the 2,300-byte executable floor did not hold once official
per-tensor charging and ONNX Runtime's actual `OneHot` dtype-support
limitations were taken into account.

This is reported to Fable 5 as evidence; no further optimization of this
Gather/OneHot direction was attempted, since even removing every optional
byte identified above (~9,000 B of avoidable UINT8 intermediates) would still
leave cost well above 5,525 (the current baseline), let alone 3,956.
