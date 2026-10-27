# Validation Protocol

No candidate enters a working submission until all applicable gates pass.

## Gate 0: rule quality

- Write a plain-language ARC transformation rule.
- Implement an independent Python/NumPy solver.
- Verify train, test, and all encodable ARC-GEN examples.
- Reject rules that depend on example identity or embedded expected output.

## Gate 1: ONNX integrity

```python
onnx.checker.check_model(model, full_check=True)
```

The graph must contain only allowed operators and canonical input/output
shapes.

## Gate 2: official sanitizer

Load `data/kagglehub/competitions/neurogolf-2026/neurogolf_utils/`
through the existing validation scripts. The sanitized model must pass the
checker and instantiate an ORT session.

## Gate 3: execution equivalence

For every available encodable example:

- compare to the expected one-hot output;
- compare official sanitized `(raw > 0)` output to the accepted baseline;
- record raw tensor equality separately as a diagnostic;
- record checked and skipped counts separately.

Different raw tensors are acceptable only when the official thresholded output
is identical and both models match every expected target. Never infer full
correctness from only the first differing example.

Fixed-size guard:

```python
encoded = encode_grid(example["input"])
expected = encode_grid(example["output"])
if encoded is None or expected is None:
    skipped += 1
    continue
```

Never pass `None` to ONNX Runtime.

## Gate 4: runtime compatibility

Test:

- current environment ORT;
- ORT 1.22.x in a separate environment;
- graph optimization disabled and enabled.

Avoid relying on a kernel newly supported by only the current ORT. Historical
task133 `TopK(UINT8)` passed a newer ORT and failed the Kaggle-like runtime.

## Gate 5: official cost

Use the official profiler through `scripts/score_bundle.py` or
`scripts/validate_rewrite_candidates.py`. Do not infer cost from ZIP size or
node count.

```text
gain = ln(old_cost / new_cost)
```

Record memory, initializer parameters, total cost, and gain.

## Gate 6: anti-overfit

```powershell
python scripts\check_overfit.py <candidate-or-bundle>
```

Manually inspect large Gather/lookup tables and constants derived from outputs.

## Gate 7: bundle audit

```powershell
python scripts\audit_onnx_bundle.py <submission.zip>
python scripts\validate_bundle.py <submission.zip> --arc-gen-limit 0
python scripts\score_bundle.py <submission.zip> --json-output score.json
```

Require:

- exactly task001 through task400;
- no missing or extra files;
- ZIP integrity;
- no new runtime/session failures;
- validation totals unchanged or intentionally improved;
- exact changed-task list;
- final SHA-256.

## Required decision record

```json
{
  "task": 123,
  "baseline_sha256": "...",
  "candidate_sha256": "...",
  "old_cost": 0,
  "new_cost": 0,
  "gain": 0.0,
  "examples_checked": 0,
  "examples_skipped": 0,
  "raw_equivalent": true,
  "current_ort": true,
  "ort_1_22": true,
  "sanitizer": true,
  "overfit_check": true,
  "decision": "verified"
}
```
