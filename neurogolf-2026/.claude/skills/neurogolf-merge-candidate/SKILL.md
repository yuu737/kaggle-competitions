---
name: neurogolf-merge-candidate
description: "Safely extract verified lower-cost task models from a local NeuroGolf candidate ZIP and merge them with the current main. Use for best-of-two requests, technical-innovation candidates, checkpoints, or explicit requests to update submission.zip from another local bundle."
argument-hint: "<candidate.zip> [--baseline-score 7190.28] [--promote]"
---

# NeuroGolf Candidate Merge

Run the non-promoting comparison first from `${CLAUDE_PROJECT_DIR}`:

```powershell
& ".\scripts\neurogolf_merge_candidate.cmd" "CANDIDATE.zip" `
  --baseline "submission.zip" `
  --baseline-score CURRENT_CONFIRMED_SCORE
```

Inspect `artifacts/latest_public_refresh.json`. Report candidate tasks, accepted
tasks, rejection reasons, gain, predicted score, output SHA, and validation.

Add `--promote` only after explicit authorization to update the main. Before
promotion, record the current `submission.zip` SHA and preserve that exact ZIP
under `submissions/`. Never overwrite or recompress the confirmed 7190.28
baseline with SHA
`48f8381369be0e4474fd42b35cbe14b7ecc99d278fb1fd84d4ed1f74b3e0a515`.

Do not accept a model from cost alone. Require raw equivalence and every
validation gate. Read [references/outputs.md](references/outputs.md) for the
result contract.
