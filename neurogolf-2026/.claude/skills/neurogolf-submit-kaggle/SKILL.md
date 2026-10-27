---
name: neurogolf-submit-kaggle
description: "Submit a specifically approved NeuroGolf ZIP to Kaggle, poll its result, and save an immutable SHA/result receipt. Use only for an explicit submission request; status mode provides read-only submission-history checks."
argument-hint: "status [--limit N] | submit --file ZIP --message TEXT"
disable-model-invocation: true
---

# NeuroGolf Kaggle Submit

Invoke this skill explicitly as `/neurogolf-submit-kaggle`.

Read-only status:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_kaggle_submit.py" `
  status --limit 10
```

Before submitting, audit the exact ZIP and record its SHA. Submit only after
explicit authorization:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_kaggle_submit.py" `
  submit --file "submission.zip" --message "SHORT_UNIQUE_DESCRIPTION"
```

Never infer submission permission from an audit, merge, or handoff request.
Poll until complete and report ref, public score, submitted SHA, and receipt
path. The daily competition limit is 100. Never print credentials and never
modify the submitted ZIP or confirmed baseline.

Read [references/outputs.md](references/outputs.md) for receipt locations.
