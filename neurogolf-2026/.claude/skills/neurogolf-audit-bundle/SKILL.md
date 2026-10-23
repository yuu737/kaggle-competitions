---
name: neurogolf-audit-bundle
description: "Run the standardized NeuroGolf safety audit on a submission ZIP, including 400-file structure, ONNX rules, official local cost, ORT examples, ARC-GEN sampling, and memorization-pattern detection. Use before submission, after merges, for score-collapse diagnosis, or for candidate-versus-main comparisons."
argument-hint: "[candidate.zip] [--baseline baseline.zip]"
---

# NeuroGolf Bundle Audit

Run from `${CLAUDE_PROJECT_DIR}` in Windows PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_audit.py" `
  "CANDIDATE.zip" --baseline "submission.zip"
```

Use `--arc-gen-limit 1` routinely and `-1` for the final expensive audit.
Read the printed `summary.json`; report SHA, estimated score/delta, invalid
tasks, new suspicious tasks, validation totals, and `ok`.

Do not promote, submit, or modify the confirmed baseline. A warning is
acceptable only when inherited from the baseline. Treat every new warning as a
blocker.

Read [references/outputs.md](references/outputs.md) when interpreting output.
