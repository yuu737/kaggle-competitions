---
name: neurogolf-build-handoff
description: "Regenerate and verify the compact NeuroGolf 7250 research handoff for Fable5 or another autonomous coding agent, including the confirmed main, data, tooling, builders, environment notes, and English autonomy prompt."
---

# NeuroGolf Handoff Builder

Run from `${CLAUDE_PROJECT_DIR}` in Windows PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\fable5_preflight.py" --full
if ($LASTEXITCODE -ne 0) { throw "Fable5 preflight failed" }
& ".\.venv\Scripts\python.exe" ".\scripts\build_fable5_handoff.py"
```

Verify that the produced ZIP contains exactly one clearly named confirmed
baseline, all 400 task models, task data, validation/scoring tools, and the
English instruction requiring Japanese replies. Exclude API keys, Kaggle
tokens, `.venv`, caches, and unrelated historical submissions.

Report path, SHA-256, size, baseline score/SHA, and preflight result. Never
modify the confirmed baseline, submit to Kaggle, or constrain the receiving
agent to a fixed optimization strategy.

Read [references/contract.md](references/contract.md) for the baseline contract.
