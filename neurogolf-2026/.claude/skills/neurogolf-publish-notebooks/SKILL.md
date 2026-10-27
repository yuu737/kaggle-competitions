---
name: neurogolf-publish-notebooks
description: "Build, inspect, and explicitly publish the five NeuroGolf public Kaggle notebooks as one series while keeping their shared public payload separate from the private main. Use for simultaneous notebook updates, medal-oriented revisions, status checks, or public payload refreshes."
argument-hint: "build|status|publish"
disable-model-invocation: true
---

# NeuroGolf Public Notebook Publisher

Invoke this skill explicitly as `/neurogolf-publish-notebooks`.

Run from `${CLAUDE_PROJECT_DIR}`:

```powershell
# Local build
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_public_notebooks.py" build

# Read-only Kaggle status
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_public_notebooks.py" status

# External publication; require explicit authorization
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_public_notebooks.py" publish
```

Treat the generator as the source of truth. Keep Notebook prose in English and
the review memo in Japanese. Verify all five use one public ZIP/SHA. Never
include the private main, private gain table, unpublished models, credentials,
or hints sufficient to reproduce the private score. Never alter the confirmed
baseline.

Read [references/series.md](references/series.md) for the series contract.
