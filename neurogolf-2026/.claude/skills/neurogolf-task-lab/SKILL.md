---
name: neurogolf-task-lab
description: "Prepare a compact reusable single-task NeuroGolf research packet containing ARC data statistics, extracted ONNX, graph metrics, operators, and node outline. Use before analyzing, redesigning, delegating, or asking an expensive model to optimize a specific task."
argument-hint: "<task-id> [--bundle submission.zip]"
---

# NeuroGolf Task Lab

Run from `${CLAUDE_PROJECT_DIR}`:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\neurogolf_task_snapshot.py" `
  TASK_ID --bundle "submission.zip"
```

Start with `artifacts/task-snapshots/taskNNN/README.md` and `snapshot.json`.
Inspect the extracted ONNX and raw examples only when analysis requires them;
do not repeatedly load all 400 models or the entire task corpus.

Keep candidates isolated. Compare them through `/neurogolf-merge-candidate`.
Never overwrite `submission.zip` or the confirmed baseline during exploratory
work.

Read [references/outputs.md](references/outputs.md) for snapshot contents.
