---
name: neurogolf-delegate-codex
description: "Delegate a bounded NeuroGolf implementation, scan, validation, or evidence-collection task from the Fable5 lead to an external Codex CLI worker, then consume its structured Japanese result. Use when parallel Codex execution would save Fable5 context or time."
argument-hint: "<task-file> [experiment|project]"
---

# Delegate NeuroGolf Work to Codex

Fable5 remains the research lead. Delegate bounded execution work; retain task
priority, ARC-rule judgment, candidate acceptance, portfolio integration, and
user communication.

1. Verify Codex CLI authentication:

   ```powershell
   & ".\.venv\Scripts\python.exe" ".\scripts\fable5_delegate_codex.py" `
     --check-auth
   ```

   If it reports `authenticated=false`, stop and ask the user to run
   `& "$env:APPDATA\npm\codex.cmd" login` once in an interactive terminal.
2. Write one task from
   [FABLE5/CODEX_TASK_TEMPLATE.md](../../../FABLE5/CODEX_TASK_TEMPLATE.md) to
   `FABLE5_WORK/codex-inbox/<descriptive-name>.md`.
3. Run from the project root in Windows PowerShell:

   ```powershell
   & ".\.venv\Scripts\python.exe" ".\scripts\fable5_delegate_codex.py" `
     --task-file ".\FABLE5_WORK\codex-inbox\<task>.md" `
     --title "<short-title>" `
     --scope experiment
   ```

4. Use `--scope project` only when the delegated task must edit project source.
5. Never pass `--allow-main-update` unless the user explicitly authorized that
   exact root-submission update.
6. Read the printed `result.json` and `manifest.json`. Check `guard_errors`,
   baseline hashes, validation evidence, and changed files before accepting the
   result.
7. Treat Codex output as evidence, not as the final research decision.

Codex runs noninteractively in `workspace-write` with approval prompts disabled,
an ephemeral session and structured output. Kaggle credentials are removed from
the Codex process. OpenAI authentication is available only to Codex itself and
is excluded from model-launched tool subprocesses. Codex may never submit to
Kaggle, publish notebooks, or mutate the confirmed baseline.
