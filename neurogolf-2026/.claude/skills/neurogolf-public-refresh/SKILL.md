---
name: neurogolf-public-refresh
description: "Automate the NeuroGolf public-top workflow: discover public Kaggle notebooks, download a selected public submission, score and compare all 400 tasks, validate cheaper models, build a best-of archive, audit it, and optionally promote it. Use when the public top changed, for public-versus-main differences, or for an explicit main update from public improvements."
argument-hint: "refresh|update [OWNER/SLUG]"
---

# NeuroGolf Public Refresh

Use the deterministic project script from `${CLAUDE_PROJECT_DIR}`.

1. Discover candidates when the public kernel is unknown:

   ```powershell
   & ".\scripts\neurogolf_public_refresh.cmd" discover
   ```

2. Select the actual highest-scoring public kernel from current evidence. Do
   not equate recency with score.
3. Match the command to user intent:

   - Compare/report only: use `refresh`.
   - Explicitly update the main: use `update`, which validates first and then
     automatically promotes accepted gains.

   Compare only:

   ```powershell
   & ".\scripts\neurogolf_public_refresh.cmd" refresh `
     --kernel "OWNER/SLUG" `
     --baseline "submission.zip" `
     --baseline-score CURRENT_CONFIRMED_SCORE
   ```

4. Explicit main update:

   ```powershell
   & ".\scripts\neurogolf_public_refresh.cmd" update `
     --kernel "OWNER/SLUG" `
     --baseline "submission.zip" `
     --baseline-score CURRENT_CONFIRMED_SCORE
   ```

5. Read `artifacts/latest_public_refresh.json`; report accepted/rejected tasks,
   gain, predicted score, SHA, validation, and `promoted`.
6. Never submit to Kaggle from this skill.

Require official-cost improvement, full ONNX checking, sanitizer compatibility,
sanitized ORT execution, target correctness, raw baseline equivalence, unchanged
validation totals, and no new memorization warning. Before every promotion,
create and SHA-check a timestamped backup of the exact prior root
`submission.zip`; atomically replace root only after the staged new copy also
passes SHA verification. Never print or package credentials.

Read [references/workflow.md](references/workflow.md) only for flags, artifacts,
and failure interpretation. Delegate only independent analysis; keep the
deterministic merge in the project script.
