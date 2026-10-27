# Workflow reference

## Windows PowerShell commands

Discover:

```powershell
& ".\scripts\neurogolf_public_refresh.cmd" discover --limit 30
```

Audit without updating:

```powershell
& ".\scripts\neurogolf_public_refresh.cmd" refresh `
  --kernel "OWNER/SLUG" `
  --baseline "submission.zip" `
  --baseline-score 7190.28
```

Audit and explicitly promote:

```powershell
& ".\scripts\neurogolf_public_refresh.cmd" update `
  --kernel "OWNER/SLUG" `
  --baseline "submission.zip" `
  --baseline-score 7190.28
```

Optional inputs:

- `--public-score`: record the observed public score.
- `--public-bundle`: reuse an already downloaded submission ZIP.
- `--output`: choose the unpromoted candidate path.
- `--username`: override the Kaggle username.

## Outputs

Every run creates run-specific score/validation reports and:

```text
artifacts/latest_public_refresh.json
```

The downloader requests only the final `submission.zip` with Kaggle
`--file-pattern`; it skips `_src_*`, `_stage`, traces, and per-task intermediate
outputs to prevent large-output timeouts.

Promotion preserves both a verified candidate and a timestamped SHA-named copy
of the previous root archive under `submissions/`. Root `submission.zip` is
replaced atomically only after backup and staged-copy hash checks pass.

Interpret `accepted`, `rejected`, `estimated_gain`, `predicted_score`,
`promoted`, and `*_invalid_tasks` from the manifest. A rejected
behavior-changing model requires separate rule analysis and validation.
