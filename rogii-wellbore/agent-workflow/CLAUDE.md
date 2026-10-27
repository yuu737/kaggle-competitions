# ROGII Workspace — Agent Instructions

Kaggle ROGII Wellbore Geology Prediction research workspace.
Entry point: `START_FABLE5.md` (mission, state files, reading order).

## Process rules (mandatory — full text in `PLAYBOOK.md`)

Distilled from NeuroGolf 2026 top-team writeups; domain-independent.

1. Local/CV score is a hypothesis, not a fact — only LB-verified parents are
   trustable; never adopt an unverified improvement into the anchor.
2. When an aggregate score surprises you, bisect (disjoint-group differential
   submissions / per-well decomposition) — don't guess.
3. Record *why* each dead end died, plus which model closed it and when —
   verdicts without the falsified hypothesis force re-derivation.
4. Delegate with concrete, achieved targets ("beat X, it's been done"), never
   open-ended "try to improve".
5. Distinguish structural rewrite from incremental polish; polish around a
   measured-dead area returns ~0.
6. Redundant independent exploration beats deduplication; get an independent
   model's review before irreversible adoption calls.
