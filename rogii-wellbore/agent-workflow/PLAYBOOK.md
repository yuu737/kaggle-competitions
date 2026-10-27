# PLAYBOOK — Process Rules for Every Agent in This Repo

Audience: any AI agent (Fable, Opus, Sonnet, Codex, workers) doing competition
work here. These are domain-independent *process* rules distilled from the
NeuroGolf 2026 top writeups (1st/9th/10th/21st/41st place) — see
`NEUROGOLF_TRANSFER_NOTES.md` for sources and the ROGII-specific mapping.
They apply to ROGII and any future competition. Follow them by default.

## 1. Local score is a hypothesis, not a fact

Local/CV improvements routinely lose on the hidden grader, and locally-perfect
candidates can be mechanically dead on hidden data (environment-dependent
behavior local checks can't see). Both miscalibrations exist: an over-strict
local gate silently taxes you (rejects candidates the real grader would
accept); an over-lenient one adopts candidates that die blind.

- Never adopt an unverified local improvement into the anchor/blend.
  Only LB-verified parents are trustable (ROGII: datum-scan V1 postmortem).
- Treat every local delta as a hypothesis until a real submission resolves it.

## 2. When an aggregate score surprises you, bisect — don't guess

If a submission's score doesn't match the sum of its parts' expected deltas,
split the changed set into disjoint groups, submit variants differing by one
group, compare deltas, repeat until the lying component is isolated. Same
move applies offline: decompose CV/OOF aggregates per well / per fold before
theorizing.

## 3. Record *why* a dead end died, plus who closed it and when

A verdict alone ("X is dead") forces the next session to re-derive the
falsified hypothesis. Every experiment note must carry:
- the hypothesis that was falsified, in enough detail to not re-test it;
- the mechanism of failure (physics kill vs. noise vs. implementation bug);
- **which model made the call, and the date.** A dead end is only a dead end
  for that model at that time — "this is the floor" claims from one model
  were routinely broken by a stronger model weeks later.

Log in `EXPERIMENTS.md` (and memory files if you have them).

## 4. Delegate with concrete, provably-reachable targets

Open-ended prompts ("try to improve this") produce plateau-polishing.
Anchored prompts ("beat X — it has been achieved on a similar mechanism; if
you think it's impossible, revisit your approach") make agents derive the
theoretical max first and stop proving impossibility. Use the anchored form
whenever spawning a worker or a fresh session.

## 5. Distinguish structural rewrite from incremental polish

At a plateau, rewrites paid ~10x more than tuning the incumbent (+0.5 vs
+0.05 per task in NeuroGolf). Frame delegated work explicitly as one or the
other, and expect polish around a measured-dead area to keep returning ~0.
The inverse also pays: instead of re-solving from scratch (which mostly
rediscovers the incumbent), profile the existing best solution and delete
its waste.

## 6. Redundant independent exploration beats efficient deduplication

Running the same hard problem through independent agents/models/starting
points, then keeping or merging the best, beats splitting work to avoid
overlap — independent runs produce genuinely different representations.
Corollaries:
- Before an irreversible adoption call, get an independent second/third
  model's review; disagreement is signal.
- A hypothesis declared dead by one model deserves an occasional cheap
  re-check by a different (or newer) model before being treated as
  permanently closed — the failure may be model-specific, not
  physics-specific. (Exception: mechanically-killed hypotheses with a
  physics-level proof stay dead, e.g. ROGII external tops.)

## Non-transferable

Competition-specific exploits (NeuroGolf's ONNX operator tricks, cost-metric
gaming) do not generalize. When mining writeups from other competitions,
extract the *process* patterns above, not the domain tricks.
