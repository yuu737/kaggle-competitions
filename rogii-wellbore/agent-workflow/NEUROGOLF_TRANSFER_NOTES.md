# Transfer Notes: NeuroGolf 2026 Top-Solution Patterns

For Fable5. Sourced from `neurogolf-2026` discussion/writeups (fetched 2026-07-18
via the nvidia-kaggle skill, cached in the plugin's local `discussions.db`).
NeuroGolf itself (ONNX-graph code-golf on ARC-AGI tasks) has nothing to do with
ROGII's regression problem — everything below is filtered for the
domain-independent *process* patterns that transferred across teams, with a
`→ ROGII` line where the parallel to our own postmortems is direct. Skip
anything not marked `→ ROGII` if you're short on time; those are the ones
worth actually acting on here.

## 1. Local score is a liar — only the real leaderboard resolves it

The single most repeated lesson across the top-40 writeups. Cheaper/higher
local score routinely lost on the hidden grader:

> "cost-15651, 265/265 local pass" scored LB 7306.85; "cost-44651" (worse
> local) scored LB 7321.14 and was kept. A 95-task harvest that graded 7205
> locally scored 7170 blind. ([41st place, silver, solo](https://www.kaggle.com/competitions/neurogolf-2026/writeups/41st-place-silver-solo-the-improvement-loop))

Two independent teams also found perfect-looking local passes that were
*mechanically* certain to fail hidden validation:
- A file can pass `onnx.checker`, shape inference, and a custom validator, and
  still be invalid: an under-length Conv bias reads past the buffer end,
  returning `0.0` on one machine and `8.7e-13` on another — same bytes,
  different runtime memory, different score. ([Georgy Mamarin, "the Conv bias nothing local flags"](https://www.kaggle.com/competitions/neurogolf-2026/discussion/712047))
- An `Einsum` cancellation that evaluates to exactly `0` on x86 can leave a
  `+ε` residual on ARM because of SIMD/FMA reduction-order differences; the
  scorer decodes with `output > 0`, so the residual silently flips a bit.
  Same model, same bytes, different hardware, different score. ([robga, 21st place, "representation collapse"](https://www.kaggle.com/competitions/neurogolf-2026/writeups/21st-place-solution-representation-collapse))

Both gate-calibration failure modes matter and point in opposite directions:
an over-strict local gate is a *silent cost tax* (you reject candidates the
real grader would accept), an over-lenient one is a *hidden-death trap* (you
adopt candidates that die on hidden data). The fix used by the winning teams:
submit small differential batches to bisect which specific candidate caused a
gap between expected and observed online gain, rather than trusting any local
proxy in isolation.

**→ ROGII:** this is exactly the CV↔LB gap already logged in
`[[rogii-lb-reality-2026-07]]` and the datum-scan V1 postmortem
(`[[rogii-datum-scan-v2]]`, "V1 LB 8.653 = SDF dead — cap-saturated noise on
hidden wells, fold-map inflation; only LB-verified parents trustable"). The
NeuroGolf teams' answer was procedural, not clever: treat every local
improvement as a *hypothesis*, never adopt an unverified change into the
anchor blend, and when a submission's aggregate score doesn't match the sum
of its parts' expected local deltas, bisect the submission (disjoint
sub-groups, compare deltas) to isolate the specific well/fold/parent that's
lying, instead of re-deriving trust from scratch each time.

## 2. Asset-ize the search: compounding notes beat a smarter model

The 1st-place team's biggest lever wasn't a bigger model, it was a two-layer
memory every worker reads before starting and appends to before stopping:

- **Global cookbook**: scorer rules, proven constructions, patterns that look
  valid locally but die on hidden data.
- **Per-task note**: inferred rule, current best cost, everything already
  tried, *why each dead end was a dead end*, and which model/when.

Without this, "every worker would spend a lot of tokens inspecting the task,
identifying its patterns... rediscovery" ([1st place pipeline overview](https://www.kaggle.com/competitions/neurogolf-2026/discussion/726799)).
The 9th-place team ran the same idea as `MEMORY.md` (compact, ~200 lines,
loaded every session) + `MEMORY_ARCHIVE.md` (full detail, searched on demand)
— "this worked much better than putting the entire history into every
prompt." The 41st-place solo entry adds one detail worth keeping: **tag each
note entry with which model made the claim and when** — "a confident 'this is
the floor' from Opus was routinely broken by Fable a week later... a dead end
is only a dead end for that model, at that time."

**→ ROGII:** this is what `EXPERIMENTS.md` / `MEMORY.md` / this memory system
are already meant to do. The actionable refinement is the *why it failed* +
*model/date* tagging discipline — several ROGII memories currently record the
verdict ("neighbor channels: dead") but not always the falsified hypothesis
in enough detail for a fresh session to avoid re-deriving it, or which
session/date closed it.

## 3. Concrete, provably-reachable targets break optimization plateaus

Open-ended prompts ("improve this for 2 hours") produced small grinding gains
until a rare "actually let's rewrite the architecture" leap. Giving a
concrete, externally-anchored target changed agent behavior completely:

> "Your goal is to improve this task by at least 1.5 points! It is possible
> and has been achieved by another competitor... if you think you can't beat
> it, please revisit your approach and keep trying." Agents started deriving
> a theoretical max *before* optimizing, instead of polishing dead ends.
> ([1st place pipeline overview](https://www.kaggle.com/competitions/neurogolf-2026/discussion/726799))

Same finding independently from 9th place ("gaslight AI that better score is
possible... this way it didn't spend time proving improvements are
impossible") and 41st place ("worst-lag: 'a competitor did X; beat it'").

**→ ROGII:** when delegating an experiment to `rogii-worker` or a fresh
Fable5 session, prefer "beat 6.300 on well-level CV, it's been done before on
a similar mechanism" framing over open-ended "try to improve this."

## 4. Architectural rewrite vs. incremental polish — track which pays off

"Architectural rewrites averaged +0.5 boost per task whereas optimizing
existing graphs averaged only +0.05... therefore agents were always
encouraged to explore and find better architectures. This avoided being stuck
in score plateaus." Corollary from 41st place: the highest-ROI ask was often
the *opposite* of "solve this from scratch" — profile the existing
best-known solution and delete its waste, since re-solving from zero mostly
rediscovers the incumbent.

**→ ROGII:** matches the `[[rogii-top10-status]]` finding that
neighbor-channel/selection/gold/datum-scan tuning around the existing PF
blend is measured dead, while the diagnosed gap (74% per-well LINE error —
datum+slope) is structural. Worth reframing future delegated work explicitly
as "structural rewrite of the datum/slope estimation" vs. "polish the
existing blend," and expect the latter to keep returning ~0.

## 5. Redundant, independent exploration beats efficient deduplication

Several top-10 teams deliberately ran the *same* task through independent
agents/prompts/starting points rather than splitting work to avoid overlap,
then kept the better (or merged) result:

> "Pavel and I did not split the 400 tasks in half... this duplicated some
> effort, but it also produced genuinely different representations." ([9th place](https://www.kaggle.com/competitions/neurogolf-2026/writeups/9th-place-gold-pavel-and-crodoc))

> "Several teams... did something I didn't: deliberate redundant exploration
> — two workers on the same task, then max-merging — which they report beats
> efficient non-overlapping search at escaping local optima. If I ran it
> again, that is where the next gain is." ([41st place](https://www.kaggle.com/competitions/neurogolf-2026/writeups/41st-place-silver-solo-the-improvement-loop))

The cost-0 rewrite in the 41st-place solution "existed only because a third
model disagreed with the first two" — independent multi-model review before
an irreversible adoption call caught something two prior models both missed.

**→ ROGII:** when a hypothesis is declared dead by one session/model, a
cheap independent re-check by a different model (not just re-reading the
memory verdict) is occasionally worth it before treating it as permanently
closed — especially for structural hypotheses like the datum/slope gap,
where the failure mode may be model-specific rather than physics-specific.

## 6. Diagnosing a hidden gap: bisect, don't guess

When a submission's aggregate score didn't match the sum of expected local
deltas, the standard move was binary search over the submission: split the
changed set into groups, submit variants differing by one group, compare
score deltas, repeat until the responsible item is isolated ([10th place](https://www.kaggle.com/competitions/neurogolf-2026/writeups/draft-10th-place-solution)).
This is generic and cheap whenever you can only observe an aggregate signal
(a leaderboard score, a CV fold average) and suspect one component is
misbehaving.

**→ ROGII:** applicable any time an OOF/CV aggregate diverges from a
component-level expectation and per-well or per-fold decomposition isn't
already isolating the culprit.

## 7. What *didn't* transfer (NeuroGolf-specific, skip)

ONNX operator tricks (`Einsum`-as-terminal-op, `ConvTranspose` as dynamic-weight
transport, fp32 ULP-collision selectors, ARC-AGI symmetry probing, MACs-are-free
cost formula exploitation) are specific to that competition's cost metric and
have no ROGII analog. Not included above.
