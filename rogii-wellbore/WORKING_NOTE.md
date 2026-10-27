# Where the Information Ends

A falsification program on GR-based geosteering: eleven approaches, most of
them negative, that map exactly where a strong PF/stacker/spatial blend stops
being able to recover the per-well line — plus one late positive result that
locked well and then failed to confirm on the real leaderboard.

*Working Note — ROGII Wellbore Geology Prediction. Public LB 7.106.*

**TL;DR.** Our defended LB 7.106 submission is a PF/stacker/spatial blend plus
a small, separately scoped profile-consistency layer (§2); everywhere below,
"blend" means the PF/stacker/spatial system alone unless we say otherwise.
That blend leaves a per-well LINE error (datum + slope) accounting for ~74% of
its remaining pooled SSE. Every generalizable legal channel we measured against that
residual — cross-well transfer, GR-scan calibration, exact inference, loop
closure, steering geometry, identity-matched external records — turned up at
most a few weak directional echoes (the PF's own flattening-prior bias, corr
−0.24; a steering-geometry correlation, Spearman −0.29), none strong, stable,
or safe enough to fund a deployable signed correction; we can flag coarse
reliability context — lateral position, spatial support distance — but not
reliably pick out which specific wells need a signed correction, let alone
its direction. A small,
mechanistically clean correction built on the strongest of those echoes (the
flattening prior) locked at +0.090 on held-out folds, then scored 7.139 on the
real LB against our 7.106 anchor — a +0.033 RMSE regression (equivalently,
−0.033 on the "positive = improvement" delta convention we use throughout
this note) that sits inside our own ~0.1 LB-noise band (§3), so we do not
call it a confirmed defeat, only a failure to confirm the locked gain, which
is reason enough not to deploy it.
The profile-consistency layer already folded into 7.106 is the one exception
to that pattern, and we treat it as a narrow, separately scoped safeguard
rather than a generalizable answer. We treat the LB 7.106 anchor as our
defended final submission and report the
rest as a map of where this line of attack is dead.

## Abstract

We ran a systematic falsification program (51 recorded experiments, all with
predeclared kill criteria, well-grouped validation, and honest negative
results) around one question: **where does the recoverable information end in
GR-to-TVT inversion?** Our defended submission is the fixed PF + projection +
stacker + spatial blend of §2, plus a small profile-consistency layer
described there. For the rest of this note, "blend" refers to the
PF/stacker/spatial system on its own; we name the profile-consistency layer
explicitly wherever it is relevant, since it is scoped separately from the
generalizable science below. Everything in §5 is what we tried around and on
top of that system — eleven distinct lines of attack. Two conclude "keep
what §2 already does and stop tuning it further" (the PF's own internal
levers, and the spatial surface against its own alternatives) — they confirm
the deployed system rather than adding to it. One more — a small,
mechanistically motivated correction for the filter's own dip-flattening bias
— was built, validated on held-out folds, submitted, and then *failed to
confirm on the real leaderboard* (a delta small enough to sit inside our own
LB-noise band, §3); we treat that failure to confirm itself as one of our
central results, not a footnote. The remaining eight died outright, each for a
distinct, measured reason, and we report every one of them, because we believe
mapping the boundary of what a PF-heavy pipeline can extract is as valuable
here as any single score.

Our headline finding is a decomposition: 73.7% of a strong blend's pooled SSE
traces to a per-well LINE component (datum + dominant slope). Every
generalizable legal signal we could construct against that residual — cross-well transfer,
GR-scan calibration, exact Bayesian inference, within-well loop closure,
trajectory geometry, identity-matched real-world records — turned up at most
a handful of weak directional echoes (§5.10's flattening-prior bias,
corr −0.24; §5.11's steering-geometry correlation, Spearman −0.29), and none
of them was strong, stable, or safe enough to fund a deployable signed
correction. We can *detect* which wells are likely to carry a large line
error; we cannot reliably *act* on that detection. The one exception is the
narrow profile-consistency layer folded into our defended 7.106 (§2), which
we treat as a separately scoped safeguard rather than a generalizable
solution. This mirrors an independent finding in the community's own working
notes (§9), and we read the convergence as evidence that this boundary
belongs to the problem and this class of methods, rather than to any one
team's specific implementation.

**Map to what this note is trying to show.** §1–§2 fix the problem and the
deployed baseline. §3 states our validation discipline, including the LB-noise
band that turned out to matter more than we expected. §4 is the central
measurement: where a strong blend's error actually lives. §5 walks through the
eleven approaches themselves, structured so every one states what we tried,
what we measured, and what we concluded from it, including the one that
worked locally and failed to confirm on the board. §6
separates physically grounded design choices from pure metric optimization. §7 is our uncertainty
analysis (detection works, actuation fails). §8 closes with the map, and §9–10
credit the community work this note builds on and confirms, extends, or
narrows.

## 1. Task and problem reduction

Each of 773 train wells provides a horizontal-well CSV (MD, X, Y, Z, GR, six
formation-top elevation columns, TVT known before a prediction-start point and
hidden after it) and a paired typewell CSV (GR indexed by TVT, plus geology labels;
752/773 typewell files are unique by content hash). The task is to predict TVT on
every post-PS lateral row of ~200 hidden wells; the metric is RMSE pooled over all
lateral rows (not mean per-well RMSE), which makes rare catastrophic wells dominate.

The target is exactly determined by one geological surface: with the true top
columns, `TVT = ref − (Z − topZ) + c` reconstructs the label at **RMSE 0.005 ft**
(the six top columns are one degree of freedom; laterals cross on average 1.05
geology zones, so marker-transition logic has essentially no signal). The task is
therefore *surface estimation along the lateral*, from (a) the GR trace matched
against the typewell GR(TVT), and (b) cross-well spatial structure. Useful physical
state: `U = TVT + Z` (structural elevation). The GR→TVT inverse is multi-modal:
rhythmic Eagle Ford bedding repeats GR signatures roughly one bundle (±15 ft) apart,
so a wrong branch can fit GR better than the truth.

Reference points (773-well harness unless noted): carry-last-TVT floor **15.91**;
a single particle filter ~13; a multi-seed likelihood-weighted PF ensemble **~9.4**;
spatial interpolation ~12 standalone; our PF/stacker/spatial base blend scores
**7.68** pooled on a 151-well row cache; the defended final system, after the
separately scoped profile-consistency layer (§2), scores **7.106** on the
public LB. (The ~9.4 PF-ensemble figure above is a projected reference on the
773-well harness; it is not directly comparable to the candidate-cache
evaluator numbers quoted in §5.4.) Diagnostic caches drift in
size (150/151/153 wells, occasionally fewer after alignment checks) across the
campaign as the row cache was rebuilt and re-deduplicated at different points;
we never compare absolute RMSEs across two different caches, only the
before/after delta within one cache and one experiment.

## 2. The deployed system

All components descend from public-lineage code and public competition data.

1. **Particle filter** (the core signal): state = (U, structural rate); dynamics
   `rate' = 0.998·rate + N(0, 0.002²)`, `U' = U + rate'·ΔMD + N(0, 0.005²)`;
   emission = Gaussian GR mismatch against typewell GR interpolated at the
   particle's TVT, with per-well noise scale = std of the known-region mismatch
   clipped to [10, 60] API; 500 particles, systematic resampling at ESS < 0.5N;
   initialization at the PS anchor (σ=4.5 ft) with the known-region final rate
   (σ=0.01). Deployed as a likelihood-weighted ensemble over **128 seeds × 4
   likelihood scales (3, 5, 8, 12)**.
2. **Projection post-processing**: robust degree-4 IRLS fit of U vs normalized MD,
   blended 0.75 fit / 0.25 raw path.
3. **Tabular stacker**: ~196 engineered features, 3×LightGBM + CatBoost → Ridge
   meta-learner, trained with GroupKFold by well (~10.6–11 standalone OOF).
4. **Spatial surface predictor**: local-plane detrend of neighbor U samples + IDW
   (k=12, w=1/(d²+1)) of residuals, PS-anchored, support-distance gate ramping to
   zero between 300 and 1500 ft to the nearest offset well (~12 standalone).
5. **Fixed blend**: 0.3 stacker + 0.5 PF + 0.2 spatial at the row level, plus a
   conservative distance-gated spatial correction (weight 0.15) at the submission
   level. Fold-refit weights lose to these fixed weights on every fold.
6. **A conservative profile-consistency correction**: on a small subset of
   wells where the public evidence gives a strict, high-confidence
   identity-level consistency signal, we apply a bounded correction gated by
   that signal. This is a different, narrower check than the identity-matched
   real-world RRC records we tested and rejected in §5.7 — that route matched
   competition wells to public well-database records and failed on physics
   (median 42 ft residual); this one is an internal consistency check within
   the competition's own data and does not touch external records at all. Its
   strength was set the same way as the spatial-weight sweep in Appendix A: a
   small number of dedicated public-LB submissions at a few discrete settings,
   not a continuous search — the same "handful of deliberate, hypothesis-driven
   probes" §3 describes, applied here to a gate strength instead of a blend
   weight. Its measured contribution is larger than our own ~0.1 LB-noise band
   (§3), so we treat it as real rather than noise, though we withhold the exact
   magnitude along with the exact construction until after the competition
   closes, since either one alone would let a reader reconstruct the mechanism.
   We treat it separately from the generalizable PF/stacker/spatial system
   above because it is a profile-consistency safeguard for a small,
   high-confidence subset, not the scientific core of this note. Because the
   competition is still active, we disclose this layer only at the level
   needed to interpret the defended score, not at the level needed to
   reproduce or copy it; we therefore exclude it from the general scientific
   claims in §4–§8, where it enters this note only as a disclosed component of the
   defended submission score.

**Deployed components and validation evidence**, from the un-tuned floor to
the defended anchor:

| Component / stage | Evidence | Decision |
|---|---|---|
| Carry-last TVT | 15.91 | baseline |
| PF ensemble | ~9.4 | core |
| Projection (deg-4 IRLS) | consistent small gain | keep |
| Tabular stacker | ~10.6–11 standalone; useful blend leg | keep |
| Spatial surface | standalone ~12; the distance-gated term transfers to the LB | keep |
| Profile-consistency layer | public-LB contribution on a small, high-confidence subset, larger than our own ~0.1 LB-noise band | keep, scoped separately from the scientific core |
| Pocket bundle (§5.10) | locked +0.090 on held-out folds; failed to confirm on the public LB (Δ −0.033, inside our noise band) | failed to confirm; not deployed |
| **Final defended anchor** | **Observed public LB 7.106** | **defended** |

Everything else in this note — eleven approaches, most of which failed to
improve on this system — is the record of what we tried on top of it, including
one deterministic correction (§5.10) that locked positive and then failed to
confirm on the real board. Nothing beyond the six numbered components above
(PF, projection, stacker, spatial surface, fixed blend, profile-consistency
layer) is deployed; Carry-last TVT and the pocket bundle in the table above
are reference points, not deployed components.

## 3. Validation protocol

- **Split unit = physical well, always.** Pseudo-episodes derived from one well
  never cross folds. Spatial surfaces are rebuilt from training-fold wells only.
- **Tune/locked discipline**: thresholds and weights are selected on tune folds and
  evaluated once on untouched locked folds; a locked set becomes development data
  after inspection and is not reused as "final" evidence.
- **Predeclared kill criteria** for every experiment (written in the script header
  before running), with cohort design mixing ordinary, catastrophic,
  sparse-neighborhood, and control wells.
- **Reported metrics**: pooled RMSE (decides the competition), per-well
  distribution, fraction of wells improved, worst-well degradation, worst-5% SSE
  share, and residual correlation against deployed components.
- **Promotion gate**: ≥ +0.10 pooled on broad/full OOF, most folds improved, no new
  catastrophic wells, and useful error diversity (residual correlation with the
  blend < ~0.95). Subset gains below this bar repeatedly failed to transfer.
- **LB noise discipline**: byte-identical notebooks score across a ~0.08 band on
  the public LB (community measurement; our own same-version resubmissions differ
  by 0.001–0.065). We treat every LB delta below ~0.1 as unverifiable. We did
  not iteratively tune continuous parameters against the public LB; our
  submission history is a handful of deliberate, hypothesis-driven probes
  (confirm or reject a specific locked-fold finding, or — for the spatial
  weight and the profile-consistency gate strength, §2 — a small discrete
  dose-response sweep), not a continuous search over the board.
- **Cohort sizes vary by design, not by accident.** Numbers below range from
  the full 773-well harness down to 8-well gate cohorts, and diagnostic
  row caches drift in size (150/151/153 wells) as they were rebuilt across the
  campaign (§1). We compare before/after deltas within one experiment's own
  cohort, never absolute RMSEs across two different cohorts or caches.

We designed this protocol to prevent exactly the failure that later happened
anyway (§5.10): a gain that locks on held-out folds but is not confirmed by
the board. The protocol caught most of that risk; the one exception was a
borderline result we knowingly probed at the edge of our own promotion bar,
and we report its outcome in full rather than quietly not mention it.

## 4. Error anatomy: where a strong blend's error lives

Before describing what we tried, it is worth stating plainly what we measured
the blend's error to *be*, because every approach in §5 is aimed at some part
of this picture.

**The line dominates.** Decomposing each cache well's blend error into a LINE
part (the OLS line of the error vs measured depth — datum plus dominant slope)
and a residual SHAPE part, on 153 wells / 720k rows: 73.7% of pooled SSE is the
line (6.61 of 7.70 total; excluding the worst 5% of wells it is still 5.31 vs
3.45 — the line dominates the median wells too, not just the tail). Within the
line, slope drift dominates over pure offset: slope-only pooled 6.46 vs
offset-only 4.32; median |slope error| is 6.2 ft per lateral (p90 = 20.7), and
mean |error| grows from 3.3 ft at the heel to 7.7 ft at the toe. A hypothetical
perfect line would put the same blend at ~3.9 pooled — below the community's
per-well line oracle, because the tracker's within-well shape ("wiggle") is
already good. The battle, for this whole program, is the line.

![Figure 1. Where the error lives: 73.7% of the deployed blend's pooled SSE is the per-well LINE component (datum + dominant slope); the remaining 26.3% is within-well shape the tracker already handles well.](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34544823%2F2fec5023fbc24d8e78325114944465b2%2Ffig1_error_decomposition.png?generation=1783368727416653&alt=media)

**The tail is mostly a wrong-branch problem, not a noisy one.** Instrumenting
the PF (500 particles, logging per-step nearest-particle distance to truth and
the 5–95% posterior band) on the 8 worst wells by blend error plus 4 median
controls, with GR noise-scale multipliers {1, 2, 4} and an aliasing statistic
(fraction of lateral steps whose GR matches ≥2 disjoint typewell positions
within ±80 ft of truth): at the deployed likelihood width, the truth exits the
particle cloud entirely on 6/8 worst wells (median nearest-particle distance
5–55 ft; band coverage 0.04–0.36). Doubling the noise scale returns the truth
to the cloud on 8/8 and repairs 6/8 dramatically (56→23, 31→7, 34→12) — while
breaking one worst well (26→39) and one control (6.7→30). The aliasing
statistic is **saturated at 0.95–1.00 on nearly every well**, winners and
losers alike — the ambiguity is everywhere, so it cannot itself flag which
wells need help. Two candidate fixes each explain exactly one well: unbiased
island resampling (position-clustered survivor allocation ∝ mass^0.5 with
exact importance correction) is deploy-safe (controls ≤ +0.28 ft) and rescues
1/8 (27.5→12.2 — notably the well noise-softening breaks); rigid typewell
depth-shift registration explains one other (known-region mismatch −23%,
26.7→16.7) but fires wrongly on a control (4.8→8.0) and loses broad (151
wells: 3 improved / 5 worsened, −0.14).

**Reading the two measurements together:** the catastrophic tail is mostly the
geology's bimodal datum — on those wells the GR evidence prefers the wrong
branch *throughout the lateral* (an observation-model failure, not an
inference failure) — and the wrong-mode subset is not identifiable from
anything visible at inference time. This is consistent with the community's
independent cost-landscape measurement that among wells with a clear second
cost minimum, the cost-favored mode is correct only ~48.8% of the time with an
uninformative margin (§9). Every approach in §5 that tries to identify,
select, or route around this ambiguity is, in one form or another, attacking
this same wall.

## 5. Approaches we explored

The campaign produced eleven distinct approach families around the
system already described in §2. Two of them are folded into that deployed
system (the PF's internal levers were tuned and mostly rejected, confirming
what to keep; the spatial surface survived its own family of alternatives).
One more was built, validated, submitted, and failed to confirm on the board
— we consider it the sharpest lesson in the whole note. The remaining eight died,
each for a distinct, measured reason. Each subsection below spells out what we
set out to test, what the numbers showed, and what we concluded from it.

![Figure 2. The falsification map. Many methods, one boundary: risk is detectable, signed correction is not.](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34544823%2F853d1f9a20ac30032f7098f89509668b%2Ffig2_falsification_map.png?generation=1783368746716020&alt=media)

Every approach in §5.1–§5.11 lands in one of the three groups above, and every
failure points to the same boundary: coarse risk is detectable (§7), but a
generalizable signed correction is not — the narrow profile-consistency
exception aside (§2).

### 5.1 The particle filter core, and why we stopped tuning it

**Idea.** The PF (§2) is the system's only sequential Bayesian tracker; before
building anything on top of it, we swept its own internals to find its
ceiling.

**Validation.**

| Experiment | Evidence | Decision |
|---|---|---|
| Carry last TVT | 15.91 full | baseline |
| Slope integration | tune win, 15.97 full | reject |
| Viterbi/DP GR matching | objective prefers wrong branches | reject |
| PF ensemble (128 seeds × 4 likelihood scales) | ~9.4 on comparable sets | **core** |
| Robust deg-4 U projection | consistent small gain | **keep** |
| Beam search | optimal blend weight 0 | reject |
| Seed mean/median/trimmed aggregation | won at 100 wells, reversed at 200 | keep likelihood mean |
| GR affine calibration (value axis) | 10.67 → 13.15 | reject |
| Process/adaptive noise | much worse | reject |
| Spatial pseudo-measurement inside PF ("dip fusion") | 11.43 → 15.14+ | reject |
| Typewell peak proposal injection | collapsed to 21–27 | reject |
| Particle trajectory smoother | 7.69 → 7.97–8.01 (5-well smoke) | reject |
| Grid HMM smoother (position-only, fixed grid) | ~39–50 | reject — an implementation artifact, see §5.9 |

**Lesson.** Every internal PF lever we tried either did nothing or actively
hurt; the ensemble plus the degree-4 projection is close to this component's
ceiling, and the seed-aggregation reversal (100→200 wells) taught us early not
to trust small-cohort wins. We stopped tuning the PF at that point and put the
remainder of the campaign into what surrounds it.

### 5.2 Learned trackers as PF replacements

**Idea.** A single-trajectory or feedforward learner might read the GR log
more efficiently than a 500-particle Bayesian filter, or at least contribute
an independent error signal to the blend.

**Validation.**

| Model | Evidence | Decision |
|---|---|---|
| Tabular-feature TCN | promising subset; marginal/redundant vs strong blend (weight ~0.1, <1%) | do not deploy |
| Raw-sequence TCN | OOF 14.79 | reject |
| Cross-attention tracker | OOF 14.68 | reject |
| Cost-volume CNN | OOF 15.49 | reject |
| Open-loop GRU (cumsum dTVT) | val 18.34 vs PF 10.17 (drift) | reject |
| Closed-loop GRU ("learned PF", re-anchors each step) | val ~17.3 | reject |

A related idea, a multi-scale normalized-cross-correlation self-similarity
feature, scored 17.8 standalone and earned blend weight 0.00.

**Lesson.** Feedforward learners are PF-independent but too weak and
stacker-correlated (error corr 0.66–0.85); single-trajectory recurrent models
drift or collapse the multi-modal hypothesis set that the PF's particle cloud
exists to maintain. Multi-modal GR inversion needs a genuine population of
hypotheses, not a single learned trajectory.

**Decision.** Reject the entire family; do not deploy the one marginal
survivor (tabular-feature TCN) because its contribution is inside blend noise.

### 5.3 Learned inference wrapping the PF

**Idea.** Rather than replace the PF, improve its likelihood, its smoothing,
or its resampling with a learned component.

**Validation.**

| Experiment | Evidence | Decision |
|---|---|---|
| Learned PF likelihood (offline-supervised grid) | ranking improved (top-5 0.058→0.230, median true rank 27→14, candidate MAE 68.9→38.3) but locked PF 10.43 → **11.09** at frozen α=0.003 — repeated use amplifies calibration error | reject |
| Bidirectional CRF stage 1 | locked PF 10.05 vs MAP 10.48 / median 11.03 / entropy-gated 10.04; joined to the strong blend the selected α = **0.0** | reject before full training |
| Differentiable SMC (learned emission correction trained through the filter, soft resampling, path-level loss) | smoke passed formally (16.19→14.71, 2-seed eval) but **reversed at verification strength** (4-seed: baseline 14.07 vs 15.35–17.77) | reject |

**Lesson.** A learned component can improve an offline ranking metric while
making the actual deployed pipeline worse, because repeated use amplifies
whatever calibration error the learned piece carries. The differentiable-SMC
reversal is the sharpest instance: a 2-seed smoke test passed a model that a
4-seed verification run reversed by 3+ RMSE. **Weak evaluation gates on
stochastic filters lie** — this became a standing rule we held to for
everything that followed (§10).

### 5.4 Selection and routing — the wall

**Idea.** If we cannot fix the tail directly, perhaps we can at least *detect*
which wells need rescuing and route them to a better estimator. The oracle
headroom is large enough to justify real effort: full 773-well PF candidate
cache = 6,957 paths, fixed base 10.60 on the candidate-cache evaluator (not
directly comparable to the projected PF-ensemble ~9.4 reference in §1),
**oracle 7.44**; per-well blend-weight
oracle 4.48; blend-vs-decode selection oracle 6.43 vs 7.68.

**Validation.** Nine independent signal families, none harvestable:

| # | Signal family | Result |
|---|---|---|
| 1 | Per-well confidence gate (ens-std/dev/gs/loglik) | no usable correlation |
| 2 | Known-region config back-test selection | ~0 |
| 3 | Synthetic zero-shot branch ranker | 15.89 on real cache |
| 4 | Real-candidate LambdaRank (OOF) | 12.18 vs fixed 10.60 |
| 5 | Continuous/inverse-variance routers | lose to fixed blend |
| 6 | Tail-safety router (cross-fitted) | 7.685 → 8.46–8.49 |
| 7 | Conservative evidence gate | final gain +0.048, 1/4 accepted improved |
| 8 | GR-cost adjudication between two concrete paths | the decode fits GR at least as well on 147/151 wells (multi-modality guarantees a well-fitting wrong branch); corr −0.45 exists but not exploitable at well level |
| 9 | ESS-adaptive tempering + saturated ambiguity statistic | trigger fires on 4×10⁻⁵ of steps (collapse is gradual weight depletion, not a single-step crash); ambiguity statistic saturated 0.95–1.00 on nearly every well |

**Lesson.** Conditional information about which estimator to trust does not
survive the move from diagnostic to deployable, no matter which of nine
distinct channels we used to try to construct it. This is our own,
independently discovered version of what other teams working this same
problem call a "representable and not selectable" wall (§9) — a strong sign
that this boundary comes from the data and the bimodal geology itself, rather
than from any one team's modeling choices.

**Decision.** Reject every selector and router; treat this family as closed.

### 5.5 The branch-level rescue that works only when told where to act

**Idea.** §5.4 shows we can't *identify* which wells need help. A different
question: if we somehow knew, could we actually *repair* them? A per-well
global decode (Viterbi and posterior mean) on an absolute TVT grid, with
emission = GR mismatch + r(s)·spatial-anchor + (1−r)·PF-anchor, r =
clip((1500 − nn_dist)/1200, 0, 1), and Huber-smooth U-rate transitions.

**Validation.** On the gate cohort (8 worst + 4 controls): 8/8 worst improved
≥3 ft at every λ; worst-cohort pooled 27.6 → 8.5, reaching the spatial oracle
on the wells where spatial knew better (32.3 → 3.89 vs oracle 3.99); controls
degrade ≤ +1.2 ft; posterior mean beats MAP nearly everywhere. On the broad
151-well cache: the unconditional swap fails — untouched ordinary wells 7.33 →
12.70 (blend 6.65 → 10.31), 91/151 worsened; corr(spatial-reliability r, swap
gain) = **−0.005**. A flat-bottomed "branch-veto" emission variant (penalize
only >±tol deviations) fails at every tolerance tried (best swap 8.64 vs
deployed 7.68). As a blend component the decode is redundant with the PF
(row-level error corr 0.66–0.80; simplex weight 0.03; hedging α monotonically
worsens from 0).

**Lesson.** Repair-when-told-where works cleanly; *where* is exactly the
unsolvable part of §5.4. `nn_dist` predicts spatial error, not filter error,
so it cannot substitute for the missing selector. We report this as a clean
demonstration that the bottleneck in this entire program is *identification*,
not *repair*.

**Decision.** Do not deploy; keep as a diagnostic that sharpens §5.4's wall.

### 5.6 Physics-side observation fixes: calibration, correlation matching, and datum scans

**Idea.** Several "obvious next steps" attack the observation model directly
rather than the inference around it: fix the GR signal itself (calibration,
denoising), match it against the typewell more cleverly (windowed
correlation), or scan for the per-well offset explicitly.

**Validation.**

| Experiment | Evidence | Decision |
|---|---|---|
| Windowed-NCC whole-trace decode | scores 20.5 standalone; degrades 145/151 wells | reject |
| Heel gain/offset GR calibration fed into the PF | 120 wells: base 11.56 → 13.74 (−2.2 pooled) | reject |
| Rolling-median GR denoise into PF | 11.56 → 12.50 | reject |
| Constant-offset datum scan on the tail trace | −0.94 to −1.47 | reject |
| Calibrated bounded (offset, slope) scan of the blend path | localizes true offset within 3 ft on only ~30% of wells (median localization error 5.3 ft); pooled 7.69 → 10.0–11.1 | reject |
| Haar wavelet / log-spectrum emission features | one locked stat up reversed on tune; broad MAE up, local top-5 down (phase loss) | reject |
| Piecewise-linear structural post-fit | +0.014 (noise) — the error lives in tracking, not smoothing | reject |
| GR-missingness audit (real gaps: mean 32.9%, median 31.2% NaN across 100 wells, silently linearly-interpolated before a fixed likelihood-noise scale) | row-level corr(gap-distance, abs(err)) ≈ 0; well-level corr(mean gap, well RMSE) shrinks toward 0 as n scales — a small-sample artifact. Root cause: NaN runs are almost all 1–few rows; wells with >20 ft gaps are only 4/150. A gap-aware sigma-inflation variant showed a promising k-sweep that failed multi-seed replication | reject — not a contributor to the line error |

Two mechanisms are worth stating explicitly, because they explain *why* these
all fail rather than just recording that they did. **Windowed correlation
matching is level-blind on laterals**: a lateral window spans ~one
stratigraphic depth, so its GR variation is lateral facies heterogeneity plus
noise, not stratigraphic traversal; normalized correlation removes the window
mean — exactly the level information that is the signal.

**Legal heel calibration is harmful in our specific PF-heavy pipeline — we do
not claim it is useless in general.** Community work suggests that a properly
constructed heel gain/offset calibration can recover much of the coarse datum
(the oft-quoted ~82% shift-scan localization figure), and we do not dispute
that. Our narrower, measured negative result is about what happens when
calibrated GR is *injected into our sequential filter*: it is net-negative
because calibration shrinks the empirical noise scale, which sharpens the
likelihood and strengthens wrong-branch lock-in wherever the residual mismatch
is structural rather than affine (our filter needs its wide noise floor to
stay multi-modal) — harmful or redundant on top of a PF-heavy observation
model, not evidence against calibration as a method. Our own attempt to read a
per-well offset back out via a *legal* scan of the blend path sits far below
the oracle figure (gain/offset fit at the true TVT), which is the expected gap
between an oracle and a legal reconstruction of the same quantity, not a
contradiction of it. A GR-shape-only oracle registration recovers the slope at
correlation −0.00 in our setup. Sharpness and hypothesis-survival trade off in
our filter; that trade-off is what we measured, not a general verdict on
calibration.

**Lesson.** Every observation-side fix we tried either destroys the level
information it needs (NCC), sharpens the likelihood into wrong-branch lock-in
(calibration, denoising), or simply cannot localize the offset legally
(datum/line scans, ~30% at best). None of them touch the wall of §5.4; they
attack the observation model, and the observation model's ambiguity is the
wall.

**Postscript: a narrower, trigger-gated bimodal midpoint hedge, also killed.**
Public working notes on this competition report a soft hedge that helps: scan
a single datum offset Δ, find two well-separated, near-tied local minima of
the GR-mismatch cost, and nudge gently toward their midpoint only on
triggered wells (§9). This is not the 2-D full-grid scan above — it is
1-D, fires only when two minima are close AND an independent disagreement
signal is elevated, and moves with a small fixed weight (α = 0.15–0.2), never
a hard commit. On our tune split (100 wells) it passed only weakly (+0.070
pooled); on the locked split (50 wells, evaluated once) it reversed to
**−0.153**, with only 1 of 9 triggered wells actually improving. In our
pipeline the trigger condition is detecting the same near-universal GR
aliasing already documented in §4 and §5.4 (saturated 0.95–1.00), not a rare,
genuinely resolvable ambiguity — so even a soft, direction-agnostic-looking
hedge still depends on the midpoint being on the correct side of the base
prediction, and our legal signals do not tell us that (§7).

**Decision.** Reject the entire family, including the narrower hedge variant.

### 5.7 Leaning on the field beyond the well: spatial surfaces, external records, and test-test sharing

**Idea.** The Eagle Ford is one connected geological body. Neighbor wells,
public formation-top records, and even other predicted test wells should, in
principle, carry information about a given well's datum.

**Validation.**

| Experiment | Evidence | Decision |
|---|---|---|
| Isotropic spatial (deployed) | LB with = 7.106/7.107 vs without = 7.243 | **keep** (~40% local→LB transfer) |
| Anisotropic IDW | tune PASS (+0.607 on 20 wells) then full-resolution locked FAIL: iso 17.5996 vs aniso 17.6981, residual corr 0.9954, selected weight 0.0 | reject |
| Ordinary kriging | 12.50 vs matched IDW 12.24 | reject |
| External RRC formation tops, regional (gate A1) | real signal: residual corr −0.718, block-CV surface RMSE 150.8→113.1, 5/5 folds | pass gate |
| External RRC tops injected into deployed spatial (A2) | locked +0.086 (<0.10), far-quartile +0.039 (<0.30); raising λ blows tails +16 ft | reject |
| Multi-typewell PF observation ensemble | 120 wells equal-budget: base 12.24 vs multi 14.15, worst +52 ft; cross-typewell likelihoods incomparable | reject |
| Test-test information sharing | see below | reject |
| Identity-matched real-world RRC records | see below | reject |
| Formation-reference variants, marker logic | negligible (1.05 zones/lateral) | reject |

**Test-test sharing, in detail.** If hidden-test wells clustered in held-out
regions, their known-region U profiles and resolved laterals would be legal
surface samples for each other at submission time. A block-holdout
construction on train kills this cleanly: in interleaved geometry the shared
samples add ±0.1 (train neighbors at ~500 ft dominate); in 20k-ft block
geometry adding shared samples makes everything worse (gated 15.0 → 33–39) —
**even true-lateral oracle samples** — because each contributed sample
inherits its well's PS-registration constant, estimated off the sparse
remainder field (chicken-and-egg). The LB calibration argument closes the
case: our spatial component transfers to the hidden test at ~40% strength
(local +0.37 → LB +0.136); under block-clustered geometry the deployed
distance gate would be ~closed (mean gate 0.037) and the transfer would be
~0%. The hidden test is close to interleaved, where sharing is worthless.

**Identity matching, in detail.** The competition wells are real Eagle Ford
wells: a fixed offset maps the synthetic coordinates onto the Texas
state-plane frame (Karnes County), and 51 train wells match real API numbers
strictly on location, length, and azimuth (178 more at medium confidence, 336
loose). We parsed the public RRC wellbore database (dbf900, ~30M lines) end to
end: formation-top records (77k picks in Karnes/Live Oak) and — after
locating the undocumented field empirically — per-well reference elevations
with datum codes (10,470 records; GL/GR dominate, KB/DF minority). Pick
elevations = reference elevation − pick depth; residuals computed against
each matched train well's competition top columns, with one constant per
(formation, datum-code) removed. At strict-matched wells: median |well-level
residual| **42.4 ft** (σ = 324, p90 = 176); medium tier 81 ft; loose 84 ft.
Root cause: the global raw residual median is +1,797 ft — completion-report
tops are *along-hole* (MD) depths, and a horizontal well crosses its target
tops in the curve, adding a well-dependent MD–TVD gap of hundreds to thousands
of feet. Correcting it requires each well's directional survey, which exists
only as scanned filings, not bulk data. The route fails on physics before any
rules question arises: public records are 1–2 orders of magnitude too coarse
for a ±3–6 ft datum problem. This also retroactively explains the ~100 ft
"pick noise" in the A1/A2 regional experiment above.

**Lesson.** Structural interpolation from nearby *training* wells (the
deployed spatial surface) is real and worth keeping — it is the one channel
in this entire section that survives contact with the LB. Everything that
tries to extend that idea further out — anisotropy, kriging, external public
records, other predicted wells, or looser formation matching — either adds
nothing or is destroyed by a registration constant it cannot see.

**Decision.** Keep isotropic spatial; reject everything else in this family.

### 5.8 Measuring what's left of cross-well information, channel by channel

**Idea.** §5.7 showed the *deployed* spatial approach is close to exhausted.
Before giving up on cross-well information entirely, we measured each
remaining channel on its own terms: not "does adding a neighbor feature help
the blend" but "how much genuine information does this channel carry, and
where does it go."

**Validation.** All 773 train laterals; per-well OLS line of U vs MD;
"wiggle" = detrended residual. Pairwise nearest-point matching between
laterals (KDTree, distance-binned); leak-free IDW transfer (up to 8 neighbor
wells within 1500 ft, own well and suspected duplicates excluded); locked α
via 5 well-folds.

| Channel | Coherence | Marginal value vs deployed blend |
|---|---|---|
| Shape ("wiggle") | pair corr 0.78 @100–250 ft, 0.62 @250–500, 0.52 @500–750, 0.47 @750–1000; vs oracle line: 5.87 → 4.57 (+1.30 locked, α≈0.7) | corr with blend residual **0.064**; locked gain **−0.015**; only nn<250 ft rows carry signal (corr 0.254 → pocket +0.04) |
| GR field (Nadaraya–Watson GR(u) from neighbor lateral samples, ±3 ft vertical, ≤800 ft radius) | 61% coverage at ±15 ft | mode-discrimination accuracy **0.504** (coin flip) vs typewell **0.706** on the same 197k rows; rescue value where the typewell is wrong: 0.50 |
| Slope (IDW of neighbor line slopes, azimuth-aligned) | neighbor-IDW vs own slope corr **0.995** (regionally smooth dip field) | corr with the blend's slope error 0.29; locked gain **−0.034** — the neighbor estimate's noise is ~3× the needed ±6 ft/lateral precision |
| Levels | (deployed spatial, §5.7) | capped by per-well datum constants; anisotropy/kriging/external tops all fail |

**Lesson.** Cross-well data is rich in exactly the *shape* the tracker already
reads from the well's own GR (coherence 0.47–0.78 is a real, strong signal),
and empty in the *level and slope* evidence it actually lacks: the neighbor GR
field cannot even discriminate the bimodal branches above chance, and the
neighbor slope, despite an almost-perfect regional smoothness (corr 0.995), is
three times too imprecise for the ±6 ft/lateral the task needs. The blend's
own wiggle content is worth ~1.6 pooled on its own (replacing it with the
transferred wiggle over a line scores 9.29): the gap to the top tier is not
missing information in the blend's shape estimate — it is the line, again. We
are not claiming the datum is globally unrecoverable (§9 credits community
heel-calibration work that recovers real datum information under the right
construction); our narrower finding is about what is left over *after* our
PF/stacker/spatial base has already consumed the visible GR and spatial
signal — on that residual, risk is detectable (§7) but the signed correction
is not, in any of these four channels.

**Decision.** No cross-well channel deploys beyond the existing spatial
surface.

### 5.9 Exact inference on the filter's own model

**Idea.** If the PF's approximate, particle-based inference is the
bottleneck, replacing it with exact inference on the identical model should
help — or at least tell us whether depletion (not the model) is the problem.

**Validation.** Exact grid forward–backward on the deployed filter's exact
2-D state (position × momentum): 49 rate cells × 481 position cells in
*PF-path-relative* coordinates, log-space forward messages, identical
emission and dynamics constants; a 30-well cohort (8 worst + 4 controls + 18
random) against the deployed PF ensemble under the same projection/hold
protocol. Model equivalence check: exact filtered 17.34 vs PF ensemble 16.23
(the likelihood-weighted seed ensemble is a soft whole-path MAP, slightly
sharper than the true filtered mean). Depletion prediction confirmed: worst
cohort 23.16 → 19.38 (filtered) → **17.36** (smoothed). Deployment fails
anyway: controls 5.18 → 12.58, random wells 9.66 → 13.39, pooled −0.45.

**Lesson.** The exact multi-modal posterior mean hedges over surviving wrong
branches that particle depletion happens to prune on ordinary wells — the
filter's approximation error is an accidental beneficial prior, and the
estimator that would actually beat it (mode identification) requires
information the data does not contain (§5.4, §4). Exact inference is not the
cure, because inference was never the bottleneck.

**Implementation warning worth publishing.** Our first exact run — and, we
suspect, many rejected "grid HMM" baselines, including our own earlier one —
failed for a spurious reason: a fixed absolute position grid. U = TVT + Z
drifts by hundreds of feet along a two-mile lateral, silently pushing the
posterior mass off any fixed grid (our first run scored 62 pooled; the
earlier internal "grid HMM ≈ 39–50" rejection in §5.1 had the same flaw plus a
position-only state). Exact inference must run in path-relative coordinates
with the momentum state included before its verdict means anything — a
harness bug we are glad we caught before drawing the wrong conclusion from it.

**Decision.** Reject; keep as a mechanistic confirmation of §4's tail
diagnosis.

### 5.10 A deterministic correction that locked well — and failed to confirm on the board

This is the approach we want future readers of this note to remember most
clearly, because it is the one place our own validation discipline came the
closest to failing us.

**Idea.** Part of the line's slope drift (§4) looked like it had an
identifiable, mechanistic cause rather than a purely unobservable one: the
deployed PF dynamics mean-revert the dip state to *zero* (time constant ~500
steps — a "wells flatten" belief), and we could measure the resulting bias
directly: corr(slope error, initial-dip × lateral length) = −0.24. If the bias
has a known sign and a known cause, it should be correctable with a small
cross-fitted term, combined with the near-range wiggle signal identified in
§5.8 (nn<250 ft rows, corr 0.254).

**Validation.** A cross-fitted linear correction (pred += γ·ir·(md−md₀), γ ∈
0.008–0.016 across folds) recovers +0.062 pooled locked on its own. Combined
with the near-range wiggle term, the deterministic "pocket" bundle locked at
**+0.090 pooled** on held-out folds (7.692 → 7.602; tails unchanged; worst
well −2.6 ft). It did not meet our formal +0.10 promotion gate (§3); it was
close enough, and mechanistically clean enough, that we submitted it as a
deliberately labeled borderline probe rather than a promoted replacement,
specifically to test whether our gate was too conservative.
**It scored LB 7.139 versus the 7.106 anchor** — a +0.033 RMSE regression
(−0.033 on the improvement-delta convention used throughout this note) that
sits inside our own ~0.1 LB-noise band (§3), so we do not call this a
confirmed defeat. What it unambiguously is: a failure to confirm the locked
gain. A
+0.090 locked gain sits near the edge of our own ~0.1 LB-noise threshold, so
if it transferred cleanly it should have produced at least some supportive
evidence on the board; instead, the board result was indistinguishable from
noise. Treating a near-miss locked gain as "close enough"
bought us an unverifiable outcome instead of a confirmed win — exactly the
failure §3's discipline exists to prevent.

![Figure 3. A locked gain that failed to confirm on the real leaderboard: a mechanistically clean pocket correction locked at +0.090 on held-out folds, short of our +0.10 gate (left), then scored 7.139 on the public LB against the 7.106 anchor — a Δ that sits inside our own ~0.1 LB-noise band (right).](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34544823%2Fef31374d8a19a94753609ea355cb3ff1%2Ffig3_pocket_reversal.png?generation=1783370970350098&alt=media)

**Lesson.** A locked local gain in the 0.06–0.10 range is not yet
distinguishable from the ~0.08 public-LB noise band (§3) until confirmed by an
actual submission, not just a locked-fold pass — no matter how mechanistically
clean the story behind it feels. We had a real, measurable, physically
motivated bias (the flattening prior); we had a real, cross-fitted correction
for it; and submitting it did not produce a confirmed improvement, because
"real and measurable" is not the same bar as "large enough to clear LB noise."
This is the sharpest instance of a pattern that recurs throughout this note
(§5.3's weak-eval-gate reversal is the same lesson from a different angle),
and it is why we now treat the whole line component as *unactioned*, not
*partially solved* (§8).

**Decision.** Not deployed. The pocket bundle is not part of our submitted
system (§2); we report it in full because the failure is more informative
than a quiet omission would have been.

**Postscript: fixing the bias at the source, instead of patching around it.**
If the flattening prior is real, why not weaken it directly, at its origin,
rather than adding a correction on top of the filter's output? We swept the
PF's rate mean-reversion coefficient (`mom` in §2, default 0.998, implying a
~500-step decay toward zero dip) up toward 1.0 on a 180-well tune/locked
split. Raising it to 0.999 improved tune pooled RMSE by +0.127 — but created
4 newly catastrophic wells with worst-case degradation over 16 ft, violating
our own predeclared catastrophic-well limit; no variant we tried cleared both
criteria at once, so nothing was promoted to the locked set. We read this as
confirmation that the 0.998 decay is not simply a bug: it is a safety valve
against wells whose known-region heel gives a noisy or wrong initial dip
estimate. Letting that estimate persist further into the lateral helps the
wells where it happened to be right and actively wrecks the wells where it
wasn't — the same trade-off, from the source side, as the pocket bundle's
failure from the correction side. Whether a well's initial dip estimate can be
trusted is exactly the kind of well-level judgment call our nine-fold
selection null (§5.4) already showed we cannot make.

### 5.11 Within-well long-range information: loop closure and steering geometry

**Idea.** Two further ideas try to read the line from the well's *own* data at
long range, without needing any neighbor at all. First: geosteering-as-SLAM —
laterals re-cross the same stratigraphic depth at different points along
their length, so GR-vs-TVT profiles of distant MD segments should
cross-correlate and expose accumulated drift. Second: the driller steers on
real-time geology, so the trajectory itself (inclination, build rate,
correction events, tortuosity) may encode the true structure drift the model
misses.

**Validation.**

| Experiment | Evidence | Decision |
|---|---|---|
| Loop closure | Geometry gate PASS on 773 wells: a median 83% of a lateral's TVT range is re-crossed within 0.5 ft at ≥1,000 ft MD separation (MD–TVT rank corr 0.04 — wells wander, they do not sink monotonically). Detection gate FAIL: at corr ≥ 0.90 with a lag-ambiguity margin, coverage collapses to ~1 accepted pair per well, and accepted offsets carry no information — corr(estimated, true segment-drift) = **0.066**, p90 error 13 ft | reject |
| Steering geometry | 150 wells, 12 features vs the per-well line error: best single feature = net dZ/dMD, Spearman −0.29 vs slope error (p ≈ 3×10⁻⁴ — real, and consistent in sign with the flattening-prior bias of §5.10); every steering-activity feature ≈ 0; leave-one-out ridge over all features R² ≈ 0. Univariate exploitation ceiling measured at +0.034 pooled — half the LB noise band | reject |

**Lesson.** Even the well's own long-range information is exhausted, not just
cross-well data. The loop-closure geometry exists (wells do re-cross their own
depth), but within-well GR-vs-TVT profiles are too low-relief over a ±12 ft
window for high-correlation matches to be anything but smooth ramps that
align at any lag — the well's own GR cannot sense its own drift, consistent
with the ~30% legal-scan localization of §5.6. The steering-geometry echo is
real (p ≈ 3×10⁻⁴) but too small and too univariate to fund a correction.

**Decision.** Reject both; a physically interesting confirmation of the wall,
not a way through it.

## 6. Where physical grounding ends and metric-driven choices begin

Every component of the deployed system (§2) descends from a physical model:
the PF is a sequential Bayesian tracker of a real structural state; the
spatial surface is a local structural interpolation gated by real support
distance; the projection is a smoothness prior on a physically smooth surface.
A few choices exist purely to minimize the metric — the fixed blend weights,
the distance-gate thresholds — and for those we kept the free parameters few,
bounded, and picked under transfer discipline (cross-fitted, evaluated once on
locked folds) rather than continuously tuned to the public board; the only
exceptions are the explicitly disclosed small discrete LB probes for the
spatial weight and the profile-consistency gate strength (§2, §3, Appendix A).

Our practical rule was strict: for the generalizable PF/stacker/spatial
components, nothing reached the deployed system without first passing
untouched-fold validation against a threshold fixed in advance, and even a
positive locked result stayed provisional until an actual submission confirmed
it (the profile-consistency layer is treated separately, as a narrowly scoped
submission safeguard, §2). §5.10 is the clearest test of that discipline —
the correction passed the mechanistic and safety checks but fell just short
of the formal +0.10 promotion margin, so we submitted it only as a labeled
borderline probe; it then failed to earn the one confirmation
that would have justified deploying it. We report that failure to confirm
rather than either suppressing it or lowering our own bar after the fact to
justify keeping it deployed. We would rather publish a defended 7.106 than an
unconfirmed 7.139 dressed up as an improvement.

## 7. Uncertainty: what the system knows about its own reliability

The deployed system expresses confidence in three mechanisms, and our program
spent a large fraction of its experiments testing whether that self-knowledge is
real. The honest answer is mixed, and we consider it one of our central
findings: **coarse reliability context is detectable; generalizable signed
actuation on the per-well line residual mostly is not.** We can flag *coarse*
reliability context — lateral position (heel vs. toe), spatial support
distance — with real, quantified signal; we do act on that, through the
conservative fixed blend weights and the distance-gated spatial correction
(§2). What we cannot do is identify *which individual wells* carry the large
line error well enough to select or route them (§5.4's nine-fold null), or
turn any of our reliability measures into the *sign or magnitude* of the
missing datum/slope correction — even where a weak directional echo exists
(§5.10, §5.11). Outside the narrow profile-consistency layer of §2 (a strict,
high-confidence exception scoped to a small subset of wells, not a
generalizable estimator), every attempt to convert risk detection into a
signed correction of the line residual failed.

**Where uncertainty is encoded.** (1) The spatial component's support-distance
gate (weight ramps to zero between 300 and 1,500 ft to the nearest offset well)
is an explicit prior reliability model, and it is *calibrated in the right
direction*: spatial error grows with support distance, and the gated blend
transfers to the hidden LB at ~40% of its local strength (§5.7). (2) The PF is
deployed as a likelihood-weighted ensemble over 128 seeds × 4 likelihood scales —
a soft self-weighting over hypotheses that empirically beats every hard
aggregation we tried. (3) Blend weights are fixed conservative values chosen
under transfer discipline rather than fold-refit optima, precisely because
fold-refit confidence did not survive contact with held-out wells (§3).

**Where predictions are reliable, quantified.** Error is position-dependent in a
predictable way: mean |error| grows from 3.3 ft at the heel to 7.7 ft at the toe
(slope drift accumulates, §4), so downstream users should trust the first third
of a lateral roughly 2× more than the last third. Across wells, reliability is
dominated by the bimodal-datum geology: the worst 5% of wells carry 35% of the
pooled SSE, and on those wells the GR evidence itself prefers a wrong
stratigraphic branch across the entire lateral (§4).

**Where self-knowledge fails — measured, not assumed.** The filter's own
posterior is *overconfident exactly where it is worst*: on 6/8 catastrophic
wells the truth leaves the particle cloud entirely (5–95% band coverage
0.04–0.36), so the posterior band is a lower bound on true uncertainty, not an
estimate of it. And no inference-time statistic we could construct — ensemble
spread, log-likelihood, ESS dynamics, an aliasing/ambiguity statistic (saturated
at 0.95–1.00 on nearly every well), known-region back-tests, GR-fit adjudication
between concrete alternative paths, learned rankers (§5.4) — ranks wells by their
true error well enough to act on. Nine independent nulls converge with the
community's cost-landscape measurement (the cost-favored mode is right ~49% of
the time on two-minimum wells): **on bimodal wells the data does not license
confidence in either mode**, and a system that reported honest per-well
uncertainty would have to report near-total ignorance on precisely the wells
where it matters most. Our deployment choices (conservative fixed weights, gated
spatial, no selectors) are the operational form of that statement: we act on
detection (widen the noise floor, gate the spatial weight, avoid hard selection)
and refuse to act on actuation (no per-well signed correction, because we have
no reliable evidence for its direction).

## 8. Conclusion: the map

The pooled error of a strong blend decomposes into (a) a per-well LINE component —
74% of SSE — that, *after our PF/stacker/spatial base has already consumed the
visible GR and spatial signal*, none of our generalizable legal signals could turn into a
deployable, safe correction on the remaining residual (three GR-scan nulls,
four cross-well channels, nine trust-arbitration nulls, exact inference, a
SLAM-style within-well loop-closure gate, trajectory/steering geometry, and
identity-matched real-world records — §5). This is not the same as zero
signal: a few weak, statistically real directional echoes turned up (the
flattening-prior bias, corr −0.24; the steering-geometry correlation,
Spearman −0.29), and one narrow, high-confidence profile-consistency safeguard
(§2) does contribute a material, deployed correction on a small subset of
wells. But none of the generalizable channels were strong, stable, or safe
enough to fund a correction beyond that narrow exception. We are not claiming
the line is unrecoverable in principle — community heel-calibration work
recovers real datum information under the right construction (§9) — only that
nothing generalizable we tried recovers the slice of it still left in our
base's residual. We did find a small, mechanistically clean slice of it — the
filter's own flattening prior — and a cross-fitted correction for that slice
locked at +0.090 on held-out folds, but failed to confirm on the real LB
(7.139 vs the 7.106 anchor, a delta inside our own noise band, §5.10); we
count the line component as unactioned, not partially solved. (b) an
irreducible bimodal tail
that our particle-cloud instrumentation and the community's cost-landscape
analysis independently agree is close to a coin flip; and (c) within-well shape
the tracker already handles well. Within this program the per-well-data ceiling
for filter-plus-blend methods sits near the public fork cluster, while the
strongest reported per-well results (~5.4) imply a matcher whose mechanism our
falsification grid did not reach. We hope the boundary mapped here — what is
recoverable, what is noise, and which obvious next steps are already dead — saves
other teams the compute we spent finding out.

## 9. Community context and credits

Several of our measurements confirm, quantify, or correct findings first raised in
public community work, and we want the credit assignment to be explicit:

- **The bimodal-datum geology** (rhythmic Milankovitch limestone–marl couplets
  at the ~15–25 ft scale, and the finding that the cost-favored mode is right
  only ~48.8% of the time among two-minimum wells) is souldrive's account.
  Our particle-cloud instrumentation (§4) reaches the same verdict from the
  inference side.
- **The oracle ladder** (flat ≈ 16, per-well constant ≈ 9, line ≈ 6.6, smooth
  ≈ 3) and the calibration-degeneracy analysis are georgymamarin's: a naive
  legal fit can collapse to shuffle-level (~8%), while a properly constructed
  heel calibration recovers much of the coarse datum, close to the oracle
  shift-scan localization (~82%). Our error decomposition (§4) mirrors that
  ladder at the per-well-error level; our own calibration result (§5.6) is
  narrower and scoped to what happens when calibrated GR is fed specifically
  into our PF-heavy pipeline, not a claim about legal calibration in general.
- **An acquisition-physics account of the same wall**: public work on the
  wireline-to-LWD GR response operator and its section-resolved coherence
  spectrum (attributed to Anthony Yanza) reports very low toe coherence across
  the fine (~5–32 ft) band needed to break the bimodal tie — an instrument-side
  reason the fine structure is largely missing. This is an independent
  physical mechanism for our own finding that the line/slope residual carries,
  at best, weak directional echoes too small and unstable to deploy in any
  channel we measured (§5.8, §5.11).
- **A matching "unobservable datum" framing**: other public working notes
  converge on the same detection/actuation split we report in §7 — presence
  detectable, sign not — reached independently from a GBM-heavy pipeline
  rather than our PF-heavy one. The convergence suggests this boundary comes
  from the problem itself, rather than from either pipeline.
- **Hedging between bimodal positions** (posterior mean dominating both the
  midpoint and hard commitment) was sharpened by pilkwang; our RMSE-convexity
  reading is consistent with it.
- **The legality of whole-trace look-ahead** (post-drilling analysis) was settled
  in the public rules thread; the claim that the local `test/` folder is a
  placeholder replaced at scoring, and that the hidden test is scored honestly,
  was confirmed publicly by the host and by rank-28 (Ioannis M.).
- The strongest reported per-well result (~5 per well with per-well data only,
  by the rank-2 competitor) is quoted from their discussion post; we did not
  reproduce it, and our program's failure to reach it from the filter-plus-blend
  side is part of the map in §8.

## 10. Lessons for practitioners

1. **Instrument the estimator, not just the score.** The single most valuable
   diagnostic we built logs the particle cloud against the truth per step (§4). It
   converted "the filter is bad on some wells" into two mechanistically distinct
   failure classes (depletion vs observation-model preference for a wrong branch)
   with different — and, it turned out, mutually exclusive — remedies.
2. **Weak evaluation gates on stochastic filters lie.** A 2-seed smoke evaluation
   passed a learned-emission model that a 4-seed evaluation of the same
   checkpoints reversed by 3+ RMSE (§5.3). Gate compute is the cheapest place to
   overspend.
3. **A locked local gain is not a result until a submission confirms it.**
   The dip-bias/wiggle pocket bundle locked +0.090 on held-out folds — just
   short of our own promotion gate — and failed to confirm on the real board,
   scoring a −0.033 delta that sits inside our own LB-noise band (§5.10). We
   now treat any locked gain under ~0.10 as provisional, not final, regardless
   of how
   mechanistically motivated it is.
4. **Exact-inference baselines fail for spurious reasons.** A fixed absolute grid
   silently loses the posterior of a drifting state (§5.9); we nearly closed a
   whole method family on an implementation artifact, and we suspect published
   "HMM baseline" rejections elsewhere share it.
5. **Distinguish an oracle from a legal signal, and a naive legal signal from a
   properly constructed one.** The community's own numbers span all three: an
   oracle calibration near 82%, a naive legal fit near shuffle-level (~8%), and
   a properly constructed legal heel calibration close to the oracle (§9) — our
   own legal scan (§5.6) landed near ~30%, a different point on that same
   spectrum, not a refutation of it. Separately, our own tune-time PASSes
   (anisotropic spatial, seed aggregation) collapsed once tuning leakage was
   removed on full/locked evaluation — a distinct failure mode from the
   oracle/legal gap. Our rule either way: any gain not reproduced on untouched
   wells with predeclared thresholds does not exist.
6. **Pooled RMSE demands tail accounting.** Every promotion decision needs the
   worst-well and worst-5%-SSE view; several "average" improvements were tail
   regressions in disguise (§5.4, §5.5).
7. **Measure information channels, not methods.** The channel-level measurements
   of §5.8 (shape coherent-but-redundant; GR field empty; slope imprecise) closed
   dozens of would-be method variants at once — far cheaper than testing the
   methods individually.

## Appendix A. Chronological experiment index (51 entries)

Phase 1 — system construction (June):
baselines (hold, slope, DP-Viterbi); in-house GR trackers; particle-filter core;
hold-blend + confidence-gate null; first LB submission (12.130); LB-7159 component
integration (beam null, projection win); the tvt_from_contacts 0.005 reduction;
spatial predictor; spatial blends; stacker de-black-boxing; augmented stacker and
the re-blending plateau; spatial-cap study; self type-log null; routing oracle
study (4.48 unreachable); TCN signal study; TF/Keras TCN + 4-way blend;
LB spatial-weight sweep (plateau confirmed on the real board).

Phase 2 — plateau falsifications (late June):
three independent learned trackers (xattn / cost-volume / rawseq); PF-internal
sweep (aggregation, dip fusion, calibration, noise); geology-domain investigation;
offline smoother + proposal injection probes; synthetic branch ranker audit;
public model-package OOF blend (and its LB refutation); anisotropic spatial
phase 1 (and its full-resolution reversal).

Phase 3 — first-principles program (July 2–3):
takeover audit (deadline correction, tail quantification); external-tops gate A1
and injection A2; multi-typewell PF; collapse anatomy + ESS tempering + ambiguity
saturation; spatial-anchored decode (gate PASS, broad FAIL, adjudication null,
veto variant, component/hedge closure); field-estimation ceiling gate G1;
island-resampling PF; typewell depth registration (cohort + broad); public
notebook mining; piecewise post-fit; GR preprocessing into PF; whole-trace
rescoring (NCC + datum scan); guarded registration broad; differentiable-SMC
smoke + verification reversal.

Phase 4 — the map (July 5–6):
wiggle coherence gate + blend-level phase 2 + distance stratification; neighbor
GR field discrimination gate; error decomposition + calibrated line scan;
neighbor slope transfer; exact (position, momentum) grid inference (+ the
grid-escape artifact); dip-bias diagnosis and correction; pocket bundle
cross-fitted validation and its LB failure-to-confirm; RRC wellbore-database elevation parse + identity-matched
tops ceiling; within-well loop-closure gates (geometry prevalence + GR detection);
steering-geometry gate (trajectory features vs line error + exploitation ceiling).

Phase 5 — closing the last two leads from public writeups (July 7):
PF rate-decay ("mom") sweep testing whether fixing the flattening prior at its
source, rather than patching it post hoc, could be promoted (killed — gains
only appear alongside catastrophic new failures); a trigger-gated 1-D
bimodal-datum midpoint hedge, narrower than the earlier 2-D scan (killed —
passed weakly on tune, reversed on locked).

## Reproducibility

All experiments run on public competition data only, except §5.7's identity-matching
analysis, which additionally uses the public bulk RRC wellbore database (source,
retrieval date, and parsing scripts recorded). Validation splits are by physical well;
pseudo-episodes never cross folds; every cross-fitted constant in §4–§5 is selected on
training folds and applied to held-out wells. Every experiment cited here exists as a
script + JSON report with predeclared kill criteria in our local working repository
(chronological ledger, 51 numbered entries); numbers in this note are quoted from
those reports, and selected scripts and reports can be shared after the competition
if requested. Heavy runs are checkpointed per fold/epoch with atomic report writes;
final outputs are verified for row count, ID order, and finiteness before any
submission.
