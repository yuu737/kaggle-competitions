# 7250 Portfolio Strategy

The target requires approximately +60, so task selection must optimize
expected portfolio gain per unit of expensive reasoning.

## First principle: build families, not isolated tricks

Search for one architectural transformation that applies to several tasks.
After proving it on one representative, scan all 400 graphs for the same
pattern.

## Track A: summary-first circuits

Replace full-screen masks followed by reductions with direct row, column,
color, count, or object summaries from the uncharged input.

High-cost representatives:

- task118
- task158
- task054
- task233
- task209
- task018
- task319
- task101

Successful historical examples include task191 and parts of task286.

## Track B: morphology and connectivity

Replace repeated full-screen dilation, flood, or MaxPool chains with:

- packed row/column closure;
- interval propagation;
- separable H/V passes;
- bounded region-local computation.

Representatives:

- task366
- task285
- task187
- task367
- task364

Historical warning: packed task367 and a general task366 rewrite were correct
but more expensive. Estimate every intermediate before construction.

## Track C: separable geometry

Replace multiple 2-D branch masks with low-rank row/column factors, matrix
multiplication, or one compact scatter.

Representatives:

- task286
- task255
- task133
- task349
- task243
- task076

task255 is the strongest successful precedent.

## Track D: categorical/label-map circuits

Avoid ten materialized color planes when one compact label map or selector
summary suffices.

Representatives:

- task018
- task319
- task349
- task243
- task118

Do not replace general logic with finite lookup tables.

## Track E: automatic graph rewrites across all tasks

Run general passes before deep task work:

- dead initializer and dead node removal;
- cast-chain composition;
- boolean algebra simplification;
- duplicate subgraph elimination;
- Gather/index composition;
- Slice/Pad fusion;
- dtype narrowing only when official kernels support it;
- direct summary extraction from input;
- repeated constant folding where the result is not a charged large tensor.

The workspace already contains builders for many of these patterns. Re-run
them against the 7190.28 baseline and measure a whole-portfolio result.

## Suggested expensive-model budget

1. 10%: verify environment, rank patterns, and generate a quantitative plan.
2. 20%: run safe automatic rewrites over all 400 tasks.
3. 50%: solve 3–5 reusable architectural families.
4. 15%: integrate, cross-runtime test, and audit.
5. 5%: final forensic review and handoff.

Stop an isolated direction early if the realistic gain is below +0.3 and the
idea does not generalize to other tasks.

## Milestones

```text
M1: +5 verified  — pipeline and first family proven
M2: +15 verified — multiple reusable families
M3: +30 verified — halfway portfolio, broad task coverage
M4: +45 verified — compatibility and integration hardening
M5: +59.72 verified — 7250 working candidate
```

At every milestone, preserve a complete independently auditable submission.

