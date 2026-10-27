# Known Failures and Traps

## Never reuse

### Memorization / finite lookup

A previous lookup-style submission scored approximately 7133.22 despite
matching known local examples. Any approach that embeds provided outputs,
example identity, or a large finite input-to-output table is rejected.

### Locally attractive but Kaggle-broken bundle

One GPT four-task bundle predicted near 7184.67 locally and scored 7168.11.
Another checkpoint scored 7167.95. This is evidence that local cost and known
examples alone are not sufficient.

## Runtime compatibility

### task133 `TopK(UINT8)`

The circuit was correct on all known examples and worked in a newer ORT, but
ORT 1.22.1 lacked the required kernel. Prefer an architecture that removes
TopK instead of adding expensive casts.

### task347 MaxUnpool

The local profiler rejects the inherited model because of a batch/channel
output-shape mismatch. Kaggle accepted the inherited archive. Keep task347
unchanged unless a replacement is proven in the target runtime.

## Cost failures

### task366

A correct general rewrite increased cost from roughly 36k to 47k. Coordinate
tensors, INT64 indices, and several full-screen `Where` operations destroyed
the expected gain.

### task367

A correct bit-packed rewrite increased cost to roughly 51k. Packing is not
automatically cheaper; count conversion, iteration, and unpacking tensors.

### sparse initializers

Sparse Conv initializers failed full checking or materialized charged dense
intermediates. Measure with the official utility; do not assume sparse syntax
reduces score cost.

## Diagnostic trap: oversized ARC-GEN

The model input is fixed at 30×30. `encode_grid` returns `None` for larger
examples. Passing that value to ORT produces a misleading “Missing Input”
error. The official local validation path skips these examples.

## Process traps

- Filenames are not proof of score.
- A predicted score is not a Kaggle-confirmed score.
- Never merge every lower-cost graph without semantic validation.
- Never overwrite the rollback archive.
- Never mix public-notebook payloads with the private baseline.
- Never expose or package `kaggleAPI.txt`.
- Do not trust node count as the objective; charged tensor memory matters.

