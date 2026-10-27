"""Banded sum-product decoder for WPS candidate-offset surfaces."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _forward_transition(previous: np.ndarray, penalties: np.ndarray, band: int) -> np.ndarray:
    states = len(previous)
    candidates = []
    for delta in range(-band, band + 1):
        shifted = np.full(states, -np.inf, dtype=np.float64)
        # next[j] receives previous[k] where j-k=delta.
        if delta >= 0:
            shifted[delta:] = previous[:states - delta]
        else:
            shifted[:states + delta] = previous[-delta:]
        candidates.append(shifted - penalties[delta + band])
    return logsumexp(np.stack(candidates, axis=0), axis=0)


def _backward_transition(future: np.ndarray, penalties: np.ndarray, band: int) -> np.ndarray:
    states = len(future)
    candidates = []
    for delta in range(-band, band + 1):
        shifted = np.full(states, -np.inf, dtype=np.float64)
        # current[k] receives future[j] where j-k=delta.
        if delta >= 0:
            shifted[:states - delta] = future[delta:]
        else:
            shifted[-delta:] = future[:states + delta]
        candidates.append(shifted - penalties[delta + band])
    return logsumexp(np.stack(candidates, axis=0), axis=0)


def decode_sum_product(
    composite_cost: np.ndarray,
    candidate_offsets: np.ndarray,
    start_offset: float,
    *,
    emission_temperature: float = 0.05,
    transition_sigma_ft: float = 2.0,
    max_step_ft: float = 8.0,
    anchor_sigma_ft: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column offset posterior and posterior-mean offsets.

    The transition operates on residual offsets around the legal prefix-linear
    center path.  Forward/backward rows are normalized independently, which
    preserves marginals while avoiding numerical overflow on long wells.
    """
    cost = np.asarray(composite_cost, dtype=np.float64)
    offsets = np.asarray(candidate_offsets, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[1] != len(offsets):
        raise ValueError("cost must be [columns, candidate_offsets]")
    if len(offsets) < 2:
        raise ValueError("at least two candidate offsets are required")
    step = float(np.median(np.diff(offsets)))
    band = max(1, int(round(max_step_ft / abs(step))))
    deltas = np.arange(-band, band + 1, dtype=float) * step
    penalties = 0.5 * (deltas / max(transition_sigma_ft, 1e-6)) ** 2
    emission = -(cost - np.min(cost, axis=1, keepdims=True)) / emission_temperature

    columns, states = cost.shape
    forward = np.empty((columns, states), dtype=np.float32)
    first = emission[0] - 0.5 * ((offsets - start_offset) / anchor_sigma_ft) ** 2
    first -= logsumexp(first)
    forward[0] = first
    for index in range(1, columns):
        row = emission[index] + _forward_transition(forward[index - 1], penalties, band)
        row -= logsumexp(row)
        forward[index] = row

    backward = np.zeros((columns, states), dtype=np.float32)
    for index in range(columns - 2, -1, -1):
        future = emission[index + 1] + backward[index + 1]
        row = _backward_transition(future, penalties, band)
        row -= logsumexp(row)
        backward[index] = row

    log_posterior = forward.astype(np.float64) + backward.astype(np.float64)
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior).astype(np.float32)
    mean_offsets = (posterior * offsets[None, :]).sum(axis=1).astype(np.float32)
    return posterior, mean_offsets

