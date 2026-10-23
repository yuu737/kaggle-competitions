"""Cost-volume construction for the Whole-Well Posterior Surface (WPS) model.

The candidate state is structural elevation ``U = TVT + Z``.  For every
measured-depth block and candidate U, the module samples the typewell at the
implied ``TVT = U - Z`` and keeps the complete candidate-depth evidence.  It
does not choose an offset or collapse the surface to a handful of matcher
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class CostVolumeConfig:
    block_rows: int = 4
    u_half_range_ft: float = 256.0
    u_step_ft: float = 2.0
    center_prefix_rows: int = 768
    window_ft: tuple[int, ...] = (16, 32, 64, 128)
    corr_window_ft: tuple[int, ...] = (32, 64)
    min_observation_scale: float = 5.0
    max_observation_scale: float = 60.0


@dataclass
class CostVolume:
    well: str
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_true: np.ndarray
    u_true: np.ndarray
    known: np.ndarray
    center_u: np.ndarray
    candidate_offsets: np.ndarray
    costs: np.ndarray
    composite: np.ndarray
    valid: np.ndarray
    channel_names: tuple[str, ...]
    channel_weights: np.ndarray
    calibration_alpha: float
    calibration_beta: float
    observation_scale: float
    split_row: int

    @property
    def candidate_u(self) -> np.ndarray:
        return self.center_u[:, None] + self.candidate_offsets[None, :]


def _finite_interp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    if finite.all():
        return values
    x = np.arange(len(values))
    return np.interp(x, x[finite], values[finite])


def _block_mean(values: np.ndarray, group: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.empty(0, dtype=float)
    pad = (-len(values)) % group
    if pad:
        values = np.pad(values, (0, pad), mode="edge")
    return values.reshape(-1, group).mean(axis=1)


def _block_any(values: np.ndarray, group: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if len(values) == 0:
        return np.empty(0, dtype=bool)
    pad = (-len(values)) % group
    if pad:
        values = np.pad(values, (0, pad), mode="edge")
    return values.reshape(-1, group).any(axis=1)


def _robust_scale(values: np.ndarray, floor: float, ceiling: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return float(floor)
    med = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - med))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(values)
    return float(np.clip(scale, floor, ceiling))


def fit_prefix_affine(type_at_prefix: np.ndarray, horizontal_gr: np.ndarray) -> tuple[float, float]:
    """Robustly fit horizontal_GR ~= alpha * typewell_GR + beta."""
    x = np.asarray(type_at_prefix, dtype=float)
    y = np.asarray(horizontal_gr, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 20 or np.std(x) < 1e-6:
        return 1.0, float(np.nanmedian(y) - np.nanmedian(x))
    active = np.ones(len(x), dtype=bool)
    alpha, beta = 1.0, float(np.median(y) - np.median(x))
    for _ in range(4):
        if active.sum() < 20:
            break
        alpha, beta = np.polyfit(x[active], y[active], 1)
        residual = y - (alpha * x + beta)
        center = np.median(residual[active])
        scale = 1.4826 * np.median(np.abs(residual[active] - center)) + 1e-6
        active = np.abs(residual - center) <= 3.5 * scale
    alpha = float(np.clip(alpha, 0.25, 4.0))
    beta = float(np.median(y - alpha * x))
    return alpha, beta


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    return uniform_filter1d(values, size=max(1, int(width)), axis=0, mode="nearest")


def _rolling_corr_cost(horizontal: np.ndarray, predicted: np.ndarray, width: int) -> np.ndarray:
    h = np.broadcast_to(horizontal[:, None], predicted.shape)
    mh = _rolling_mean(h, width)
    mp = _rolling_mean(predicted, width)
    cov = _rolling_mean(h * predicted, width) - mh * mp
    vh = np.maximum(_rolling_mean(h * h, width) - mh * mh, 1e-6)
    vp = np.maximum(_rolling_mean(predicted * predicted, width) - mp * mp, 1e-6)
    rho = cov / np.sqrt(vh * vp)
    return np.clip(1.0 - rho, 0.0, 2.0).astype(np.float32)


def _rank_composite(
    costs: np.ndarray,
    valid: np.ndarray,
    known: np.ndarray,
    u_true: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse channel ranks using only legal known-prefix reconstruction quality."""
    n, states, channels = costs.shape
    rank_cube = np.empty((n, states, channels), dtype=np.float32)
    denominator = max(states - 1, 1)
    for channel in range(channels):
        values = costs[:, :, channel]
        order = np.argsort(values, axis=1, kind="stable")
        rank = np.empty_like(order)
        rows = np.arange(n)[:, None]
        rank[rows, order] = np.arange(states)[None, :]
        rank_cube[:, :, channel] = rank.astype(np.float32) / denominator
    nearest = np.abs(candidates - u_true[:, None]).argmin(axis=1)
    prefix_quality = np.empty(channels, dtype=np.float32)
    for channel in range(channels):
        prefix_quality[channel] = float(
            np.mean(rank_cube[np.arange(n)[known], nearest[known], channel])
        )
    # Prefix rank differences are strong enough to suppress channels that look
    # plausible in isolation but fail on the observed section (notably the
    # first M0 derivative/correlation channels).  No hidden label enters here.
    logits = -(prefix_quality - prefix_quality.min()) / 0.08
    weights = np.exp(np.clip(logits, -20.0, 0.0)).astype(np.float32)
    weights /= weights.sum()
    out = np.sum(rank_cube * weights[None, None, :], axis=2)
    out[~valid] = 1.25
    return out, weights


def _prepare_blocks(horizontal: pd.DataFrame, group: int) -> dict[str, np.ndarray | int]:
    known_rows = horizontal["TVT_input"].notna().to_numpy()
    if not known_rows.any():
        raise ValueError("horizontal well has no known TVT_input prefix")
    split = int(np.flatnonzero(known_rows)[-1])
    gr_rows = _finite_interp(horizontal["GR"].to_numpy(float))
    tvt_rows = horizontal["TVT"].to_numpy(float)
    md_rows = horizontal["MD"].to_numpy(float)
    z_rows = horizontal["Z"].to_numpy(float)

    # Block prefix and suffix separately so no block straddles the legal mask.
    slices = (slice(0, split + 1), slice(split + 1, None))
    packed: dict[str, list[np.ndarray]] = {k: [] for k in ("md", "z", "gr", "tvt", "known")}
    for part_index, slc in enumerate(slices):
        packed["md"].append(_block_mean(md_rows[slc], group))
        packed["z"].append(_block_mean(z_rows[slc], group))
        packed["gr"].append(_block_mean(gr_rows[slc], group))
        packed["tvt"].append(_block_mean(tvt_rows[slc], group))
        flags = np.ones(np.asarray(md_rows[slc]).shape, dtype=bool) if part_index == 0 else np.zeros(np.asarray(md_rows[slc]).shape, dtype=bool)
        packed["known"].append(_block_any(flags, group))
    result = {key: np.concatenate(value) for key, value in packed.items()}
    result["prefix_blocks"] = len(packed["md"][0])
    result["split_row"] = split
    return result


def build_cost_volume(
    well: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: CostVolumeConfig = CostVolumeConfig(),
) -> CostVolume:
    """Build the complete MD x candidate-U evidence tensor for one well."""
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    blocks = _prepare_blocks(horizontal, config.block_rows)
    md = np.asarray(blocks["md"], dtype=float)
    z = np.asarray(blocks["z"], dtype=float)
    gr = np.asarray(blocks["gr"], dtype=float)
    tvt_true = np.asarray(blocks["tvt"], dtype=float)
    known = np.asarray(blocks["known"], dtype=bool)

    tw = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    type_tvt = tw["TVT"].to_numpy(float)
    type_gr = tw["GR"].to_numpy(float)
    unique = np.r_[True, np.diff(type_tvt) > 0]
    type_tvt, type_gr = type_tvt[unique], type_gr[unique]
    if len(type_tvt) < 20:
        raise ValueError(f"{well}: insufficient typewell samples")

    prefix_rows = horizontal.loc[horizontal["TVT_input"].notna()]
    prefix_tvt = prefix_rows["TVT_input"].to_numpy(float)
    prefix_gr = _finite_interp(prefix_rows["GR"].to_numpy(float))
    raw_at_prefix = np.interp(prefix_tvt, type_tvt, type_gr)
    alpha, beta = fit_prefix_affine(raw_at_prefix, prefix_gr)
    type_gr_cal = alpha * type_gr + beta
    residual = prefix_gr - (alpha * raw_at_prefix + beta)
    obs_scale = _robust_scale(
        residual, config.min_observation_scale, config.max_observation_scale
    )

    last_known = prefix_rows.iloc[-1]
    # A flat U center only covers ~42% of hidden rows at +/-64 ft.  A legal
    # linear extrapolation from the last 768 known rows covers >99.9% at
    # +/-256 ft while keeping the candidate tensor compact.
    fit_rows = prefix_rows.tail(config.center_prefix_rows)
    fit_md = fit_rows["MD"].to_numpy(float) - float(last_known["MD"])
    fit_u = fit_rows["TVT_input"].to_numpy(float) + fit_rows["Z"].to_numpy(float)
    if len(fit_rows) >= 20 and np.ptp(fit_md) > 1e-6:
        slope, intercept = np.polyfit(fit_md, fit_u, 1)
        center_u = intercept + slope * (md - float(last_known["MD"]))
    else:
        center_u = np.full(len(md), float(last_known["TVT_input"] + last_known["Z"]))
    offsets = np.arange(
        -config.u_half_range_ft,
        config.u_half_range_ft + 0.5 * config.u_step_ft,
        config.u_step_ft,
        dtype=np.float32,
    )
    center_u = np.asarray(center_u, dtype=np.float32)
    implied_tvt = center_u[:, None] + offsets[None, :] - z[:, None]
    valid = (implied_tvt >= type_tvt[0]) & (implied_tvt <= type_tvt[-1])
    predicted = np.interp(implied_tvt.ravel(), type_tvt, type_gr_cal).reshape(implied_tvt.shape)

    point = np.log1p(((gr[:, None] - predicted) / obs_scale) ** 2).astype(np.float32)
    channels: list[np.ndarray] = [point]
    names = ["point_cauchy"]

    md_step = float(np.median(np.diff(md))) if len(md) > 1 else float(config.block_rows)
    for window_ft in config.window_ft:
        width = max(1, int(round(window_ft / max(md_step, 1e-6))))
        channels.append(_rolling_mean(point, width).astype(np.float32))
        names.append(f"cauchy_mean_{window_ft}ft")

    dgr = np.gradient(gr, md)
    dpred = np.gradient(predicted, md, axis=0)
    prefix_n = int(blocks["prefix_blocks"])
    dscale = _robust_scale(
        (dgr[:, None] - dpred)[max(0, prefix_n // 3):prefix_n].ravel(),
        max(0.05, config.min_observation_scale / 32.0),
        config.max_observation_scale,
    )
    derivative = np.log1p(((dgr[:, None] - dpred) / dscale) ** 2).astype(np.float32)
    channels.append(derivative)
    names.append("derivative_cauchy")

    for window_ft in config.corr_window_ft:
        width = max(3, int(round(window_ft / max(md_step, 1e-6))))
        channels.append(_rolling_corr_cost(gr, predicted, width))
        names.append(f"corr_cost_{window_ft}ft")

    costs = np.stack(channels, axis=-1).astype(np.float32)
    for channel in range(costs.shape[-1]):
        finite = costs[:, :, channel][valid & np.isfinite(costs[:, :, channel])]
        penalty = float(np.quantile(finite, 0.99) + 2.0) if len(finite) else 1e3
        bad = ~valid | ~np.isfinite(costs[:, :, channel])
        costs[:, :, channel][bad] = penalty
    candidates = center_u[:, None] + offsets[None, :]
    composite, channel_weights = _rank_composite(
        costs, valid, known, tvt_true + z, candidates
    )

    return CostVolume(
        well=str(well), md=md.astype(np.float32), z=z.astype(np.float32),
        gr=gr.astype(np.float32), tvt_true=tvt_true.astype(np.float32),
        u_true=(tvt_true + z).astype(np.float32), known=known,
        center_u=center_u, candidate_offsets=offsets,
        costs=costs, composite=composite, valid=valid,
        channel_names=tuple(names), channel_weights=channel_weights,
        calibration_alpha=alpha,
        calibration_beta=beta, observation_scale=obs_scale,
        split_row=int(blocks["split_row"]),
    )


def posterior_mean_u(volume: CostVolume, temperature: float = 0.08) -> np.ndarray:
    logits = -(volume.composite - volume.composite.min(axis=1, keepdims=True)) / temperature
    logits[~volume.valid] = -60.0
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return (weights * volume.candidate_u).sum(axis=1)


def true_rank_percentile(volume: CostVolume) -> np.ndarray:
    nearest = np.abs(volume.candidate_u - volume.u_true[:, None]).argmin(axis=1)
    values = volume.composite[np.arange(len(nearest)), nearest]
    return np.mean(volume.composite < values[:, None], axis=1).astype(np.float32)
