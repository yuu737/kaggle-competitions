"""Learned bidirectional emission model and banded linear-chain CRF."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


GRID_RADIUS = 100.0
GRID_STEP = 2.0
GRID_OFFSETS = np.arange(-GRID_RADIUS, GRID_RADIUS + 0.5 * GRID_STEP, GRID_STEP, dtype=np.float32)
TRANSITION_BAND = 16


@dataclass(frozen=True)
class FeatureBatch:
    time: np.ndarray
    state: np.ndarray
    target_index: np.ndarray | None
    target_in_grid: np.ndarray | None


def _interp_fill(values: np.ndarray, fallback: float) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    return s.interpolate(limit_direction="both").fillna(float(fallback)).to_numpy(np.float32)


def _rolling(values: np.ndarray, window: int, stat: str) -> np.ndarray:
    s = pd.Series(values)
    roll = s.rolling(window, center=True, min_periods=2)
    out = roll.mean() if stat == "mean" else roll.std()
    return out.interpolate(limit_direction="both").fillna(0.0).to_numpy(np.float32)


def build_features(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    pf_tvt: np.ndarray,
    pf_std: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    gr_scale: float,
    target_tvt: np.ndarray | None = None,
    offsets: np.ndarray = GRID_OFFSETS,
) -> FeatureBatch:
    """Build leak-free time/state features around a PF center path."""
    md = np.asarray(md, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    pf_tvt = np.asarray(pf_tvt, dtype=np.float32)
    pf_std = np.asarray(pf_std, dtype=np.float32)
    tw_tvt = np.asarray(tw_tvt, dtype=np.float32)
    tw_gr = np.asarray(tw_gr, dtype=np.float32)
    if not (len(md) == len(z) == len(gr) == len(pf_tvt) == len(pf_std)):
        raise ValueError("point arrays must have equal length")
    if len(md) == 0:
        raise ValueError("empty sequence")

    order = np.argsort(tw_tvt)
    tw_tvt, tw_gr = tw_tvt[order], tw_gr[order]
    tw_gr = _interp_fill(tw_gr, float(np.nanmean(tw_gr)))
    gr = _interp_fill(gr, float(np.nanmean(tw_gr)))
    scale = max(float(gr_scale), 1e-3)

    dmd = np.maximum(np.diff(md, prepend=md[0] - 1.0), 1.0)
    dgr = np.gradient(gr, md).astype(np.float32) if len(md) > 2 else np.zeros_like(gr)
    dz = np.diff(z, prepend=z[0]) / dmd
    dpf = np.diff(pf_tvt, prepend=pf_tvt[0]) / dmd
    gr_mean = _rolling(gr, 21, "mean")
    gr_std = _rolling(gr, 21, "std")
    md_since = (md - md[0]) / max(float(md[-1] - md[0]), 1.0)
    time = np.column_stack(
        [
            (gr - np.mean(gr)) / max(float(np.std(gr)), 1.0),
            dgr / max(float(np.nanstd(dgr)), 1e-3),
            (gr_mean - np.mean(gr)) / max(float(np.std(gr)), 1.0),
            gr_std / max(float(np.mean(gr_std)), 1.0),
            np.clip(dz, -1.0, 1.0),
            np.clip(dpf, -1.0, 1.0),
            np.clip(pf_std / 30.0, 0.0, 5.0),
            md_since,
            np.sin(np.pi * md_since),
            np.cos(np.pi * md_since),
        ]
    ).astype(np.float32)

    candidate_tvt = pf_tvt[:, None] + offsets[None, :]
    expected_gr = np.interp(candidate_tvt.reshape(-1), tw_tvt, tw_gr).reshape(candidate_tvt.shape)
    tw_grad = np.gradient(tw_gr, tw_tvt)
    expected_grad = np.interp(candidate_tvt.reshape(-1), tw_tvt, tw_grad).reshape(candidate_tvt.shape)
    resid = np.clip((gr[:, None] - expected_gr) / scale, -12.0, 12.0)
    state = np.stack(
        [
            resid,
            np.abs(resid),
            np.minimum(resid * resid, 25.0),
            np.clip(expected_grad / max(float(np.nanstd(tw_grad)), 1e-3), -8.0, 8.0),
            np.broadcast_to(offsets[None, :] / GRID_RADIUS, candidate_tvt.shape),
        ],
        axis=-1,
    ).astype(np.float32)

    target_index = None
    target_in_grid = None
    if target_tvt is not None:
        delta = np.asarray(target_tvt, dtype=np.float32) - pf_tvt
        target_in_grid = np.abs(delta) <= GRID_RADIUS + 0.5 * GRID_STEP
        target_index = np.rint((delta + GRID_RADIUS) / GRID_STEP).astype(np.int64)
        target_index = np.clip(target_index, 0, len(offsets) - 1)
    return FeatureBatch(time=time, state=state, target_index=target_index, target_in_grid=target_in_grid)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        y = self.dropout(F.gelu(self.norm2(self.conv2(y))))
        return x + y


class BidirectionalTCNEmission(nn.Module):
    def __init__(self, time_features: int = 10, state_features: int = 5, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.time_in = nn.Conv1d(time_features, hidden, 1)
        self.tcn = nn.Sequential(
            *[ResidualTCNBlock(hidden, dilation, dropout) for dilation in (1, 2, 4, 8, 16, 32)]
        )
        self.state_in = nn.Sequential(nn.Linear(state_features, 32), nn.GELU())
        self.emission = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, time_features: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        # Symmetric TCN padding gives each point both past and future context.
        context = self.tcn(self.time_in(time_features.transpose(1, 2))).transpose(1, 2)
        context = context[:, :, None, :].expand(-1, -1, state_features.shape[2], -1)
        state = self.state_in(state_features)
        return self.emission(torch.cat([context, state], dim=-1)).squeeze(-1)


def _shift_prev_to_dest(prev: torch.Tensor, delta: int, fill: float) -> torch.Tensor:
    out = prev.new_full(prev.shape, fill)
    if delta > 0:
        out[:, delta:] = prev[:, :-delta]
    elif delta < 0:
        out[:, :delta] = prev[:, -delta:]
    else:
        out = prev
    return out


def _shift_dest_to_source(nxt: torch.Tensor, delta: int, fill: float) -> torch.Tensor:
    out = nxt.new_full(nxt.shape, fill)
    if delta > 0:
        out[:, :-delta] = nxt[:, delta:]
    elif delta < 0:
        out[:, -delta:] = nxt[:, :delta]
    else:
        out = nxt
    return out


class BandedLinearCRF(nn.Module):
    def __init__(
        self,
        n_states: int = len(GRID_OFFSETS),
        grid_step: float = GRID_STEP,
        band: int = TRANSITION_BAND,
    ):
        super().__init__()
        self.n_states = int(n_states)
        self.grid_step = float(grid_step)
        self.band = int(band)
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_huber_delta = nn.Parameter(torch.tensor(1.0))
        offsets = (torch.arange(n_states, dtype=torch.float32) - (n_states - 1) / 2) * grid_step
        self.register_buffer("offsets", offsets)
        self.register_buffer("deltas", torch.arange(-band, band + 1, dtype=torch.long))
        destination = torch.arange(n_states)[:, None]
        previous = destination - self.deltas[None, :]
        following = destination + self.deltas[None, :]
        self.register_buffer("previous_index", previous.clamp(0, n_states - 1))
        self.register_buffer("previous_valid", (previous >= 0) & (previous < n_states))
        self.register_buffer("following_index", following.clamp(0, n_states - 1))
        self.register_buffer("following_valid", (following >= 0) & (following < n_states))

    def transition_scores(
        self,
        center_u: torch.Tensor,
        md: torch.Tensor,
        initial_rate: torch.Tensor,
    ) -> torch.Tensor:
        dmd = torch.clamp(md[:, 1:] - md[:, :-1], min=1.0)
        center_du = center_u[:, 1:] - center_u[:, :-1]
        expected = initial_rate[:, None] * dmd
        delta_u = self.deltas.to(center_u.dtype)[None, None, :] * self.grid_step
        residual = center_du[:, :, None] + delta_u - expected[:, :, None]
        huber_delta = F.softplus(self.log_huber_delta) + 1e-3
        abs_r = residual.abs()
        huber = torch.where(
            abs_r <= huber_delta,
            0.5 * residual.square() / huber_delta,
            abs_r - 0.5 * huber_delta,
        )
        return -F.softplus(self.log_strength) * huber

    def start_scores(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return (-0.5 * (self.offsets.to(device=device, dtype=dtype) / 8.0).square())

    def forward_messages(
        self, emissions: torch.Tensor, transitions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = emissions.shape
        neg = torch.finfo(emissions.dtype).min / 4
        messages = [emissions[:, 0] + self.start_scores(emissions.dtype, emissions.device)]
        for t in range(1, length):
            prev = messages[-1]
            candidates = prev[:, self.previous_index]
            candidates = candidates.masked_fill(~self.previous_valid[None], neg)
            candidates = candidates + transitions[:, t - 1, None, :]
            messages.append(emissions[:, t] + torch.logsumexp(candidates, dim=-1))
        alpha = torch.stack(messages, dim=1)
        log_z = torch.logsumexp(alpha[:, -1], dim=-1)
        return alpha, log_z

    def backward_messages(self, emissions: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        length = emissions.shape[1]
        neg = torch.finfo(emissions.dtype).min / 4
        messages = [emissions.new_zeros(emissions.shape[0], emissions.shape[2])]
        for t in range(length - 2, -1, -1):
            nxt = emissions[:, t + 1] + messages[-1]
            candidates = nxt[:, self.following_index]
            candidates = candidates.masked_fill(~self.following_valid[None], neg)
            candidates = candidates + transitions[:, t, None, :]
            messages.append(torch.logsumexp(candidates, dim=-1))
        return torch.stack(messages[::-1], dim=1)

    def marginals(
        self, emissions: torch.Tensor, transitions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, log_z = self.forward_messages(emissions, transitions)
        beta = self.backward_messages(emissions, transitions)
        log_prob = alpha + beta - log_z[:, None, None]
        return torch.softmax(log_prob, dim=-1), log_z

    def path_nll(
        self,
        emissions: torch.Tensor,
        transitions: torch.Tensor,
        target: torch.Tensor,
        log_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if log_z is None:
            _, log_z = self.forward_messages(emissions, transitions)
        emit_score = emissions.gather(2, target[:, :, None]).squeeze(-1).sum(dim=1)
        delta = target[:, 1:] - target[:, :-1]
        valid = (delta.abs() <= self.band).all(dim=1)
        delta_index = torch.clamp(delta + self.band, 0, 2 * self.band)
        trans_score = transitions.gather(2, delta_index[:, :, None]).squeeze(-1).sum(dim=1)
        start = self.start_scores(emissions.dtype, emissions.device)[target[:, 0]]
        return log_z - (start + emit_score + trans_score), valid

    def viterbi(self, emissions: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        batch, length, _ = emissions.shape
        neg = torch.finfo(emissions.dtype).min / 4
        score = emissions[:, 0] + self.start_scores(emissions.dtype, emissions.device)
        back = []
        for t in range(1, length):
            candidates = score[:, self.previous_index]
            candidates = candidates.masked_fill(~self.previous_valid[None], neg)
            candidates = candidates + transitions[:, t - 1, None, :]
            best, which = candidates.max(dim=-1)
            score = emissions[:, t] + best
            back.append(which)
        state = score.argmax(dim=-1)
        path = [state]
        for which in reversed(back):
            chosen = which.gather(1, state[:, None]).squeeze(1)
            state = state - self.deltas[chosen]
            path.append(state)
        return torch.stack(path[::-1], dim=1)


class LearnedCRFGeosteer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emission_model = BidirectionalTCNEmission()
        self.crf = BandedLinearCRF()

    def forward(
        self,
        time_features: torch.Tensor,
        state_features: torch.Tensor,
        center_u: torch.Tensor,
        md: torch.Tensor,
        initial_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emissions = self.emission_model(time_features, state_features)
        transitions = self.crf.transition_scores(center_u, md, initial_rate)
        return emissions, transitions


def training_loss(
    model: LearnedCRFGeosteer,
    time_features: torch.Tensor,
    state_features: torch.Tensor,
    center_u: torch.Tensor,
    md: torch.Tensor,
    initial_rate: torch.Tensor,
    target_index: torch.Tensor,
    target_in_grid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    emissions, transitions = model(time_features, state_features, center_u, md, initial_rate)
    marginals, log_z = model.crf.marginals(emissions, transitions)
    nll, valid_path = model.crf.path_nll(emissions, transitions, target_index, log_z=log_z)
    log_marginal = torch.log(torch.clamp(marginals, min=1e-8))
    point_nll = -log_marginal.gather(2, target_index[:, :, None]).squeeze(-1)
    point_nll = (point_nll * target_in_grid).sum(dim=1) / torch.clamp(target_in_grid.sum(dim=1), min=1)
    sequence_loss = torch.where(valid_path & target_in_grid.all(dim=1), nll / emissions.shape[1], point_nll).mean()

    posterior_offset = (marginals * model.crf.offsets[None, None, :]).sum(dim=-1)
    true_offset = model.crf.offsets[target_index]
    huber = F.smooth_l1_loss(posterior_offset[target_in_grid], true_offset[target_in_grid])
    curvature = torch.diff(posterior_offset, n=2, dim=1).abs().mean()
    loss = sequence_loss + 0.2 * huber + 0.02 * curvature
    return loss, {
        "crf_nll": float(sequence_loss.detach()),
        "posterior_huber": float(huber.detach()),
        "curvature": float(curvature.detach()),
        "valid_path_rate": float(valid_path.float().mean().detach()),
    }


@torch.no_grad()
def decode(
    model: LearnedCRFGeosteer,
    time_features: torch.Tensor,
    state_features: torch.Tensor,
    center_tvt: torch.Tensor,
    z: torch.Tensor,
    md: torch.Tensor,
    initial_rate: torch.Tensor,
) -> dict[str, torch.Tensor]:
    center_u = center_tvt + z
    emissions, transitions = model(time_features, state_features, center_u, md, initial_rate)
    marginals, _ = model.crf.marginals(emissions, transitions)
    path = model.crf.viterbi(emissions, transitions)
    map_offset = model.crf.offsets[path]
    cdf = marginals.cumsum(dim=-1)
    median_index = (cdf < 0.5).sum(dim=-1).clamp(max=model.crf.n_states - 1)
    median_offset = model.crf.offsets[median_index]
    entropy = -(marginals * torch.log(torch.clamp(marginals, min=1e-8))).sum(dim=-1)
    entropy = entropy / np.log(model.crf.n_states)
    boundary = (path <= 1) | (path >= model.crf.n_states - 2)
    confidence = torch.clamp((0.75 - entropy) / 0.35, 0.0, 1.0)
    confidence = torch.where(boundary, torch.zeros_like(confidence), confidence)
    confidence = F.avg_pool1d(confidence[:, None], kernel_size=21, stride=1, padding=10).squeeze(1)
    gated = center_tvt + confidence * map_offset
    finite = torch.isfinite(gated)
    gated = torch.where(finite, gated, center_tvt)
    return {
        "map_tvt": center_tvt + map_offset,
        "median_tvt": center_tvt + median_offset,
        "gated_tvt": gated,
        "entropy": entropy,
        "confidence": confidence,
        "boundary": boundary,
    }
