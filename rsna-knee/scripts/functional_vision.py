"""Sensory calibration of the complete MaleCNS graph on procedurally held-out movies.

This is a connectome-constrained rate model, not a simulation of a living fly.
Photoreceptor positions are inferred from annotated postsynaptic optic columns.
All original nodes and directed edges remain; no image-to-readout shortcut.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from trainable_wholebrain import FullGraph

ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = ROOT / 'data/connectome/male-cns-v1.0'
OUT = ROOT / 'outputs/functional_vision'


class SensoryGraph(FullGraph):
    def __init__(self, root=GRAPH_ROOT, device='cuda', seed=20261016, spatial_bins=4):
        super().__init__(root, device=device, seed=seed)
        ids = self.mapping['body_ids']
        root = Path(root)
        a = pd.read_feather(root / 'body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId').reindex(ids)
        nt = pd.read_feather(root / 'neurotransmitters.feather').set_index('body').reindex(ids).consensus_nt.fillna('unknown')
        # Histaminergic photoreceptor transmission through chloride receptors is
        # inhibitory. Keep the older model immutable; correct this model only.
        for name, source_axis in [('adjacency', True), ('transpose', False)]:
            matrix = getattr(self, name)
            if source_axis:
                hist = torch.as_tensor(nt.eq('histamine').to_numpy().copy(), device=device)[matrix.col_indices().long()]
            else:
                h = torch.as_tensor(nt.eq('histamine').to_numpy().copy(), device=device)
                hist = torch.repeat_interleave(h, matrix.crow_indices()[1:] - matrix.crow_indices()[:-1])
            matrix.values()[hist] *= -1
        counts = sp.load_npz(root / 'counts.npz').tocsr()
        known = np.flatnonzero(a[['assignedOlHex1', 'assignedOlHex2']].notna().all(axis=1))
        retinal = np.flatnonzero(a.superclass.eq('ol_sensory') & a.type.fillna('').str.match(r'^R[1-8]'))
        coordinates = a.iloc[known][['assignedOlHex1', 'assignedOlHex2']].to_numpy(dtype=np.float64)
        links = counts[known][:, retinal]
        mass = np.asarray(links.sum(0)).ravel()
        mean = (links.T @ coordinates) / np.maximum(mass[:, None], 1)
        variance = (links.T @ (coordinates ** 2)) / np.maximum(mass[:, None], 1) - mean ** 2
        spread = np.maximum(variance.sum(1), 0) ** .5
        valid = (mass > 0) & (spread <= 1.0)
        retina = retinal[valid]
        def hex_xy(hexes):
            q, p = (hexes - [18, 19]).T
            return np.stack([np.sqrt(3) / 2 * (q - p), -.5 * (q + p)], 1)
        limits = np.stack([hex_xy(coordinates).min(0), hex_xy(coordinates).max(0)])
        def grid(hexes):
            return (2 * (hex_xy(hexes) - limits[0]) / (limits[1] - limits[0]) - 1).astype(np.float32)
        retinal_grid, visual_grid = grid(mean[valid]), grid(coordinates)
        olfactory = np.flatnonzero(a['class'].eq('olfactory') & a.type.fillna('').str.startswith('ORN_'))
        odor_types, odor_group = np.unique(a.iloc[olfactory].type.to_numpy().astype(str), return_inverse=True)
        spatial_types, spatial_group = np.unique(a.iloc[known].type.to_numpy().astype(str), return_inverse=True)
        bins = np.minimum(((visual_grid + 1) * (spatial_bins/2)).astype(int), spatial_bins-1).clip(0, spatial_bins-1)
        spatial_pool = spatial_group * spatial_bins**2 + bins[:, 1] * spatial_bins + bins[:, 0]
        spatial_counts = np.bincount(spatial_pool, minlength=len(spatial_types) * spatial_bins**2).clip(1)
        # Conjugated topology null preserves cell type, NT, side, and input route.
        membership = np.zeros(self.n, dtype=int); membership[retina] = 1; membership[olfactory] = 2
        side = a.rootSide.fillna(a.somaSide).fillna('unknown').astype(str).to_numpy()
        keys = np.array([f'{g}|{s}|{m}' for g, s, m in zip(self.mapping['group_index'], side, membership)])
        _, partition = np.unique(keys, return_inverse=True)
        order = np.argsort(partition, kind='stable')
        permutation = np.arange(self.n); rng = np.random.default_rng(seed)
        for block in np.split(order, np.flatnonzero(np.diff(partition[order])) + 1):
            permutation[block] = rng.permutation(block)
        self.spatial_features = len(spatial_counts)
        self.spatial_bins = spatial_bins
        self.spatial_channels = len(spatial_types)
        self.odor_channels = len(odor_types)
        arrays = dict(retina=retina, retinal_grid=retinal_grid, olfactory=olfactory, odor_group=odor_group,
                      spatial_nodes=known, spatial_pool=spatial_pool, spatial_counts=spatial_counts.astype(np.float32),
                      permutation=permutation, inverse=np.argsort(permutation))
        for k, v in arrays.items():
            self.register_buffer(k, torch.as_tensor(v, device=device), persistent=False)
        self.sensory_mapping = {**self.mapping, **arrays, 'retinal_hex': mean[valid],
            'retinal_spread': spread[valid], 'retinal_synapse_mass': mass[valid],
            'spatial_types': spatial_types, 'odor_types': odor_types, 'hex_xy_limits': limits}
        self.metadata.update(retinal_total=len(retinal), retinal_mapped=len(retina),
            retinal_mapping='Synapse-count weighted coordinates of postsynaptic annotated optic cells; mass>0 and hex RMS spread<=1; inferred, not directly measured',
            retinal_coverage_by_type=a.iloc[retinal].assign(mapped=valid).groupby('type').mapped.agg(['count', 'sum']).to_dict(),
            input='Same grayscale movie in both eyes; only mapped R1-R6/R7/R8 driven',
            olfactory_nodes=len(olfactory), olfactory_types=len(odor_types),
            nt_sign='GABA/glutamate/histamine negative; other/unclear positive. Receptor-dependent and modulatory effects remain simplified.',
            histamine_nodes=int(nt.eq('histamine').sum()),
            spatial_readout=f'{spatial_bins}x{spatial_bins} position bins within each of 15 officially column-annotated optic types, in addition to ALL-neuron type means',
            coordinate_limits=limits.tolist(), all_original_edges_retained=True)

    def propagate(self, h, mode='real'):
        if mode == 'type_shuffle':
            return super().propagate(h[self.inverse], 'real')[self.permutation]
        return super().propagate(h, mode)

    def save(self, out=OUT):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'sensory_mapping.npz', **self.sensory_mapping)
        (out / 'graph.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')


class SpatialStateReadout(nn.Module):
    """Small classifier of neural states, never of input pixels or movies."""
    def __init__(self, graph):
        super().__init__(); self.groups=graph.groups;self.bins=graph.spatial_bins;self.channels=graph.spatial_channels
        self.global_branch=nn.Sequential(nn.Linear(graph.groups,64),nn.GELU())
        self.spatial_branch=nn.Sequential(nn.Conv2d(self.channels,32,3,padding=1),nn.GELU(),
            nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear(512,64),nn.GELU())
        self.classifier=nn.Linear(128,13)
        nn.init.normal_(self.global_branch[0].weight,std=.003);nn.init.zeros_(self.global_branch[0].bias)

    def forward(self,x):
        global_features=self.global_branch(x[:,:self.groups])
        spatial=self.spatial_branch(x[:,self.groups:].reshape(-1,self.channels,self.bins,self.bins))
        return self.classifier(torch.cat([global_features,spatial],1))


class SensoryWholeBrain(nn.Module):
    def __init__(self, graph, route='visual', frames=12, warmup=8, dt=.02, readout_hidden=0):
        super().__init__()
        self.graph = graph; self.route = route; self.frames = frames; self.warmup = warmup; self.dt = dt
        # Positive learned time constants and synaptic gains shared by cell type.
        self.tau_raw = nn.Parameter(torch.full((graph.groups,), math.log(.03 / .27)))
        self.bias_raw = nn.Parameter(torch.zeros(graph.groups))
        self.pre_gain = nn.Parameter(torch.zeros(graph.groups))
        self.post_gain = nn.Parameter(torch.zeros(graph.groups))
        self.light_gain = nn.Parameter(torch.tensor(0.))
        # Artificial sensory substitution: 8x8 image -> actual ORN type channels.
        # Framewise, no labels or hand-coded motion vector enter either route.
        self.odor_adapter = nn.Linear(64, graph.odor_channels)
        nn.init.normal_(self.odor_adapter.weight, std=.1); nn.init.zeros_(self.odor_adapter.bias)
        self.features = graph.groups + graph.spatial_features
        self.register_buffer('feature_center', torch.zeros(self.features))
        self.register_buffer('feature_scale', torch.ones(self.features))
        # The same readout is used across all route/topology comparisons.
        if readout_hidden == -1:
            self.readout = SpatialStateReadout(graph)
        elif readout_hidden:
            self.readout = nn.Sequential(nn.Linear(self.features, readout_hidden), nn.GELU(), nn.Linear(readout_hidden, 13))
            nn.init.normal_(self.readout[0].weight, std=.003); nn.init.zeros_(self.readout[0].bias)
            nn.init.normal_(self.readout[2].weight, std=.03); nn.init.zeros_(self.readout[2].bias)
        else:
            self.readout = nn.Linear(self.features, 13)  # direction4, shape3, quadrant4, polarity2
            nn.init.normal_(self.readout.weight, std=.001); nn.init.zeros_(self.readout.bias)
        self.spec = dict(architecture='sensory_full_malecns_v1', route=route, frames=frames, warmup=warmup, dt=dt, readout_hidden=readout_hidden, spatial_bins=graph.spatial_bins,
            dynamics='h+=(dt/tau)*(-h+bias+post*A*(pre*relu(h))+u_t); tau=.02+.3*sigmoid(raw); bias=.5+.5*tanh(raw)',
            gains='.25+1.5*sigmoid(raw), positive; all fixed source signs preserved after histamine correction',
            readout='Final state only, all-neuron type means plus column-position means; train-only fixed feature scaling; no image skip',
            physiological_claim=False)

    def inputs(self, movies, route=None):
        route = route or self.route
        b, t, height, width = movies.shape; g = self.graph
        flat = movies.reshape(b * t, 1, height, width)
        retina = F.grid_sample(flat, g.retinal_grid[None, :, None].expand(b * t, -1, -1, -1),
            mode='bilinear', align_corners=True)[:, 0, :, 0].reshape(b, t, -1).permute(1, 2, 0)
        retina = retina * (.25 + 1.5 * self.light_gain.sigmoid())
        odor = self.odor_adapter(F.adaptive_avg_pool2d(flat, 8).flatten(1)).sigmoid()
        odor = odor[:, g.odor_group].reshape(b, t, -1).permute(1, 2, 0)
        if route == 'visual': odor = odor * 0
        elif route == 'olfactory': retina = retina * 0
        elif route != 'both': raise ValueError(route)
        return retina, odor

    def state_features(self, h):
        g = self.graph
        type_features = torch.zeros((g.groups, h.shape[1]), device=h.device).index_add(0, g.group, h) / g.group_count[:, None]
        spatial = torch.zeros((g.spatial_features, h.shape[1]), device=h.device).index_add(0, g.spatial_pool, h[g.spatial_nodes]) / g.spatial_counts[:, None]
        return torch.cat([type_features, spatial]).T

    def forward(self, movies, mode='real', diagnostics=False, route=None):
        g = self.graph; original_batch = len(movies)
        # A separately propagated gray reference removes resting activity from
        # the measurement. It carries no image information into any other node.
        movies = torch.cat([movies, torch.full_like(movies[:1], .5)])
        batch = len(movies)
        retina, odor = self.inputs(movies, route)
        blank = torch.full_like(movies[:1, :1], .5)
        retina_blank, odor_blank = self.inputs(blank, route)
        tau = (.02 + .3 * self.tau_raw.sigmoid())[g.group, None]
        bias = (.5 + .5 * self.bias_raw.tanh())[g.group, None]
        pre = (.25 + 1.5 * self.pre_gain.sigmoid())[g.group, None]
        post = (.25 + 1.5 * self.post_gain.sigmoid())[g.group, None]
        h = bias.clone()
        for t in range(self.warmup):
            u = torch.zeros_like(h).index_copy(0, g.retina, retina_blank[0]).index_copy(0, g.olfactory, odor_blank[0])
            h = h + (self.dt / tau) * (-h + bias + g.propagate(pre * F.relu(h), mode) * post + u)
        h = h.expand(-1, batch).contiguous()
        for t in range(movies.shape[1]):
            r, o = retina[t], odor[t]
            u = torch.zeros_like(h).index_copy(0, g.retina, r).index_copy(0, g.olfactory, o)
            current = g.propagate(pre * F.relu(h), mode) * post
            h = h + (self.dt / tau) * (-h + bias + current + u)
        all_features = self.state_features(h)
        features = all_features[:original_batch] - all_features[original_batch:]
        logits = self.readout((features - self.feature_center) * self.feature_scale)
        return (logits, h, features) if diagnostics else logits

    @torch.no_grad()
    def calibrate(self, movies, mode='real', spatial_normalization='location'):
        self.eval()
        chunks = [self(x, mode, True)[2] for x in movies.split(8)]
        x = torch.cat(chunks)
        self.feature_center.copy_(x.mean(0))
        self.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(.003).reciprocal().clamp_max(300))
        if spatial_normalization == 'channel':
            # The gray response is already subtracted. One scale per optic
            # cell type preserves translation consistency across location bins.
            spatial=x[:, self.graph.groups:].reshape(-1,self.graph.spatial_channels,self.graph.spatial_bins,self.graph.spatial_bins)
            scale=spatial.square().mean((0,2,3)).sqrt().clamp_min(.003).reciprocal().clamp_max(300)
            self.feature_center[self.graph.groups:]=0
            self.feature_scale[self.graph.groups:]=scale[:,None,None].expand(-1,self.graph.spatial_bins,self.graph.spatial_bins).flatten()
        elif spatial_normalization != 'location':
            raise ValueError(spatial_normalization)
        self.spec['spatial_normalization']=spatial_normalization
        return {'train_calibration_examples': len(x), 'features_with_std_gt_1e_5': int((x.std(0) > 1e-5).sum()),
            'total_features': x.shape[1], 'max_scale': float(self.feature_scale.max()), 'spatial_normalization':spatial_normalization}


def make_movies(seed, groups=4, frames=12, size=48, ood=False, device='cuda'):
    """Each quartet has identical final pixels and four distinct motion labels.

    Random train/dev/test streams have independent shape, contrast, position,
    texture and size. A held-out OOD set additionally changes speed/contrast.
    """
    rng = np.random.default_rng(seed)
    grid = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    movies, labels = [], []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for k in range(groups):
        cx, cy = rng.uniform(-.38, .38, 2)
        shape = int(rng.integers(3)); polarity = int(rng.integers(2))
        radius = rng.uniform(.16, .25); contrast = rng.uniform(.20, .45) if not ood else rng.uniform(.10, .20)
        distance = rng.uniform(.2, .4) if not ood else rng.uniform(.4, .55)
        phase = rng.uniform(0, 2 * np.pi)
        background = .025 * torch.sin(xx * rng.uniform(3, 8) + yy * rng.uniform(3, 8) + phase)
        for label, (dx, dy) in enumerate(directions):
            clip = []
            for t in range(frames):
                fraction = (frames - 1 - t) / (frames - 1)
                x, y = xx - (cx - dx * distance * fraction), yy - (cy - dy * distance * fraction)
                if shape == 0: signed = radius - torch.sqrt(x*x + y*y + 1e-8)
                elif shape == 1: signed = radius - torch.maximum(x.abs(), y.abs())
                else: signed = radius - (x.abs() + y.abs())
                mask = torch.sigmoid(signed / .025)
                clip.append((.5 + background + (2 * polarity - 1) * contrast * mask).clamp(0, 1))
            movies.append(torch.stack(clip))
            labels.append([label, shape, int(cx > 0) + 2 * int(cy > 0), polarity])
    x = torch.stack(movies); y = torch.tensor(labels, device=device)
    assert torch.equal(x[0, -1], x[1, -1]) and torch.equal(x[0, -1], x[3, -1])
    return x, y


SLICES = [(0, 4), (4, 7), (7, 11), (11, 13)]
TASKS = ['motion', 'shape', 'position', 'polarity']


def task_loss(logits, labels):
    return sum(F.cross_entropy(logits[:, lo:hi], labels[:, k]) for k, (lo, hi) in enumerate(SLICES)) / 4


def metrics(logits, labels):
    return {name: float((logits[:, lo:hi].argmax(1) == labels[:, k]).float().mean())
            for k, ((lo, hi), name) in enumerate(zip(SLICES, TASKS))}
