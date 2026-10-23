"""Task-trained full MaleCNS model; topology and neurotransmitter signs retained.

This is a computational graph model, not a physiological fly simulation.
All states and retained directed edges are used. MRI input mapping, type-shared
positive presynaptic/postsynaptic gains, neural biases, and the readout learn.
There is no image-to-classifier skip and no original-model blend in this model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn


class FixedSparseMM(torch.autograd.Function):
    """Differentiate only dense activity, with an explicitly cached CSR transpose."""
    @staticmethod
    def forward(ctx, activity, adjacency, transpose):
        ctx.transpose = transpose
        return torch.sparse.mm(adjacency, activity)

    @staticmethod
    def backward(ctx, derivative):
        return torch.sparse.mm(ctx.transpose, derivative.contiguous()), None, None


class FullGraph(nn.Module):
    def __init__(self, root, device='cuda', mapping=None, seed=20261009):
        super().__init__()
        root = Path(root)
        ids = np.load(root / 'body_ids.npy')
        nt = pd.read_feather(root / 'neurotransmitters.feather').set_index('body').reindex(ids)
        transmitter = nt.consensus_nt.fillna('unknown').astype(str)
        sign = np.where(transmitter.isin(['gaba', 'glutamate']), -1, 1).astype(np.float32)
        counts = sp.load_npz(root / 'counts.npz').tocsr()
        incoming = np.asarray(counts.sum(1)).ravel()
        adjacency = counts.copy()
        adjacency.data *= sign[adjacency.indices]
        adjacency = (sp.diags(1 / np.maximum(incoming, 1)) @ adjacency).tocsr()
        adjacency.sort_indices()
        def csr(a):
            return torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int32)),
                torch.from_numpy(a.indices.astype(np.int32)),
                torch.from_numpy(a.data.astype(np.float32)), size=a.shape, device=device)
        self.register_buffer('adjacency', csr(adjacency), persistent=False)
        self.register_buffer('transpose', csr(adjacency.T.tocsr()), persistent=False)
        if mapping is None:
            annotation = pd.read_feather(root / 'body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId').reindex(ids)
            annotation = annotation[['type', 'superclass', 'class', 'somaSide']].astype('string').fillna('unknown')
            keys = np.where(annotation['type'].ne('unknown'),
                'type:' + annotation['type'] + '|nt:' + transmitter.to_numpy(),
                'untyped:' + annotation['superclass'] + '|' + annotation['class'] + '|' +
                annotation['somaSide'] + '|nt:' + transmitter.to_numpy()).astype(str)
            names, group = np.unique(keys, return_inverse=True)
        else:
            with np.load(mapping, allow_pickle=False) as z:
                assert np.array_equal(ids, z['body_ids']), 'Graph/body mapping changed'
                names, group = z['group_names'], z['group_index']
        self.n = len(ids)
        self.groups = len(names)
        self.register_buffer('group', torch.from_numpy(group.astype(np.int64)).to(device), persistent=False)
        self.register_buffer('group_count', torch.from_numpy(np.bincount(group).astype(np.float32)).to(device), persistent=False)
        permutation = np.random.default_rng(seed).permutation(self.n)
        self.register_buffer('shuffle', torch.from_numpy(permutation.astype(np.int64)).to(device), persistent=False)
        self.mapping = {'body_ids': ids, 'group_index': group.astype(np.int32), 'group_names': names}
        self.metadata = {'nodes': self.n, 'edges': adjacency.nnz, 'groups': self.groups,
            'graph_seed': seed, 'normalization': 'signed synapse counts / total incoming synapse count',
            'grouping': 'annotated cell type + neurotransmitter; untyped: superclass/class/side/NT',
            'shuffle': 'fixed permutation of adjacency destination rows; same weights and edge count',
            'nt_sign': 'GABA/glutamate negative, other/unknown positive; modelling simplification'}
        assert self.n == 199713 and adjacency.nnz == 26013498

    def save_mapping(self, path):
        np.savez_compressed(path, **self.mapping)

    def propagate(self, h, mode):
        if mode == 'zero':
            return torch.zeros_like(h)
        if mode == 'bypass':
            return h
        m = FixedSparseMM.apply(h, self.adjacency, self.transpose)
        return m[self.shuffle] if mode == 'shuffle' else m


class TrainableWholeBrain(nn.Module):
    def __init__(self, graph, input_dim=12288, latent_dim=64, steps=4, dropout=.2):
        super().__init__()
        self.graph = graph
        self.steps = steps
        self.adapter = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.routing = nn.Linear(latent_dim, graph.groups)
        self.pre_gain = nn.Parameter(torch.zeros(graph.groups))
        self.post_gain = nn.Parameter(torch.zeros(graph.groups))
        self.neural_bias = nn.Parameter(torch.zeros(graph.groups))
        self.readout = nn.Sequential(nn.LayerNorm(graph.groups), nn.Dropout(dropout), nn.Linear(graph.groups, 12))
        nn.init.normal_(self.readout[-1].weight, std=.003)
        nn.init.zeros_(self.readout[-1].bias)
        self.spec = {'architecture': 'trainable_full_malecns_v1', 'input_dim': input_dim,
            'latent_dim': latent_dim, 'steps': steps, 'dropout': dropout,
            'residual_activity': .25, 'new_message_activity': .75,
            'gain': '0.5 + sigmoid(raw), preserves each original connection sign and support',
            'recurrence': 'h0=tanh(route(adapter(x))); m=A*(g_pre*h); m=g_post*m; m=m/RMS_nodes(m); h=.25*h+.75*tanh(m+bias)',
            'readout': 'mean of ALL final states by cell-type group, LayerNorm, linear 12 logits',
            'no_image_to_readout_skip': True, 'original_model_blend_weight': 0.0}

    def forward(self, x, mode='real', diagnostics=False):
        graph = self.graph
        h = torch.tanh(self.routing(self.adapter(x)))[:, graph.group].T.contiguous()
        pre = (.5 + self.pre_gain.sigmoid())[graph.group, None]
        post = (.5 + self.post_gain.sigmoid())[graph.group, None]
        bias = (.25 * self.neural_bias.tanh())[graph.group, None]
        records = []
        for step in range(self.steps):
            raw = graph.propagate(pre * h, mode) * post
            # Per-examination gain normalization prevents averaging from washing
            # out the propagated signal. No batch/test-cohort statistics used.
            rms = raw.square().mean(dim=0, keepdim=True).add(1e-6).sqrt()
            m = raw / rms
            h = .25 * h + .75 * torch.tanh(m + bias)
            if diagnostics:
                records.append({'step': step, 'raw_message_rms': float(raw.detach().square().mean().sqrt()),
                    'normalized_message_rms': float(m.detach().square().mean().sqrt()),
                    'state_rms': float(h.detach().square().mean().sqrt())})
        pooled = torch.zeros((graph.groups, len(x)), device=x.device, dtype=x.dtype)
        pooled = pooled.index_add(0, graph.group, h) / graph.group_count[:, None]
        logits = self.readout(pooled.T)
        return (logits, h, records) if diagnostics else logits


class ConventionalMLP(nn.Module):
    def __init__(self, input_dim=12288):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(.2), nn.Linear(128, 512), nn.GELU(), nn.Dropout(.2), nn.Linear(512, 12))
        self.spec = {'architecture': 'conventional_mlp', 'input_dim': input_dim,
                     'hidden_dims': [128, 512], 'dropout': .2}

    def forward(self, x, mode=None):
        return self.net(x)


def save_checkpoint(path, model, **metadata):
    # Sparse topology/mapping are immutable external artifacts, not repeated in
    # each checkpoint. state_dict contains only trainable network parameters.
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'spec': model.spec, **metadata}, path)


def load_checkpoint(path, graph, device='cuda'):
    record = torch.load(path, map_location=device, weights_only=True)
    spec = record['spec']
    if spec['architecture'] == 'conventional_mlp':
        model = ConventionalMLP(spec['input_dim'])
    else:
        model = TrainableWholeBrain(graph, spec['input_dim'], spec['latent_dim'], spec['steps'], spec['dropout'])
    model.to(device).load_state_dict(record['state_dict'])
    return model, record
