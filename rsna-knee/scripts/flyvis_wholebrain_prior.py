"""Transfer named Flyvis dynamics to existing MaleCNS cells and edges only.

Connectomes differ in species specimen, cell inventory and spatial layout.
This transfer is an explicit engineering hypothesis, not biological equivalence.
Synapse density is calibrated per type pair; every original MaleCNS edge remains.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from functional_vision import SensoryGraph, SensoryWholeBrain, GRAPH_ROOT, OUT

REFERENCE = OUT / 'flyvis_reference'


def reference_name(name):
    if name == 'R1-R6' or name in [f'R{i}' for i in range(1, 7)]: return 'R1-R6'
    if name.startswith('R7') and not name.startswith('R7R8'): return 'R7'
    if name.startswith('R8'): return 'R8'
    return name


def build_prior(root=GRAPH_ROOT, out=OUT):
    root, out = Path(root), Path(out)
    ids = np.load(root / 'body_ids.npy')
    a = pd.read_feather(root / 'body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId').reindex(ids)
    nt = pd.read_feather(root / 'neurotransmitters.feather').set_index('body').reindex(ids).consensus_nt.fillna('unknown')
    dynamics = pd.read_csv(REFERENCE / 'type_dynamics.csv')
    dynamics['canonical'] = dynamics.type.map(reference_name)
    dynamics = dynamics.groupby('canonical', sort=False)[['bias', 'tau']].mean()
    names = dynamics.index.tolist(); ntype = len(names) + 1
    lookup = {t: i+1 for i, t in enumerate(names)}
    mapped = np.array([lookup.get(reference_name(str(t)), 0) for t in a.type], dtype=np.int32)
    pair = pd.read_csv(REFERENCE / 'type_pair_weights.csv')
    original_targets = pair.target_type.copy()
    pair['source_canonical'] = pair.source_type.map(reference_name)
    pair['target_canonical'] = pair.target_type.map(reference_name)
    # Merge six outer receptor source layers by summing incoming current; when
    # the target is R1-R6, average over its six native target layers.
    desired = pair.groupby(['source_canonical', 'target_canonical'], sort=False).mean_total_weight.sum()
    target_multiplicity = pd.Series(original_targets.unique()).map(reference_name).value_counts()
    target_weight = np.full((ntype, ntype), np.nan, dtype=np.float32)
    for (src, tgt), value in desired.items():
        target_weight[lookup[tgt], lookup[src]] = value / target_multiplicity[tgt]
    counts = sp.load_npz(root / 'counts.npz').tocsr().astype(np.float32)
    row = np.repeat(np.arange(len(ids), dtype=np.int32), np.diff(counts.indptr))
    key = mapped[row] * ntype + mapped[counts.indices]
    mass = np.bincount(key, weights=counts.data, minlength=ntype*ntype).reshape(ntype, ntype)
    population = np.bincount(mapped, minlength=ntype).clip(1)
    mean_count = mass / population[:, None]
    defined = np.isfinite(target_weight) & (mean_count > 0)
    factor = np.zeros((ntype, ntype), dtype=np.float32)
    # If a learned pair is zero, retain a tiny nonzero native-sign connection;
    # explicitly report this support-preserving departure from the reference.
    signs = pair.groupby(['source_canonical', 'target_canonical'], sort=False).sign.first()
    zero_pairs = 0
    for ti, si in zip(*np.where(defined)):
        value = target_weight[ti, si]
        if abs(value) < 1e-4:
            value = float(signs.loc[(names[si-1], names[ti-1])]) * 1e-4
            zero_pairs += 1
        factor[ti, si] = value / mean_count[ti, si]
    mask = defined.reshape(-1)[key]
    incoming = np.asarray(counts.sum(1)).ravel().clip(1)
    sign = np.where(nt.isin(['gaba', 'glutamate', 'histamine']), -1., 1.).astype(np.float32)
    weights = counts.data * sign[counts.indices] / incoming[row] * .2
    # Normalizing by a type-wide mean produced huge currents in unusually
    # well-connected cells (including a Lawf2 cell). Match each target cell's
    # total input from a source type instead; preserve relative synapse counts.
    node_source_key = row.astype(np.int64) * ntype + mapped[counts.indices]
    node_source_mass = np.bincount(node_source_key, weights=counts.data, minlength=len(ids)*ntype)
    desired_current = (factor * mean_count).reshape(-1)
    weights[mask] = counts.data[mask] * desired_current[key[mask]] / node_source_mass[node_source_key[mask]]
    assert np.isfinite(weights).all() and np.all(weights != 0)
    prior = sp.csr_matrix((weights, counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
    assert prior.nnz == 26013498 and np.array_equal(prior.indices, counts.indices)
    sp.save_npz(out / 'wholebrain_flyvis_prior.npz', prior)
    dynamics.to_csv(out / 'transferred_type_dynamics.csv')
    record = {'method': 'Named type-pair expected total current from native interior cells; rescale existing MaleCNS synapse counts separately for EACH target neuron and source type, preserving relative synapse counts',
        'reference_model': 'flow/0000/000', 'nodes': len(ids), 'edges': prior.nnz,
        'matched_dynamics_nodes': int((mapped > 0).sum()), 'matched_dynamics_types': len(np.unique(mapped[mapped > 0])),
        'transferred_edges': int(mask.sum()), 'unchanged_support': True,
        'support_floor_pairs': zero_pairs, 'support_floor_mean_current': 1e-4,
        'other_edges': 'Signed original synapse count / incoming total, multiplied by 0.2; all retained',
        'ambiguous_type_mapping': 'R1-R6 source currents summed; target bias/tau/current averaged across six native layers. R7/R8 subtypes mapped to R7/R8; CT1 subtype correspondence not guessed.',
        'positive_negative_weights': [int((weights > 0).sum()), int((weights < 0).sum())],
        'reference_pair_csv_sha256': hashlib.sha256((REFERENCE/'type_pair_weights.csv').read_bytes()).hexdigest()}
    (out / 'prior_transfer.json').write_text(json.dumps(record, indent=2))
    print('PRIOR', json.dumps(record), flush=True)
    return record


class PriorSensoryGraph(SensoryGraph):
    def __init__(self, root=GRAPH_ROOT, out=OUT, device='cuda', seed=20261016, spatial_bins=4):
        super().__init__(root, device, seed, spatial_bins)
        out = Path(out)
        if not (out / 'wholebrain_flyvis_prior.npz').exists(): build_prior(root, out)
        a = sp.load_npz(out / 'wholebrain_flyvis_prior.npz').tocsr()
        def csr(m):
            return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int32)),
                torch.from_numpy(m.indices.astype(np.int32)), torch.from_numpy(m.data.astype(np.float32)), size=m.shape, device=device)
        self.adjacency, self.transpose = csr(a), csr(a.T.tocsr())
        self.metadata['prior_transfer'] = json.loads((out / 'prior_transfer.json').read_text())
        dyn = pd.read_csv(out / 'transferred_type_dynamics.csv').set_index('canonical')
        bias = np.full(self.groups, .5, dtype=np.float32); tau = np.full(self.groups, .05, dtype=np.float32)
        for i, name in enumerate(self.mapping['group_names']):
            typ = reference_name(name.split('|')[0].removeprefix('type:'))
            if typ in dyn.index: bias[i], tau[i] = dyn.loc[typ, ['bias', 'tau']]
        self.prior_bias, self.prior_tau = bias, tau


def make_prior_model(graph, route='visual', readout_hidden=0):
    model = SensoryWholeBrain(graph, route, warmup=50, readout_hidden=readout_hidden).to(graph.adjacency.device)
    with torch.no_grad():
        b = torch.as_tensor(graph.prior_bias, device=graph.adjacency.device)
        t = torch.as_tensor(graph.prior_tau, device=graph.adjacency.device)
        model.bias_raw.copy_(torch.atanh((2*b-1).clamp(-.99999,.99999)))
        model.tau_raw.copy_(torch.logit(((t-.02)/.3).clamp(1e-5,1-1e-5)))
    model.spec['architecture'] = 'flyvis_prior_full_malecns_v1'
    model.spec['visual_prior'] = 'Named official model 000 parameters transferred onto original MaleCNS topology; no native graph substitution'
    return model


if __name__ == '__main__': build_prior()
