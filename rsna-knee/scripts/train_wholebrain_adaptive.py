"""Bounded, study-separated training and controls for task-trained MaleCNS."""
from __future__ import annotations
from pathlib import Path
import argparse, gc, hashlib, json, time
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import roc_auc_score
from trainable_wholebrain import FullGraph, TrainableWholeBrain, ConventionalMLP, FixedSparseMM, save_checkpoint, load_checkpoint
from public_brain_readout import normalize, weighted_bce, fit, predict
from mri_features import LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/adaptive_wholebrain'


def load_data(root):
    root = Path(root)
    parts = []
    for f in sorted((root / 'features').glob('features_*.npz')):
        with np.load(f) as z:
            parts.append({k: z[k] for k in ['ids', 'features', 'split', 'predictions']})
    data = {k: np.concatenate([z[k] for z in parts]) for k in parts[0]}
    order = np.argsort(data['ids'].astype(str))
    data = {k: v[order] for k, v in data.items()}
    ids = data['ids'].astype(str)
    assert len(ids) == len(set(ids)) == 4339
    weak = pd.read_csv(root / 'weak_labels.csv').set_index('StudyInstanceUID').reindex(ids)
    gold = pd.read_csv(root / 'gold_labels.csv').set_index('StudyInstanceUID')
    y = weak[LABELS].to_numpy(np.float32)
    w = weak[[s + '__weight' for s in LABELS]].to_numpy(np.float32)
    eligible = (np.nan_to_num(w).sum(1) > 0) & (~np.isin(ids, gold.index)) & (data['split'] == 'train')
    bucket = np.array([int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 for uid in ids])
    splits = {'train': np.flatnonzero(eligible & (bucket >= 6)),
              'dev': np.flatnonzero(eligible & (bucket >= 3) & (bucket < 6)),
              'test': np.flatnonzero(eligible & (bucket < 3)),
              'gold58': np.flatnonzero(np.isin(ids, gold.index)),
              'public3': np.flatnonzero(data['split'] == 'test')}
    assert sum(len(splits[k]) for k in ['train', 'dev', 'test']) == 4278
    assert not set(ids[splits['train']]) & set(ids[splits['dev']])
    assert not set(ids[splits['train']]) & set(ids[splits['test']])
    assert not set(ids[eligible]) & set(gold.index)
    data.update(x=normalize(data.pop('features')), y=np.nan_to_num(y, nan=.5), w=np.nan_to_num(w),
                splits=splits, gold=gold, ids=ids)
    return data


def metrics(y, w, p):
    scores = {}
    for j, label in enumerate(LABELS):
        use = w[:, j] > 0
        target = y[use, j] >= .5
        assert len(np.unique(target)) == 2, f'{label}: split lacks a class'
        scores[label] = float(roc_auc_score(target, p[use, j]))
    return {'macro_auc': float(np.mean(list(scores.values()))), 'per_label': scores,
            'weighted_soft_bce': weighted_bce(y, p, w)}


def balanced_weights(y, w, train):
    # Training-derived class balance. Missing targets remain exactly weight zero.
    positive = (w[train] * (y[train] >= .5)).sum(0)
    negative = (w[train] * (y[train] < .5)).sum(0)
    assert (positive > 0).all() and (negative > 0).all()
    weights = w * np.where(y >= .5, len(train) / (2 * positive), len(train) / (2 * negative))
    return weights.astype(np.float32)


def masked_loss(logits, y, w):
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * w).mean()


@torch.no_grad()
def evaluate(model, x, indices, mode, batch_size):
    model.eval()
    return np.concatenate([model(x[indices[i:i + batch_size]], mode).sigmoid().cpu().numpy()
                           for i in range(0, len(indices), batch_size)])


def sparse_gradient_check(device):
    a = torch.tensor([[0., 2., 0.], [-1., 0., .4], [1., 0., 0.]], device=device)
    x = torch.randn(3, 4, device=device, requires_grad=True)
    target = torch.randn(3, 4, device=device)
    actual = FixedSparseMM.apply(x, a.to_sparse_csr(), a.T.contiguous().to_sparse_csr())
    gradient = torch.autograd.grad((actual * target).sum(), x)[0]
    difference = float((gradient - a.T @ target).abs().max())
    assert difference < 1e-6
    return difference


def smoke(data, graph, out, device, batch_size):
    torch.manual_seed(20261009)
    model = TrainableWholeBrain(graph).to(device)
    model.eval()
    idx = data['splits']['train'][:batch_size]
    x = torch.from_numpy(data['x'][idx]).to(device)
    y = torch.from_numpy(data['y'][idx]).to(device)
    w = torch.from_numpy(data['w'][idx]).to(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    exact = sparse_gradient_check(device)
    logits, h, trace = model(x, 'real', True)
    h.retain_grad()
    loss = masked_loss(logits, y, w)
    altered = torch.where(w == 0, 1 - y, y)
    assert torch.equal(loss, masked_loss(logits, altered, w)), 'Masked labels affect loss'
    loss.backward()
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters()
                 if name in ['adapter.0.weight', 'routing.weight', 'pre_gain', 'post_gain', 'neural_bias', 'readout.2.weight']}
    assert all(v > 0 for v in gradients.values()), 'A required learning path has no gradient'
    count = int((h.grad.abs().sum(1) > 0).sum())
    assert count == graph.n
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in gradients}
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    opt.step()
    updates = {n: float((dict(model.named_parameters())[n].detach() - p).norm()) for n, p in before.items()}
    assert all(v > 0 for v in updates.values())
    del logits, loss, h, before
    interventions = {}
    with torch.no_grad():
        original = model(x, 'real').sigmoid()
        for mode in ['zero', 'shuffle', 'bypass']:
            delta = abs(original - model(x, mode).sigmoid())
            interventions[mode] = {'max_delta': float(delta.max()), 'mean_delta': float(delta.mean()),
                'changed_entries': int((delta > 1e-6).sum()), 'total_entries': delta.numel()}
            assert float(delta.max()) > 1e-5
    # Warm up and time real MRI forward/backward/optimizer steps.
    model.train()
    durations = []
    for _ in range(4):
        if device == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        masked_loss(model(x, 'real'), y, w).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if device == 'cuda': torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    graph.save_mapping(out / 'neuron_type_mapping.npz')
    result = {'real_mri_studies': len(idx), 'graph': graph.metadata, 'model': model.spec,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'gradient_norms': gradients, 'parameter_update_l2': updates,
              'all_final_states_with_nonzero_loss_gradient': count,
              'custom_sparse_backward_max_error': exact, 'masked_target_invariance': True,
              'fixed_weight_interventions': interventions, 'step_activity': trace,
              'training_step_seconds': durations, 'median_step_seconds_after_warmup': float(np.median(durations[1:])),
              'peak_vram_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
              'rss_bytes': psutil.Process().memory_info().rss,
              'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}
    (out / 'smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    del model, opt
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return result


def train_one(data, graph, x, y, w, out, mode, seed, epochs, batch_size, deadline):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = (ConventionalMLP() if mode == 'mlp' else TrainableWholeBrain(graph)).to(x.device)
    internal = [p for n, p in model.named_parameters() if n in {'pre_gain', 'post_gain', 'neural_bias'}]
    other = [p for p in model.parameters() if all(p is not a for a in internal)]
    opt = torch.optim.AdamW([{'params': other, 'lr': .0005}, {'params': internal, 'lr': .002}], weight_decay=.001)
    train, dev = data['splits']['train'], data['splits']['dev']
    key = f'{mode}_seed{seed}'
    best = -np.inf
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Bounded training time exhausted; checkpoints preserved')
        epoch_start = time.perf_counter()
        schedule = .2 + .8 * .5 * (1 + np.cos(np.pi * (epoch - 1) / epochs))
        opt.param_groups[0]['lr'] = .0005 * schedule
        opt.param_groups[1]['lr'] = .002 * schedule
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train)
        total = 0.
        for i in range(0, len(order), batch_size):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Bounded training time exhausted; checkpoints preserved')
            ix = order[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model(x[ix], mode), y[ix], w[ix])
            assert torch.isfinite(loss), f'Nonfinite training loss: {key}'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += float(loss.detach()) * len(ix)
        p = evaluate(model, x, dev, mode, batch_size)
        score = metrics(data['y'][dev], data['w'][dev], p)
        record = {'key': key, 'epoch': epoch, 'train_objective': total / len(train),
                  'dev': score, 'seconds': time.perf_counter() - epoch_start}
        history.append(record)
        if score['macro_auc'] > best:
            best = score['macro_auc']
            save_checkpoint(out / (key + '.pt'), model, mode=mode, seed=seed, selected_epoch=epoch,
                            dev_macro_auc=best, train_studies=len(train))
        with (out / 'training_history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps({'key': key, 'epoch': epoch, 'dev_auc': round(score['macro_auc'], 6),
                          'loss': round(total / len(train), 6), 'seconds': round(record['seconds'], 2)}), flush=True)
    del model, opt
    gc.collect()
    if x.is_cuda: torch.cuda.empty_cache()
    return {'key': key, 'mode': mode, 'seed': seed, 'epochs': epochs, 'best_dev_auc': best,
            'seconds': time.perf_counter() - started, 'history': history}


def train(data, graph, out, device, batch_size, epochs, seconds_limit):
    started = time.perf_counter()
    deadline = started + seconds_limit
    cases = [('real', 20261009), ('shuffle', 20261009), ('bypass', 20261009), ('mlp', 20261009),
             ('real', 20261010), ('shuffle', 20261010)]
    protocol = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': 'Train MRI adapter, neural type gains/biases, and whole-brain readout end to end locally.',
        'cases': cases, 'epochs': epochs, 'batch_size': batch_size, 'local_seconds_limit': seconds_limit,
        'checkpoint_selection': 'Highest masked weak-label dev macro ROC-AUC; test and Gold58 not inspected during training.',
        'splits': {k: len(v) for k, v in data['splits'].items()},
        'split_rule': 'sha256(StudyInstanceUID) first 8 hex %20: 0..2 test, 3..5 dev, 6..19 train; Gold58 always excluded.',
        'supervision': 'Same pre-existing two-source agreement soft labels and confidence mask; training-only class balancing.',
        'loss': 'Mean confidence-masked, class-balanced soft-target BCE over samples and 12 labels; AdamW weight decay .001.',
        'optimizer': 'AdamW; lr .0005 adapter/routing/readout and .002 type gains/bias; cosine to 20%; clip norm 5.',
        'independence_limit': 'New trainable components never see held-out study labels. Public encoder and weak-label source provenance prevent an independent clinical-validation claim.',
        'no_gold_or_lb_tuning': True, 'competition_submission': 'No new competition submissions authorized.',
        'references': ['https://www.nature.com/articles/s41586-024-07939-3',
                       'https://docs.pytorch.org/docs/2.10/generated/torch.sparse.mm.html']}
    path = out / 'protocol.json'
    assert not path.exists(), 'Existing experiment protocol; do not silently overwrite/restart'
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    split_names = np.full(len(data['ids']), 'excluded', dtype='<U12')
    for tag, ix in data['splits'].items(): split_names[ix] = tag
    pd.DataFrame({'StudyInstanceUID': data['ids'], 'split': split_names}).to_csv(out / 'study_splits.csv', index=False)
    x = torch.from_numpy(data['x']).to(device)
    y = torch.from_numpy(data['y']).to(device)
    balanced = balanced_weights(data['y'], data['w'], data['splits']['train'])
    w = torch.from_numpy(balanced).to(device)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    records = []
    for mode, seed in cases:
        records.append(train_one(data, graph, x, y, w, out, mode, seed, epochs, batch_size, deadline))
    result = {'completed_cases': records, 'training_seconds': time.perf_counter() - started,
              'peak_vram_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
              'rss_bytes': psutil.Process().memory_info().rss,
              'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}
    (out / 'training_complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'completed_cases'}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['smoke', 'train'])
    p.add_argument('--features', type=Path, default=ROOT / 'outputs/public_wholebrain/kaggle_features')
    p.add_argument('--graph', type=Path, default=ROOT / 'data/connectome/male-cns-v1.0')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seconds-limit', type=int, default=3600)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.features)
    graph = FullGraph(args.graph, device)
    if args.action == 'smoke':
        smoke(data, graph, args.out, device, args.batch_size)
    else:
        assert (args.out / 'smoke.json').exists(), 'Run the full-graph MRI training smoke first'
        train(data, graph, args.out, device, args.batch_size, args.epochs, args.seconds_limit)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
