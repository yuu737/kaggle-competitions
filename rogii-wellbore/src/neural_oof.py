"""Neural TCN tracker — GroupKFold OOF over many wells. Saves OOF deviation per id
for honest evaluation and blend integration.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, flush=True)

CSV = 'kaggle_datasets/wellbore-geology-prediction-artifacts/data/train.csv'
hdr = pd.read_csv(CSV, nrows=0).columns.tolist()
feats = [c for c in hdr if c not in ('well', 'id', 'target')]
NF = len(feats)
NWELLS = 400

t0 = time.time(); keep = []; ws = []
for ch in pd.read_csv(CSV, usecols=['well', 'id', 'target'] + feats, chunksize=400000):
    for w in ch['well'].unique():
        if w not in ws:
            ws.append(w)
    sel = set(w for i, w in enumerate(ws) if i % 2 == 0)
    keep.append(ch[ch['well'].isin(sel)])
    if pd.concat([k[['well']] for k in keep])['well'].nunique() >= NWELLS:
        break
sel = set(w for i, w in enumerate(ws) if i % 2 == 0)
df = pd.concat(keep); df = df[df['well'].isin(sel)].copy()
df = df[df['well'].notna() & df['target'].notna()].copy()
df['well'] = df['well'].astype(str)
df['row'] = df['id'].str[9:].astype(int)
df = df.sort_values(['well', 'row']).reset_index(drop=True)
df[feats] = df[feats].astype('float32')
print('loaded %d rows %d wells %.0fs' % (len(df), df['well'].nunique(), time.time() - t0), flush=True)


class TCN(nn.Module):
    def __init__(self, nf, h=128):
        super().__init__()

        def blk(ci, co, d):
            return nn.Sequential(nn.Conv1d(ci, co, 3, padding=d, dilation=d), nn.ReLU(), nn.BatchNorm1d(co), nn.Dropout(0.15))
        self.net = nn.Sequential(blk(nf, h, 1), blk(h, h, 2), blk(h, h, 4), blk(h, h, 8), blk(h, h, 16), blk(h, h, 32))
        self.head = nn.Conv1d(h, 1, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


def batches(data, bs=8, shuf=True):
    order = list(range(len(data)))
    if shuf:
        np.random.shuffle(order)
    for i in range(0, len(data), bs):
        grp = [data[j] for j in order[i:i + bs]]
        T = max(len(x) for x, _ in grp)
        X = np.zeros((len(grp), NF, T), 'float32'); Y = np.zeros((len(grp), T), 'float32'); M = np.zeros((len(grp), T), 'float32')
        for k, (x, y) in enumerate(grp):
            X[k, :, :len(x)] = x.T; Y[k, :len(y)] = y; M[k, :len(y)] = 1
        yield torch.tensor(X).to(DEV), torch.tensor(Y).to(DEV), torch.tensor(M).to(DEV)


wells = df['well'].values
uniqw = np.array(sorted(df['well'].unique()))
rng0 = np.random.default_rng(0); perm = rng0.permutation(len(uniqw))
folds = np.array_split(perm, 5)
oof = np.full(len(df), np.nan, dtype='float32')
EPOCHS = 32
for fold in range(5):
    vai = folds[fold]; tri = np.concatenate([folds[f] for f in range(5) if f != fold])
    trw = set(uniqw[tri]); vaw = set(uniqw[vai])
    trmask = df['well'].isin(trw).values
    mu = np.nanmean(df.loc[trmask, feats].values, 0)
    sd = np.nanstd(df.loc[trmask, feats].values, 0); sd[sd < 1e-6] = 1
    Xall = np.nan_to_num((df[feats].values - mu) / sd, nan=0.0).astype('float32')
    yall = df['target'].values.astype('float32')

    def seqs(wset):
        return [(Xall[np.where(wells == w)[0]], yall[np.where(wells == w)[0]], np.where(wells == w)[0]) for w in wset]
    tr = [(a, b) for a, b, _ in seqs(trw)]
    va_full = seqs(vaw)
    net = TCN(NF).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        net.train()
        for X, Y, M in batches(tr, 8, True):
            opt.zero_grad(); p = net(X); l = (((p - Y) ** 2) * M).sum() / M.sum()
            l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        sched.step()
    # predict held-out wells
    net.eval()
    with torch.no_grad():
        for x, y, idxs in va_full:
            xt = torch.tensor(x.T[None]).to(DEV)
            p = net(xt).squeeze(0).cpu().numpy()
            oof[idxs] = p[:len(idxs)]
    r = np.sqrt(np.nanmean((oof[df['well'].isin(vaw).values] - df.loc[df['well'].isin(vaw), 'target'].values) ** 2))
    print('fold %d done, fold val pooled=%.4f, %.0fs' % (fold, r, time.time() - t0), flush=True)

y = df['target'].values
pooled = np.sqrt(np.nanmean((oof - y) ** 2))
print('\n=== NEURAL OOF pooled (GroupKFold, %d wells) = %.4f ===' % (df['well'].nunique(), pooled))
print('(GBDT stacker OOF ~11, PF ~11-12 on representative wells, hold ~16)')
out = pd.DataFrame({'id': df['id'].values, 'well': df['well'].values, 'neural_dev': oof, 'target': y})
out.to_csv('output/neural_oof.csv', index=False)
print('saved output/neural_oof.csv')
