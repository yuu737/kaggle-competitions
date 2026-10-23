"""Neural sequence tracker (TCN) over the 196 features, per-well trajectory.
v2: CUDA, LR schedule, grad clip, longer training; fair GBDT baseline on the SAME
val wells; saves best val predictions for blend testing.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)

CSV = 'kaggle_datasets/wellbore-geology-prediction-artifacts/data/train.csv'
hdr = pd.read_csv(CSV, nrows=0).columns.tolist()
feats = [c for c in hdr if c not in ('well', 'id', 'target')]
NF = len(feats)

t0 = time.time(); keep = []; ws = []
for ch in pd.read_csv(CSV, usecols=['well', 'id', 'target'] + feats, chunksize=400000):
    for w in ch['well'].unique():
        if w not in ws:
            ws.append(w)
    sel = set(w for i, w in enumerate(ws) if i % 3 == 0)
    keep.append(ch[ch['well'].isin(sel)])
    if pd.concat([k[['well']] for k in keep])['well'].nunique() >= 260:
        break
sel = set(w for i, w in enumerate(ws) if i % 3 == 0)
df = pd.concat(keep); df = df[df['well'].isin(sel)].copy()
df['row'] = df['id'].str[9:].astype(int)
df = df.sort_values(['well', 'row']).reset_index(drop=True)
df[feats] = df[feats].astype('float32')
print('loaded %d rows %d wells %.0fs' % (len(df), df['well'].nunique(), time.time() - t0), flush=True)

wells = df['well'].unique().copy()
rng = np.random.default_rng(0); rng.shuffle(wells)
nv = len(wells) // 5; val_w = list(wells[:nv]); tr_w = list(wells[nv:])
trmask = df['well'].isin(set(tr_w)).values
mu = np.nanmean(df.loc[trmask, feats].values, 0)
sd = np.nanstd(df.loc[trmask, feats].values, 0); sd[sd < 1e-6] = 1
Xall = np.nan_to_num((df[feats].values - mu) / sd, nan=0.0).astype('float32')
yall = df['target'].values.astype('float32'); warr = df['well'].values


def seqs(wset):
    return [(Xall[np.where(warr == w)[0]], yall[np.where(warr == w)[0]]) for w in wset]


tr = seqs(tr_w); va = seqs(val_w)


class TCN(nn.Module):
    def __init__(self, nf, h=128):
        super().__init__()

        def blk(ci, co, d):
            return nn.Sequential(nn.Conv1d(ci, co, 3, padding=d, dilation=d), nn.ReLU(), nn.BatchNorm1d(co), nn.Dropout(0.1))
        self.net = nn.Sequential(blk(nf, h, 1), blk(h, h, 2), blk(h, h, 4), blk(h, h, 8), blk(h, h, 16), blk(h, h, 32))
        self.head = nn.Conv1d(h, 1, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


net = TCN(NF).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
EPOCHS = 70
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)


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


def val_eval():
    net.eval(); se = 0; n = 0; preds = []
    with torch.no_grad():
        for X, Y, M in batches(va, 8, False):
            p = net(X); se += (((p - Y) ** 2) * M).sum().item(); n += M.sum().item()
            for k in range(X.shape[0]):
                L = int(M[k].sum().item()); preds.append(p[k, :L].cpu().numpy())
    return (se / n) ** 0.5, preds


best = 9e9; best_preds = None
for ep in range(EPOCHS):
    net.train()
    for X, Y, M in batches(tr, 8, True):
        opt.zero_grad(); p = net(X); l = (((p - Y) ** 2) * M).sum() / M.sum()
        l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
    sched.step()
    if ep % 5 == 0 or ep > EPOCHS - 6:
        r, preds = val_eval()
        if r < best:
            best = r; best_preds = preds
        print('ep %02d val pooled=%.4f (best %.4f) %.0fs' % (ep, r, best, time.time() - t0), flush=True)

# fair GBDT stacker on SAME train/val wells
gbm = lgb.LGBMRegressor(device='gpu', gpu_use_dp=False, n_estimators=800, learning_rate=0.03,
                        num_leaves=95, subsample=0.8, colsample_bytree=0.6, min_child_samples=200,
                        reg_lambda=2.0, verbose=-1)
gbm.fit(Xall[trmask], yall[trmask])
vmask = df['well'].isin(set(val_w)).values
gp = gbm.predict(Xall[vmask])
gbdt_rmse = float(np.sqrt(np.mean((gp - yall[vmask]) ** 2)))

print('\n=== RESULTS (same 52 val wells) ===')
print('NEURAL TCN best val pooled = %.4f' % best)
print('GBDT stacker  val pooled  = %.4f' % gbdt_rmse)
print('(PF ~9.7, hold ~16 reference)')

# save neural val preds aligned to id for blend testing
val_ids = []
for w in val_w:
    g = df[df['well'] == w].sort_values('row')
    val_ids.append(g['id'].values)
flat_ids = np.concatenate(val_ids); flat_pred = np.concatenate(best_preds)
pd.DataFrame({'id': flat_ids, 'neural_dev': flat_pred}).to_csv('output/neural_val_preds.csv', index=False)
print('saved output/neural_val_preds.csv (%d rows)' % len(flat_ids))
