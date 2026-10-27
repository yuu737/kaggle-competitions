"""Stage 2 builder: mix OOF predictions into the soft labels and write a training script with the mixed labels embedded.
usage: python -I scripts/make_stage2.py <cv out dir> <alpha> <slug> [oof-kind: swa|epens|plain]"""
import sys, glob, base64, io, re, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
cv, alpha, slug = sys.argv[1], float(sys.argv[2]), sys.argv[3]
kind = sys.argv[4] if len(sys.argv) > 4 else 'swa'
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
pat = {'swa': '*oof_*_swa_fold*.npz', 'epens': '*oof_*_epens_fold*.npz', 'plain': '*oof_*_fold?.npz'}[kind]
fs = sorted(glob.glob(cv + '/**/' + pat, recursive=True)); print(len(fs), 'oof files')
ids, P, Y = [], [], []
for f in fs:
    z = np.load(f, allow_pickle=True); ids += list(z['ids']); P.append(z['pred']); Y.append(z['truth'])
P, Y = np.concatenate(P), np.concatenate(Y)
print('OOF studies', len(ids))
au = []
for j, c in enumerate(LAB):
    yb = (Y[:, j] >= 0.5).astype(int)
    au.append(roc_auc_score(yb, P[:, j]) if 0 < yb.sum() < len(yb) else float('nan'))
print('OOF AUC vs hard(label>=0.5):', dict(zip(LAB, np.round(au, 3))), 'mean', np.nanmean(au))
M = (1 - alpha) * Y + alpha * P
df = pd.DataFrame(M.astype('float32'), columns=LAB); df.insert(0, 'StudyInstanceUID', [str(i) for i in ids])
buf = io.BytesIO(); df.to_parquet(buf, index=False)
b64 = base64.b64encode(buf.getvalue()).decode()
src0 = open('D:/RSNA/notebooks/train_own2/rsna-knee-train-own2.py', encoding='utf-8').read()
hdr = src0[:src0.index(chr(10), src0.index('TRAINER_B64'))] + chr(10)
body = open(sys.argv[5] if len(sys.argv) > 5 else 'D:/RSNA/scripts/stage2_body.tmpl', encoding='utf-8').read().replace('__MIX_B64__', b64)
src = hdr + body
import os
d = 'D:/RSNA/notebooks/' + slug.replace('rsna-knee-', '').replace('-', '_'); os.makedirs(d, exist_ok=True)
open(d + '/%s.py' % slug, 'w', encoding='utf-8').write(src)
md = json.load(open('D:/RSNA/notebooks/train_own2/kernel-metadata.json'))
md['dataset_sources'] = md['dataset_sources'] + ['nartaa/rsna-knee-publication-swa-weights-20261007']
md.update(id='yuu111111111/' + slug, title=slug, code_file=slug + '.py')
json.dump(md, open(d + '/kernel-metadata.json', 'w'), indent=1); print('built', d)
