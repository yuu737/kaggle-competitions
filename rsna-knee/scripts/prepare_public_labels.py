"""Reproduce conservative weak targets from the two archived public label tables."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from mri_features import LABELS

ROOT=Path(__file__).resolve().parents[1]

def main():
    t=pd.read_csv(ROOT/'data/raw/train.csv').set_index('StudyInstanceUID')
    ap=ROOT/'data/phase2_labels/pilkwang/report_labels_v2.csv'
    bp=ROOT/'data/phase2_labels/stevenleehans/llm_labels_full.csv'
    a=pd.read_csv(ap).set_index('StudyInstanceUID').reindex(t.index)
    b=pd.read_csv(bp).set_index('StudyInstanceUID').reindex(t.index)
    out=t[[]].copy();counts={}
    for label in LABELS:
        known=t[label].isna()&a[label+'__verdict'].isin(['YES','NO'])&np.isfinite(a[label])&np.isfinite(b[label])
        positive=known&(a[label]>=.8)&(b[label]>=.8)
        negative=known&(a[label]<=.1)&(b[label]<=.1);keep=positive|negative
        out[label]=((a[label]+b[label])/2).where(keep)
        out[label+'__weight']=np.minimum(a[label+'__conf'],2*(b[label]-.5).abs()).where(keep,0)
        out[label+'__source']='pilkwang+steven_v1_consensus'
        counts[label]={'positive':int(positive.sum()),'negative':int(negative.sum()),'masked':int((~keep).sum())}
    out['label_source']='public_report_weak_consensus';out['official_complete']=t[LABELS].notna().all(axis=1)
    cols=[l+'__weight' for l in LABELS]
    assert len(out)==4407 and out.official_complete.sum()==58
    assert out.loc[out.official_complete,cols].to_numpy().sum()==0
    p=ROOT/'outputs/public_wholebrain';p.mkdir(parents=True,exist_ok=True)
    destination=p/'weak_labels.csv';out.reset_index().to_csv(destination,index=False)
    report={'counts':counts,'studies_with_supervision':int((out[cols].sum(axis=1)>0).sum()),'target_sha256':hashlib.file_digest(destination.open('rb'),'sha256').hexdigest(),'sources':[{'path':str(f.relative_to(ROOT)),'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()} for f in [ap,bp]],'official_labels_used_for_training':False,'missing_targets_are_normal':False}
    (p/'weak_label_reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
