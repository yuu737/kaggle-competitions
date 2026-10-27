"""Whole-brain features, masked soft-label readouts, and explicit intervention controls."""
from pathlib import Path
import argparse,json,time,gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from wholebrain_model import WholeBrain
from mri_features import LABELS

MODES=['real','bypass','shuffle','zero']

def normalize(x):
    shape=x.shape
    v=x.astype(np.float32).reshape(len(x),12,-1)
    return ((v-v.mean(2,keepdims=True))/np.maximum(v.std(2,keepdims=True),1e-6)).reshape(shape)

def predict(head,x):
    z=(x-head['mean'])/head['scale']/np.sqrt(x.shape[1])
    logits=z@head['coef'].T+head['intercept']
    return 1/(1+np.exp(-np.clip(logits,-40,40)))

def load_head(file):
    with np.load(file) as z:return {k:z[k] for k in z.files}

def fit(x,y,w,regularization,device='cuda'):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-4)
    z=torch.tensor((x-mean)/scale/np.sqrt(x.shape[1]),dtype=torch.float32,device=device)
    target=torch.tensor(np.nan_to_num(y,nan=.5),dtype=torch.float32,device=device)
    weight=torch.tensor(w,dtype=torch.float32,device=device)
    denom=weight.sum(0).clamp_min(1)
    model=torch.nn.Linear(x.shape[1],12,device=device)
    with torch.no_grad():
        model.weight.zero_()
        prior=(target*weight).sum(0)/denom
        model.bias.copy_(torch.logit(prior.clamp(.01,.99)))
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=180,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
    def closure():
        opt.zero_grad()
        loss=((torch.nn.functional.binary_cross_entropy_with_logits(model(z),target,reduction='none')*weight).sum(0)/denom).mean()+regularization*.5*model.weight.square().sum()/12
        loss.backward();return loss
    initial=float(closure().detach());opt.step(closure);final=float(closure().detach())
    result={'mean':mean,'scale':scale,'coef':model.weight.detach().cpu().numpy(),'intercept':model.bias.detach().cpu().numpy(),'regularization':np.array(regularization),'initial_objective':np.array(initial),'final_objective':np.array(final)}
    assert final<=initial+1e-5
    return result

def weighted_bce(y,p,w):
    y=np.nan_to_num(y,nan=.5);p=np.clip(p,1e-6,1-1e-6)
    return float(((-(y*np.log(p)+(1-y)*np.log(1-p))*w).sum(0)/np.maximum(w.sum(0),1)).mean())

def auc(y,p):return float(np.mean([roc_auc_score(y[:,j],p[:,j]) for j in range(12)]))

def uncertainty(y,preds,base_key,n=2000):
    rng=np.random.default_rng(20261009);weights=rng.multinomial(len(y),np.ones(len(y))/len(y),n).astype(float);samples={}
    for key,p in preds.items():
        by=[]
        for j in range(12):
            pos=y[:,j]==1;neg=y[:,j]==0
            wins=(p[pos,j,None]>p[neg,j][None]).astype(float)+.5*(p[pos,j,None]==p[neg,j][None])
            denom=weights[:,pos].sum(1)*weights[:,neg].sum(1)
            value=np.einsum('bi,ij,bj->b',weights[:,pos],wins,weights[:,neg])/np.maximum(denom,1)
            value[denom==0]=np.nan;by.append(value)
        a=np.stack(by,1);a=np.where(np.isnan(a).any(1),np.nan,a.mean(1));samples[key]=a
    return {k:{'ci95':np.nanpercentile(a,[2.5,97.5]).tolist(),'difference_from_original_ci95':np.nanpercentile(a-samples[base_key],[2.5,97.5]).tolist(),'bootstrap_valid_replicates':int(np.isfinite(a).sum()),'bootstrap_requested_replicates':n} for k,a in samples.items()}

def run(features_dir,graph,out):
    import psutil
    started=time.perf_counter();torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    extraction=json.loads((features_dir/'features/extraction.json').read_text())
    assert extraction['completed']==extraction['requested'] and not extraction['failed'],'Feature extraction must finish without dropped studies'
    # Loading stored activations does not import or execute the external public encoder.
    parts=[]
    for file in sorted((features_dir/'features').glob('features_*.npz')):
        with np.load(file) as z:parts.append({k:z[k] for k in z.files})
    z={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order=np.argsort(z['ids'].astype(str));z={k:v[order] for k,v in z.items()};ids=z['ids'].astype(str)
    assert len(set(ids))==len(ids)
    weak=pd.read_csv(features_dir/'weak_labels.csv').set_index('StudyInstanceUID').reindex(ids)
    gold=pd.read_csv(features_dir/'gold_labels.csv').set_index('StudyInstanceUID')
    train=(weak[[l+'__weight' for l in LABELS]].sum(axis=1).to_numpy()>0)&(~np.isin(ids,gold.index))&(z['split']=='train')
    assert train.sum()>3000,'Large-cohort experiment requires at least 3000 actual weak-labelled MRI studies'
    assert not set(ids[train])&set(gold.index)
    expected=weak.loc[~weak.index.isin(gold.index),[l+'__weight' for l in LABELS]].sum(axis=1)>0
    assert set(ids[train])==set(expected.index[expected]),'Readout cohort must match all eligible weak-labelled studies'
    y=weak[LABELS].to_numpy(np.float32)[train];w=weak[[l+'__weight' for l in LABELS]].to_numpy(np.float32)[train]
    assert np.isfinite(w).all() and (w.sum(0)>0).all() and np.all(np.isfinite(y[w>0]))
    x=normalize(z['features']);brain=WholeBrain(graph,input_dim=x.shape[1],device=device)
    assert brain.n==199713 and brain.meta['edges']==26013498
    brain.save_mapping(out/'circuit_mapping.csv.gz')
    allfeatures={'conventional':x};checks={};times={}
    for mode in MODES:
        t=time.perf_counter();values=[]
        with torch.inference_mode():
            for i in range(0,len(x),8):
                v,h=brain(torch.from_numpy(x[i:i+8]).to(device),mode,True)
                values.append(v.cpu().numpy())
                if i==0:checks[mode]={'active_nodes':int((h.abs().sum(1)>0).sum()),'activity_rms':float(h.square().mean().sqrt())}
        allfeatures[mode]=np.concatenate(values);times[mode]=time.perf_counter()-t
        np.savez_compressed(out/(mode+'_features.npz'),ids=ids,x=allfeatures[mode])
        print('CIRCUIT',mode,len(ids),round(times[mode],2),flush=True)
    # A UID-stable 20% weak-label holdout selects only the new readout regularization.
    import hashlib
    val=np.array([int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)%5==0 for uid in ids[train]])
    assert (w[val].sum(0)>0).all() and (w[~val].sum(0)>0).all()
    train_uid_file=pd.DataFrame({'StudyInstanceUID':ids[train],'weak_validation_for_lambda':val})
    train_uid_file.to_csv(out/'training_studies.csv',index=False)
    heads=out/'heads';heads.mkdir(exist_ok=True);preds={};fits={};head_models={}
    for mode,xx in allfeatures.items():
        scores={}
        for reg in [.0001,.001,.01]:
            h=fit(xx[train][~val],y[~val],w[~val],reg,device)
            scores[reg]=weighted_bce(y[val],predict(h,xx[train][val]),w[val])
        reg=min(scores,key=scores.get);h=fit(xx[train],y,w,reg,device)
        np.savez_compressed(heads/(mode+'_head.npz'),**h)
        head_models[mode]=h;preds[mode]=predict(h,xx)
        fits[mode]={'lambda':reg,'weak_holdout_bce':scores,'fit_initial_objective':float(h['initial_objective']),'fit_final_objective':float(h['final_objective']),'parameters':int(h['coef'].size+h['intercept'].size)}
        print('READOUT',mode,fits[mode],flush=True)
    # Interventions keep the trained REAL readout fixed, avoiding the retraining confound.
    r=head_models['real'];real=preds['real'];pool=brain.pool_index.cpu().numpy()
    interventions={mode:float(np.max(np.abs(real-predict(r,allfeatures[mode])))) for mode in ['bypass','shuffle','zero']}
    interventions['silenced_activity']=float(np.max(np.abs(real-predict(r,np.zeros_like(allfeatures['real'])))))
    interventions['nodes_with_nonzero_readout_derivative']=int(np.any(r['coef'][:,pool]!=0,axis=0).sum())
    assert interventions['nodes_with_nonzero_readout_derivative']==brain.n
    assert interventions['zero']>1e-6 and interventions['shuffle']>1e-6
    protocol=json.loads((features_dir/'protocol.json').read_text());alpha=protocol['branch_blend_weight']
    result={};structure_comparisons={};from scipy.stats import rankdata
    for variant in ['apex_common','dinosaur_v30']:
        base=pd.read_csv(features_dir/('train_'+variant+'_baseline.csv')).set_index('StudyInstanceUID')
        assert set(base.index)==set(gold.index)
        sel=np.array([int(np.flatnonzero(ids==uid)[0]) for uid in base.index]);truth=gold.loc[base.index,LABELS].to_numpy(float)
        p={'original':base[LABELS].to_numpy(float)}
        for mode in allfeatures:
            prob=preds[mode][sel];p[mode+'_only']=prob
            ranked=(rankdata(prob,axis=0,method='average')-(.5 if variant=='dinosaur_v30' else 0))/len(prob)
            p['plus_'+mode]=(1-alpha)*p['original']+alpha*ranked
        ci=uncertainty(truth,p,'original')
        result[variant]={k:{'macro_auc':auc(truth,v),'per_label':{l:float(roc_auc_score(truth[:,j],v[:,j])) for j,l in enumerate(LABELS)},**ci[k]} for k,v in p.items()}
        structure_comparisons[variant]={}
        for control in ['conventional','bypass','shuffle','zero']:
            pair={'real':p['real_only'],'control':p[control+'_only']}
            u=uncertainty(truth,pair,'control')['real']
            structure_comparisons[variant]['real_minus_'+control]={'difference':auc(truth,pair['real'])-auc(truth,pair['control']),'ci95':u['difference_from_original_ci95']}
        for mode,v in p.items():
            f=pd.DataFrame(v,columns=LABELS);f.insert(0,'StudyInstanceUID',base.index);f.to_csv(out/(variant+'_'+mode+'_gold58.csv'),index=False)
        print(variant,{k:round(v['macro_auc'],5) for k,v in result[variant].items()},flush=True)
    report={'n_encoded':len(ids),'n_readout_train':int(train.sum()),'gold_n':58,'gold_not_independent':True,'reason':'Public encoder/weak-label development selected on Gold58; no new readout trained on its labels or reports. This is an exploratory diagnostic, not independent model validation.','graph':brain.meta,'activity_checks':checks,'fixed_readout_interventions':interventions,'fits':fits,'branch_weight':alpha,'results':result,'structure_comparisons':structure_comparisons,'circuit_seconds':times,'seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name() if device=='cuda' else 'cpu','peak_gpu_allocated':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'rss_bytes':psutil.Process().memory_info().rss}
    memory=psutil.Process().memory_info()
    report['gold_label_counts']={l:{'positive':int(gold[l].sum()),'negative':int(len(gold)-gold[l].sum())} for l in LABELS}
    report['bootstrap_method']='Paired examination resampling, 2000 draws; discard a draw if any of 12 labels lacks a positive or negative. Conditional on this fixed, selection-exposed cohort and fixed models.'
    report['peak_rss_bytes']=getattr(memory,'peak_wset',memory.rss)
    report['rss_measurement']='OS peak working set on Windows; current RSS on platforms without peak_wset'
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=Path('outputs/public_wholebrain/kaggle_features'));p.add_argument('--graph',type=Path,default=Path('data/connectome/male-cns-v1.0'));p.add_argument('--out',type=Path,default=Path('outputs/public_wholebrain'));a=p.parse_args()
    with threadpool_limits(limits=4):run(a.features,a.graph,a.out)
