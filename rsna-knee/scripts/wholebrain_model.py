"""Whole MaleCNS sparse recurrent reservoir, with explicit source-to-model mapping.

This is a machine-learning rate/message-passing model, not a validated biological
simulation. All annotated non-glial bodies are computational states. All retained
directed edges participate in each sparse matrix multiplication.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

class WholeBrain(torch.nn.Module):
    def __init__(self,root,input_dim=12288,pool_dim=2048,steps=4,device='cuda',seed=20261008):
        super().__init__();root=Path(root);self.steps=steps;self.input_dim=input_dim;self.pool_dim=pool_dim
        started=time.perf_counter()
        counts=sp.load_npz(root/'counts.npz').tocsr()
        ids=np.load(root/'body_ids.npy');n=len(ids);self.n=n
        nt=pd.read_feather(root/'neurotransmitters.feather').set_index('body').reindex(ids)
        transmitter=nt.consensus_nt.fillna('unknown')
        # Simplification: GABA and glutamate inhibitory, other/unknown NT positive.
        sign=np.where(transmitter.isin(['gaba','glutamate']),-1,1).astype(np.float32)
        incoming=np.asarray(counts.sum(1)).ravel()
        a=counts.copy();a.data*=sign[a.indices]
        a=sp.diags(1/np.maximum(incoming,1))@a;a=a.tocsr();a.sort_indices()
        self.register_buffer('adj',torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int64)),torch.from_numpy(a.indices.astype(np.int64)),torch.from_numpy(a.data.astype(np.float32)),size=a.shape,device=device))
        rng=np.random.default_rng(seed)
        idx=rng.integers(input_dim,size=(n,4),dtype=np.int64)
        projection_sign=rng.choice(np.array([-1,1],np.float32),size=(n,4))
        # Every state maps to exactly one balanced signed readout bin; no state is discarded.
        pool=rng.permutation(n)%pool_dim
        pool_sign=rng.choice(np.array([-1,1],np.float32),size=n)
        pool_count=np.bincount(pool,minlength=pool_dim)
        perm=rng.permutation(n)
        self.register_buffer('input_index',torch.from_numpy(idx).to(device))
        self.register_buffer('input_sign',torch.from_numpy(projection_sign).to(device))
        self.register_buffer('pool_index',torch.from_numpy(pool.astype(np.int64)).to(device))
        self.register_buffer('pool_sign',torch.from_numpy(pool_sign).to(device))
        self.register_buffer('pool_count',torch.from_numpy(np.sqrt(pool_count).astype(np.float32)).to(device))
        self.register_buffer('shuffle',torch.from_numpy(perm.astype(np.int64)).to(device))
        self.meta={'nodes':n,'edges':a.nnz,'steps':steps,'input_dim':input_dim,'pool_dim':pool_dim,'seed':seed,'negative_nodes':int((sign<0).sum()),'unknown_nt_nodes':int((transmitter=='unknown').sum()),'load_seconds':time.perf_counter()-started,'csr_bytes':sum(t.numel()*t.element_size() for t in [self.adj.crow_indices(),self.adj.col_indices(),self.adj.values()]),'injection':'four signed feature coordinates per node divided by 2','output':'all final node activities, signed balanced bins divided by sqrt(bin count)','recurrence':'h(t+1)=tanh(0.2*u+0.95*A*h(t)); h(0)=tanh(u)','bypass':'same injection and readout; h=tanh(u)','zero_edges':'same recurrence, sparse edge contribution set to zero','shuffle':'row permutation of the normalized directed adjacency, fixed seed; injection and readout unchanged'}
        self._mapping={'node_index':np.arange(n),'bodyId':ids,'nt':transmitter.to_numpy(),'synaptic_sign':sign,'readout_bin':pool,'readout_sign':pool_sign,'shuffle_destination_source_row':perm}
        for k in range(4):self._mapping['feature_index_'+str(k)]=idx[:,k];self._mapping['feature_sign_'+str(k)]=projection_sign[:,k]

    def save_mapping(self,path):
        pd.DataFrame(self._mapping).to_csv(path,index=False)

    def forward(self,x,mode='real',return_activity=False):
        # x is normalized per study, never fitted on validation/test examples.
        u=(x[:,self.input_index]*self.input_sign).sum(-1).T.contiguous()/2
        h=torch.tanh(u)
        if mode!='bypass':
            for _ in range(self.steps):
                message=torch.zeros_like(h) if mode=='zero' else torch.sparse.mm(self.adj,h)
                if mode=='shuffle':message=message[self.shuffle]
                h=torch.tanh(.2*u+.95*message)
        z=torch.zeros((self.pool_dim,x.shape[0]),device=x.device,dtype=x.dtype)
        z.index_add_(0,self.pool_index,h*self.pool_sign[:,None])
        z=(z/self.pool_count[:,None]).T
        return (z,h) if return_activity else z

def normalize_features(x):
    # Six 2048-d blocks, each normalized independently using only this examination.
    x=x.reshape(len(x),6,2048).astype(np.float32)
    mean=x.mean(2,keepdims=True);std=x.std(2,keepdims=True)
    return ((x-mean)/np.maximum(std,1e-6)).reshape(len(x),-1)

def smoke(root,out):
    import psutil
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261008);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();brain=WholeBrain(root,device=device)
    x=torch.randn(3,12288,device=device)
    t=time.perf_counter();z,h=brain(x,return_activity=True)
    if device=='cuda':torch.cuda.synchronize()
    forward=time.perf_counter()-t
    head=torch.nn.Linear(brain.pool_dim,12).to(device)
    y=torch.randint(0,2,(3,12),device=device).float()
    mask=torch.ones_like(y);mask[0,0]=0
    original=head.weight.detach().clone();optimizer=torch.optim.SGD(head.parameters(),lr=.1)
    logits=head(z.detach());loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*mask).sum()/mask.sum()
    optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        p=head(z).sigmoid();zero=head(brain(x,'zero')).sigmoid();shuffle=head(brain(x,'shuffle')).sigmoid()
        # Intervention on all final states: readout has nonzero derivative for every state.
        activity=h.detach().clone().requires_grad_(True)
    zz=torch.zeros((brain.pool_dim,3),device=device).index_add(0,brain.pool_index,activity*brain.pool_sign[:,None])
    pp=head((zz/brain.pool_count[:,None]).T)
    pp[:,0].sum().backward()
    result={**brain.meta,'input':'synthetic smoke only; not an MRI accuracy result','device':torch.cuda.get_device_name() if device=='cuda' else 'cpu','forward_seconds_batch3':forward,'seconds_per_study':forward/3,'total_seconds':time.perf_counter()-started,'output_shape':list(p.shape),'active_nodes':int((h.abs().sum(1)>0).sum()),'nodes_with_nonzero_output_gradient':int((activity.grad.abs().sum(1)>0).sum()),'edge_zero_max_probability_change':float((p-zero).abs().max()),'shuffle_max_probability_change':float((p-shuffle).abs().max()),'readout_update_l2':float((head.weight-original).norm()),'masked_loss':float(loss.detach()),'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0,'rss_bytes':psutil.Process().memory_info().rss}
    info=psutil.Process().memory_info();result['peak_rss_bytes']=getattr(info,'peak_wset',info.rss)
    assert result['active_nodes']==brain.n
    assert result['nodes_with_nonzero_output_gradient']==brain.n
    assert result['edge_zero_max_probability_change']>1e-6
    assert result['readout_update_l2']>0 and p.shape==(3,12) and torch.isfinite(p).all()
    brain.save_mapping(out/'circuit_mapping.csv.gz')
    (out/'smoke.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/connectome/male-cns-v1.0');p.add_argument('--out',default='outputs/wholebrain');a=p.parse_args();smoke(a.root,a.out)
