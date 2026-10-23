from __future__ import annotations
from pathlib import Path
import json, zipfile
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'submission_7182_67.zip'
OUT=ROOT/'artifacts'/'task286_compact_frontend.onnx'
MAN=ROOT/'artifacts'/'task286_compact_frontend_manifest.json'

def arr(init): return numpy_helper.to_array(init)
def T(name,a,dtype=None): return numpy_helper.from_array(np.asarray(a,dtype=dtype),name=name)

def build():
    with zipfile.ZipFile(BASE) as z:
        base=onnx.load_model_from_string(z.read('task286.onnx'))

    # Retain nodes 17 onward exactly; determine which original initializers they reference.
    kept_nodes=[onnx.NodeProto().FromString(n.SerializeToString()) for n in base.graph.node[17:]]
    produced={o for n in kept_nodes for o in n.output if o}
    used={i for n in kept_nodes for i in n.input if i and i not in produced and i not in {'v3','v4_qlinear','v11','v14','v17'}}
    old_init={x.name:x for x in base.graph.initializer}
    inits=[onnx.TensorProto().FromString(old_init[n].SerializeToString()) for n in sorted(used) if n in old_init]
    names={x.name for x in inits}
    def add(name,a,dtype=None):
        if name not in names:
            inits.append(T(name,a,dtype)); names.add(name)
        return name

    # Reuse official scalar quantization constants when possible.
    if 'v4_scale' not in names: add('v4_scale',np.array(1,np.float32))
    if 'v4_zero' not in names: add('v4_zero',np.array(0,np.uint8))
    if 'z8' not in names: add('z8',np.array([0],np.uint8))

    # Sparse 6x6 float Conv decodes labels and crops 30x30 -> 25x25 in one node.
    lw=np.zeros((1,10,6,6),np.float32); lw[0,:,0,0]=np.arange(10,dtype=np.float32)
    add('front_label_w',lw)
    add('front_axes_row',np.array([1,3],np.int64)); add('front_axes_col',np.array([1,2],np.int64))
    add('front_row_s',np.array([0],np.int64)); add('front_row_e',np.array([25],np.int64)); add('front_row_ax',np.array([2],np.int64))
    add('front_col_s',np.array([0],np.int64)); add('front_col_e',np.array([25],np.int64)); add('front_col_ax',np.array([3],np.int64))
    add('front_zero_f',np.array(0,np.float32)); add('front_eight',np.array(8,np.uint8))
    add('front_row_shape',np.array([1,1,25],np.int64))
    add('front_S8',np.array(8,np.uint32)); add('front_S16',np.array(16,np.uint32))
    for j in range(4):
        w=np.zeros((25,1),np.uint8); lo=8*j; hi=min(25,lo+8)
        for c in range(lo,hi): w[c,0]=np.uint8(1<<(c-lo))
        add(f'front_pack_w{j}',w)

    nodes=[]; idx=0
    def node(op,ins,outs=None,**attrs):
        nonlocal idx
        if outs is None: outs=[f'front_{idx}']; idx+=1
        elif isinstance(outs,str): outs=[outs]
        nodes.append(helper.make_node(op,ins,outs,name=f'front_node_{idx}_{op}',**attrs))
        return outs[0] if len(outs)==1 else outs

    # Label and compact validity rectangle.
    lf=node('Conv',['input','front_label_w'])
    node('Cast',[lf],outs='v4_qlinear',to=TensorProto.UINT8)
    rs=node('ReduceSum',['input','front_axes_row'],keepdims=1)
    r25=node('Slice',[rs,'front_row_s','front_row_e','front_row_ax'])
    rv=node('Greater',[r25,'front_zero_f'])
    cs=node('ReduceSum',['input','front_axes_col'],keepdims=1)
    c25=node('Slice',[cs,'front_col_s','front_col_e','front_col_ax'])
    cv=node('Greater',[c25,'front_zero_f'])
    node('And',[rv,cv],outs='v3')

    wall=node('Equal',['v4_qlinear','front_eight'])
    notwall=node('Not',[wall])
    walk=node('And',['v3',notwall])
    bg=node('Equal',['v4_qlinear','z8'])
    notbg=node('Not',[bg])
    node('And',[walk,notbg],outs='v11')

    def pack(mask_name,out_name):
        u=node('Cast',[mask_name],to=TensorProto.UINT8)
        words=[]
        for j in range(4):
            q=node('QLinearMatMul',[u,'v4_scale','v4_zero',f'front_pack_w{j}','v4_scale','v4_zero','v4_scale','v4_zero'])
            q32=node('Cast',[q],to=TensorProto.UINT32)
            if j==1: q32=node('BitShift',[q32,'front_S8'],direction='LEFT')
            elif j==2: q32=node('BitShift',[q32,'front_S16'],direction='LEFT')
            elif j==3:
                q32=node('BitShift',[q32,'front_S16'],direction='LEFT')
                q32=node('BitShift',[q32,'front_S8'],direction='LEFT')
            words.append(q32)
        x=node('BitwiseOr',[words[0],words[1]])
        x=node('BitwiseOr',[x,words[2]])
        x=node('BitwiseOr',[x,words[3]])
        node('Reshape',[x,'front_row_shape'],outs=out_name)
    pack(walk,'v14')
    pack('v11','v17')

    nodes.extend(kept_nodes)
    graph=helper.make_graph(nodes,'task286_compact_frontend',
        [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
        [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    OUT.parent.mkdir(parents=True,exist_ok=True); onnx.save(model,OUT)
    MAN.write_text(json.dumps([{'task':286,'path':str(OUT),'kind':'compact_frontend'}],indent=2)+'\n')
    print(OUT,'nodes',len(nodes),'inits',len(inits))

if __name__=='__main__': build()
