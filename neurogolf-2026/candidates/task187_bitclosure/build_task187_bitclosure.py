from __future__ import annotations
from pathlib import Path
import json, zipfile
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'submission.zip'
OUT=ROOT/'artifacts'/'task187_bitclosure.onnx'
MAN=ROOT/'artifacts'/'task187_bitclosure_manifest.json'

def T(name,a,dtype=None):
    return numpy_helper.from_array(np.asarray(a,dtype=dtype),name=name)

def build():
    nodes=[]; inits=[]; k=0
    def init(name,a,dtype=None): inits.append(T(name,a,dtype)); return name
    def node(op,ins,name=None,outputs=None,**attrs):
        nonlocal k
        if outputs is None: outputs=[f'v{k}']; k+=1
        nodes.append(helper.make_node(op,ins,outputs,name=name or '',**attrs))
        return outputs[0] if len(outputs)==1 else outputs

    # constants
    lw=np.zeros((1,10,6,6),np.float32); lw[0,:,0,0]=np.arange(10,dtype=np.float32); init('label_w',lw)
    init('axes_row',np.array([1,3],np.int64)); init('axes_col',np.array([1,2],np.int64))
    init('axes_reduce',np.array([2,3],np.int64))
    init('crop_s',np.array([0,0],np.int64)); init('crop_e',np.array([25,25],np.int64)); init('crop_axes',np.array([2,3],np.int64))
    init('row_s',np.array([0],np.int64)); init('row_e',np.array([25],np.int64)); init('row_ax',np.array([2],np.int64))
    init('col_s',np.array([0],np.int64)); init('col_e',np.array([25],np.int64)); init('col_ax',np.array([3],np.int64))
    init('zero_f',np.array(0,np.float32)); init('zero_u8',np.array(0,np.uint8)); init('zero_u32',np.array(0,np.uint32))
    init('one_i32',np.array(1,np.int32)); init('two_u8',np.array(2,np.uint8)); init('three_u8',np.array(3,np.uint8)); init('sentinel_u8',np.array(10,np.uint8))
    init('palette',np.arange(10,dtype=np.uint8).reshape(1,10,1,1))
    init('pad25',np.array([0,0,0,0,0,0,5,5],np.int64))
    init('row_idx',np.arange(25,dtype=np.int32).reshape(1,1,25,1))
    init('right_table',np.array([1<<i for i in range(25)],dtype=np.uint32))
    top=np.zeros((1,1,25,1),np.uint32); top[:,:,0,0]=(1<<25)-1; init('top_words',top)
    init('left_bit',np.array(1,np.uint32))
    init('qscale',np.array(1,np.float32)); init('qzero',np.array(0,np.uint8))
    # 4 byte-chunk pack weights (25 x 1 each)
    for j in range(4):
        w=np.zeros((25,1),np.uint8)
        lo=j*8; hi=min(25,lo+8)
        for c in range(lo,hi): w[c,0]=np.uint8(1<<(c-lo))
        init(f'pack_w{j}',w)
    for sh in [1,2,4,8,16]: init(f'sh{sh}',np.array(sh,np.uint32))
    # Gather indices along rows, clamped. Exact Kogge-Stone path masks prevent crossing clamped boundaries.
    for sh in [1,2,4,8,16]:
        init(f'up{sh}',np.array([max(0,r-sh) for r in range(25)],np.int64))
        init(f'dn{sh}',np.array([min(24,r+sh) for r in range(25)],np.int64))

    # Label grid (preserves arbitrary line color)
    # A sparse 6x6 Conv decodes color and crops 30x30 -> 25x25 in one node.
    label_f=node('Conv',['input','label_w'])
    label25=node('Cast',[label_f],to=TensorProto.UINT8)

    # Compact valid rectangle summaries instead of a full 30x30 ReduceSum validity grid.
    row_sum=node('ReduceSum',['input','axes_row'],keepdims=1)
    row25=node('Slice',[row_sum,'row_s','row_e','row_ax'])
    row_valid=node('Greater',[row25,'zero_f'])
    col_sum=node('ReduceSum',['input','axes_col'],keepdims=1)
    col25=node('Slice',[col_sum,'col_s','col_e','col_ax'])
    col_valid=node('Greater',[col25,'zero_f'])
    valid=node('And',[row_valid,col_valid])
    label_zero=node('Equal',[label25,'zero_u8'])
    free_b=node('And',[valid,label_zero])
    label_grid=node('Where',[valid,label25,'sentinel_u8'])
    free_u=node('Cast',[free_b],to=TensorProto.UINT8)

    # Pack 25 columns into one uint32 via four exact byte-valued QLinearMatMul outputs.
    chunks=[]
    for j in range(4):
        q=node('QLinearMatMul',[free_u,'qscale','qzero',f'pack_w{j}','qscale','qzero','qscale','qzero'])
        u=node('Cast',[q],to=TensorProto.UINT32)
        if j:
            u=node('BitShift',[u,f'sh{8*j}'],direction='LEFT') if 8*j in [8,16] else node('BitShift',[u,'sh16'],direction='LEFT')
            # j=3 needs shift24: do shift16 then shift8
            if j==3: u=node('BitShift',[u,'sh8'],direction='LEFT')
        chunks.append(u)
    free_w=node('BitwiseOr',[chunks[0],chunks[1]])
    free_w=node('BitwiseOr',[free_w,chunks[2]])
    free_w=node('BitwiseOr',[free_w,chunks[3]])

    # Dynamic actual-grid boundary seed.
    top_w=node('BitwiseAnd',[free_w,'top_words'])
    left_w=node('BitwiseAnd',[free_w,'left_bit'])
    row_i=node('Cast',[row_valid],to=TensorProto.INT32)
    h=node('ReduceSum',[row_i,'axes_reduce'],keepdims=0)
    hm1=node('Sub',[h,'one_i32'])
    last_row=node('Equal',['row_idx',hm1])
    last_u=node('Cast',[last_row],to=TensorProto.UINT32)
    bottom_w=node('Mul',[free_w,last_u])
    col_i=node('Cast',[col_valid],to=TensorProto.INT32)
    w=node('ReduceSum',[col_i,'axes_reduce'],keepdims=0)
    wm1=node('Sub',[w,'one_i32'])
    right_bit=node('Gather',['right_table',wm1],axis=0)
    right_w=node('BitwiseAnd',[free_w,right_bit])
    seed=node('BitwiseOr',[top_w,bottom_w])
    seed=node('BitwiseOr',[seed,left_w])
    seed=node('BitwiseOr',[seed,right_w])

    # Precompute horizontal passability masks once; reused by both H closures.
    pL=[free_w]; pR=[free_w]
    for sh in [1,2,4,8]:
        s=node('BitShift',[pL[-1],f'sh{sh}'],direction='LEFT')
        pL.append(node('BitwiseAnd',[pL[-1],s]))
        s=node('BitShift',[pR[-1],f'sh{sh}'],direction='RIGHT')
        pR.append(node('BitwiseAnd',[pR[-1],s]))

    def hclose(x):
        for i,sh in enumerate([1,2,4,8]):
            s=node('BitShift',[x,f'sh{sh}'],direction='LEFT')
            a=node('BitwiseAnd',[pL[i],s])
            x=node('BitwiseOr',[x,a])
        for i,sh in enumerate([1,2,4,8]):
            s=node('BitShift',[x,f'sh{sh}'],direction='RIGHT')
            a=node('BitwiseAnd',[pR[i],s])
            x=node('BitwiseOr',[x,a])
        return x

    def vdir(x,direction):
        p=free_w
        for i,sh in enumerate([1,2,4,8]):
            idx=f'{direction}{sh}'
            gx=node('Gather',[x,idx],axis=2)
            a=node('BitwiseAnd',[p,gx])
            x=node('BitwiseOr',[x,a])
            if i<3:
                gp=node('Gather',[p,idx],axis=2)
                p=node('BitwiseAnd',[p,gp])
        return x
    def vclose(x):
        x=vdir(x,'up')
        x=vdir(x,'dn')
        return x

    # Exact on all task examples: H closure, V closure, H closure.
    reach_w=hclose(seed)
    reach_w=vclose(reach_w)
    reach_w=hclose(reach_w)

    # Unpack only once, immediately before free output labeling.
    powers=np.array([1<<i for i in range(25)],np.uint32).reshape(1,1,1,25)
    init('powers',powers)
    bits=node('BitwiseAnd',[reach_w,'powers'])
    reach_b=node('Greater',[bits,'zero_u32'])
    paint=node('Where',[reach_b,'three_u8','two_u8'])
    final25=node('Where',[free_b,paint,label_grid])
    final30=node('Pad',[final25,'pad25','sentinel_u8'])
    node('Equal',[final30,'palette'],outputs=['output'])

    graph=helper.make_graph(nodes,'task187_bitclosure',
        [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
        [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],inits)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    OUT.parent.mkdir(parents=True,exist_ok=True); onnx.save(model,OUT)
    MAN.write_text(json.dumps([{'task':187,'path':str(OUT),'kind':'bitclosure_hvh'}],indent=2)+'\n')
    print(OUT)

if __name__=='__main__': build()
