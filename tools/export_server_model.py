"""Export a trusted private checkpoint to server ONNX; validate 20 inputs.

Export-only rewrites implement index_add with ScatterElements(add), retaining
duplicate-index accumulation, and expm1(x) as exp(x)-1. Original training code
and the PyTorch checkpoint are not modified.
"""
from pathlib import Path
import ast
import argparse
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from model_runtime import core

NAMES=['z','x_cont','x_cont_mask','src','dst','edge_vec','edge_relation','edge_bond_score','batch','descriptor']


def scatter_add(data,dim,index,source):
    expanded=index.reshape([-1]+[1]*(source.ndim-1)).expand_as(source)
    return data.scatter_add_(dim,expanded,source)


def expm1_compatible(value):
    return torch.exp(value)-1.0


class ExportRewrite(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='len' and len(node.args)==1:
            arg=node.args[0]
            if isinstance(arg,ast.Subscript) or (isinstance(arg,ast.Name) and arg.id=='descriptor'):
                return ast.copy_location(ast.Subscript(value=ast.Attribute(value=arg,attr='shape',ctx=ast.Load()),slice=ast.Constant(value=0),ctx=ast.Load()),node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='index_add_':
            return ast.copy_location(ast.Call(func=ast.Name(id='_scatter_add',ctx=ast.Load()),
                                              args=[node.func.value,*node.args],keywords=node.keywords),node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='expm1':
            return ast.copy_location(ast.Call(func=ast.Name(id='_expm1_compatible',ctx=ast.Load()),args=node.args,keywords=node.keywords),node)
        return node


def build_model(checkpoint_path):
    source=ROOT/'app'/'assets'/'15.DualGas_ALCHEMI_50K_L4_Training.py'
    # Load only a trusted research checkpoint; pickle is not sandboxed.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    wanted={'segment_softmax','GaussianRBF','vector_linear','EnrichedEquivariantAttentionBlock',
            'InvariantTokenPool','AlchemiStructureTower','DescriptorTower','PressureAdapter',
            'positive_monotonic_curve','positive_monotonic_dual_gas','DualGasAlchemiModel'}
    body=[n for n in ast.parse(source.read_text(encoding='utf-8')).body
          if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    tree=ast.fix_missing_locations(ExportRewrite().visit(ast.Module(body=body,type_ignores=[])))
    namespace={'torch':torch,'nn':nn,'F':F,'math':math,'np':np,
               'NODE_FEATURE_INDICES':core.NODE_FEATURE_INDICES,
               'CO2_TARGET_COLUMNS':core.CO2_TARGET_COLUMNS,'PRESSURES_BAR':core.PRESSURES_BAR,
               'GAS_INDEX':core.GAS_INDEX,'_scatter_add':scatter_add,'_expm1_compatible':expm1_compatible}
    exec(compile(tree,str(source),'exec'),namespace)
    model=namespace['DualGasAlchemiModel'](int(checkpoint['descriptor_dim']),SimpleNamespace(**checkpoint['config']))
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.eval()


class Wrapper(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.network=model
    def forward(self,z,x_cont,x_cont_mask,src,dst,edge_vec,edge_relation,edge_bond_score,batch,descriptor):
        inputs=dict(zip(NAMES,[z,x_cont,x_cont_mask,src,dst,edge_vec,edge_relation,edge_bond_score,batch,descriptor]))
        inputs['n_graphs']=1
        return self.network(inputs)['fusion']


def read_inputs(path):
    payload=json.loads(path.read_text(encoding='utf-8'))
    return {name:np.asarray(value['data'],dtype={'float32':np.float32,'int64':np.int64,'bool':np.bool_}[value['type']]).reshape(value['dims']) for name,value in payload.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New private file OUTSIDE this public checkout')
    args=parser.parse_args()
    dest=args.output.resolve()
    if dest.is_relative_to(ROOT) or dest.exists():
        parser.error('Output must be a NEW private file outside the public checkout')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    catalog=json.loads((ROOT/'app'/'examples'/'catalog.json').read_text(encoding='utf-8'))
    first=read_inputs(ROOT/'tools'/'fixtures'/catalog['examples'][0]['graph'])
    model=Wrapper(build_model(args.checkpoint)).eval()
    dest.parent.mkdir(parents=True,exist_ok=True)
    axes={name:{0:'nodes' if name in ['z','x_cont','x_cont_mask','batch'] else 'edges'} for name in NAMES if name!='descriptor'}
    with torch.no_grad():
        torch.onnx.export(model,tuple(torch.from_numpy(first[n]) for n in NAMES),str(dest),
                          input_names=NAMES,output_names=['uptake'],dynamic_axes=axes,
                          opset_version=18,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(dest))
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=4
    opts.log_severity_level=3
    session=ort.InferenceSession(str(dest),sess_options=opts,providers=['CPUExecutionProvider'])
    actual_names={x.name for x in session.get_inputs()}
    differences=[]
    for example in catalog['examples']:
        tensors=read_inputs(ROOT/'tools'/'fixtures'/example['graph'])
        result=session.run(None,{n:tensors[n] for n in actual_names})[0].reshape(-1)
        expected=np.asarray(example['reference_output'])
        error=float(np.max(np.abs(result-expected)))
        assert np.isfinite(result).all() and error<1e-4,(example['mof_id'],error)
        assert np.all(np.diff(result[:5])>=-1e-6) and np.all(np.diff(result[5:])>=-1e-6)
        differences.append({'mof_id':example['mof_id'],'max_absolute_difference_mmol_g':error})
        print(example['mof_id'],'ONNX diff',error,flush=True)
    report={'onnx_opset':18,'examples_validated':len(differences),
            'max_absolute_difference_mmol_g':max(d['max_absolute_difference_mmol_g'] for d in differences),
            'model_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'model_bytes':dest.stat().st_size,
            'reference':'Original frozen PyTorch model, not the export-rewritten model',
            'graph_preprocessing':'Frozen reference tensors for export validation; deployed server parses uploaded CIFs.',
            'export_rewrites':['index_add -> scatter_add with expanded indices','expm1(x) -> exp(x)-1','tensor len -> dynamic shape[0]'],
            'per_example':differences}
    (dest.parent/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Validated model',report['max_absolute_difference_mmol_g'],flush=True)


if __name__=='__main__':
    main()
