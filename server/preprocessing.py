"""ASE preprocessing with frozen numeric functions and preprocessing statistics."""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import warnings

import numpy as np
from ase.data import atomic_numbers,chemical_symbols
from ase.io import read as read_cif
from ase.io.cif import parse_cif
from ase.neighborlist import neighbor_list as ase_neighbor_list

BASE=Path(__file__).resolve().parent
CONFIG=json.loads((BASE/'frozen_assets.json').read_text(encoding='utf-8'))
ELEMENT_FEATURE_NAMES=['atomic_number','electronegativity_pauling','atomic_mass_da','period','group','metal_flag','valence_electrons','polarizability_bohr3','vdw_radius_angstrom','covalent_radius_angstrom']
NODE_FEATURE_NAMES=ELEMENT_FEATURE_NAMES+['coordination_number','soft_coordination_number','local_density_r3','local_density_r6']
NODE_FEATURE_INDICES=np.arange(1,14,dtype=np.int64)
POLAR_NONMETAL_Z={atomic_numbers[s] for s in ('N','O','F','P','S','Cl','Br','I')}
C_Z,H_Z=atomic_numbers['C'],atomic_numbers['H']
warnings.filterwarnings('ignore',message="crystal system 'triclinic' is not interpreted")
metal=CONFIG['metal_reference']
ASSETS=SimpleNamespace(elements=CONFIG['elements'],metal_reference={int(k):v for k,v in metal['by_atomic_number'].items()},metal_fallback=metal['fallback_all_training_metals'],
                       metal_z_values=sorted(int(v['atomic_number']) for v in CONFIG['elements'].values() if v['metal_flag']>.5),descriptor_columns=CONFIG['descriptor_columns'])


def validated_atoms(path,max_atoms):
    blocks=[b for b in parse_cif(str(path)) if b.has_structure()]
    if len(blocks)!=1:raise ValueError('원자 구조가 한 개 들어 있는 CIF만 지원합니다.')
    b=blocks[0]
    occupation=b.get('_atom_site_occupancy')
    if occupation is not None and any(not isinstance(x,(int,float)) or abs(float(x)-1)>1e-6 for x in occupation):
        raise ValueError('부분 점유율·혼합 점유 구조는 지원하지 않습니다.')
    rows=len(b.get('_atom_site_label',b.get('_atom_site_type_symbol',[])))
    if rows>max_atoms:raise ValueError(f'이 서버는 최대 {max_atoms}개 원자를 지원합니다.')
    # Guard against a pathological symmetry expansion before ASE allocates it.
    operations=b.get('_space_group_symop_operation_xyz',b.get('_symmetry_equiv_pos_as_xyz',['x,y,z']))
    if rows*len(operations)>max(10000,max_atoms*20):raise ValueError('대칭 확장 규모가 너무 큽니다.')
    atoms=read_cif(str(path),format='cif')
    if not 0<len(atoms)<=max_atoms:raise ValueError(f'대칭 확장 후 원자가 1~{max_atoms}개인 구조만 지원합니다.')
    cell=atoms.cell.array
    if atoms.cell.rank!=3 or not np.isfinite(cell).all() or not np.isfinite(atoms.positions).all():raise ValueError('3차원 격자와 유한한 좌표가 필요합니다.')
    volume=abs(float(np.linalg.det(cell)))
    if volume<1:raise ValueError('단위격자 부피가 너무 작습니다.')
    heights=1/np.linalg.norm(np.linalg.inv(cell),axis=0)
    if min(heights)<1 or len(atoms)**2*(4*math.pi*6**3/3)/volume>500000:raise ValueError('너무 작거나 조밀한 격자입니다.')
    unknown=sorted(set(atoms.get_chemical_symbols())-set(CONFIG['elements']))
    if unknown:raise ValueError('학습 시 동결된 원소 표에 없는 원소: '+', '.join(unknown))
    return atoms


def prepare(path,geometry,max_atoms=200):
    atoms=validated_atoms(path,max_atoms)
    # A per-call namespace prevents cross-request mutation of parser state.
    namespace={'np':np,'math':math,'chemical_symbols':chemical_symbols,'ELEMENT_FEATURE_NAMES':ELEMENT_FEATURE_NAMES,
               'NODE_FEATURE_NAMES':NODE_FEATURE_NAMES,'POLAR_NONMETAL_Z':POLAR_NONMETAL_Z,'C_Z':C_Z,'H_Z':H_Z,
               'ase_neighbor_list':ase_neighbor_list,'ase_read':lambda *args,**kwargs:atoms}
    exec(compile((BASE/'frozen_functions.py').read_text(encoding='utf-8'),'frozen_functions.py','exec'),namespace)
    graph=namespace['build_cpu_graph'](path,ASSETS)
    raw=namespace['graph_descriptors']('request',graph,geometry,ASSETS)
    x=graph['x_cont'][:,NODE_FEATURE_INDICES].astype(np.float32)
    mask=graph['x_cont_mask'][:,NODE_FEATURE_INDICES].astype(bool)
    x=np.where(mask,(x-np.asarray(CONFIG['node_mean'],dtype=np.float32))/np.asarray(CONFIG['node_std'],dtype=np.float32),0).astype(np.float32)
    values=np.asarray([raw[k] for k in CONFIG['descriptor_columns']],dtype=np.float64)
    values=np.where(np.isfinite(values),values,CONFIG['descriptor_median'])
    descriptor=((values-CONFIG['descriptor_center'])/CONFIG['descriptor_scale']).astype(np.float32)[None,:]
    tensors={'z':graph['z'].astype(np.int64),'x_cont':x,'x_cont_mask':mask,'src':graph['src'].astype(np.int64),'dst':graph['dst'].astype(np.int64),
             'edge_vec':graph['edge_vec'].astype(np.float32),'edge_relation':graph['edge_relation'].astype(np.int64),
             'edge_bond_score':graph['edge_bond_score'].astype(np.float32),'batch':np.zeros(len(x),dtype=np.int64),'descriptor':descriptor}
    if not all(np.isfinite(a).all() for a in tensors.values()):raise ValueError('구조 전처리 결과에 유효하지 않은 수치가 있습니다.')
    return tensors,{'atoms':len(x),'periodic_edges':len(graph['src']),'elements':sorted({chemical_symbols[int(z)] for z in graph['z']})}
