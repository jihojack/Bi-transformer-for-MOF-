# Generated from frozen app/core_inference.py; only annotations and unsupported fallback removed.
def parse_cif_file(path, assets):
    try:
        atoms = ase_read(path, format='cif')
        if len(atoms) == 0 or atoms.cell.rank != 3:
            raise ValueError('ASE returned empty/non-3D structure')
        atoms.set_pbc((True, True, True))
        atoms.wrap(eps=1e-10)
        positions = np.asarray(atoms.positions, dtype=np.float32)
        cell = np.asarray(atoms.cell.array, dtype=np.float32)
        symbols = atoms.get_chemical_symbols()
    except Exception as parse_error:
        raise ValueError('CIF 해석 실패 또는 지원하지 않는 구조: ' + str(parse_error)) from parse_error
    if abs(float(np.linalg.det(cell))) < 1e-06:
        raise ValueError('Singular unit cell')
    unknown = sorted(set(symbols) - set(assets.elements))
    if unknown:
        raise KeyError(f'Missing frozen element properties: {unknown}')
    values, masks, z = ([], [], [])
    for symbol in symbols:
        record = assets.elements[symbol]
        row = [record[name] for name in ELEMENT_FEATURE_NAMES]
        mask = [value is not None and math.isfinite(value) for value in row]
        values.append([float(value) if valid else 0.0 for value, valid in zip(row, mask)])
        masks.append(mask)
        z.append(int(record['atomic_number']))
    return (atoms, positions, cell, np.asarray(z, dtype=np.uint8), np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=np.uint8))

def build_cpu_graph(path, assets):
    atoms, positions, cell, z, x_element, x_element_mask = parse_cif_file(path, assets)
    src, dst, shifts = ase_neighbor_list('ijS', atoms, cutoff=6.0, self_interaction=False)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    edge_vec = (positions[dst] - positions[src] + np.einsum('ei,ij->ej', shifts.astype(np.float32), cell)).astype(np.float32)
    distance = np.linalg.norm(edge_vec, axis=1).astype(np.float32)
    valid = (distance > 1e-05) & (distance <= 6.001)
    src, dst, shifts, edge_vec, distance = (src[valid], dst[valid], shifts[valid], edge_vec[valid], distance[valid])
    order = np.argsort(src.astype(np.float64) * 7.0 + distance, kind='stable')
    kept = []
    count_by_src: dict[int, int] = {}
    for index in order:
        source = int(src[index])
        rank = count_by_src.get(source, 0)
        if rank < 32:
            kept.append(int(index))
        count_by_src[source] = rank + 1
    keep = np.asarray(kept, dtype=np.int64)
    src, dst, shifts, edge_vec, distance = (src[keep], dst[keep], shifts[keep], edge_vec[keep], distance[keep])
    if len(src) == 0:
        raise RuntimeError('No periodic edges')
    metal_index = ELEMENT_FEATURE_NAMES.index('metal_flag')
    covalent_index = ELEMENT_FEATURE_NAMES.index('covalent_radius_angstrom')
    is_metal = x_element[:, metal_index] > 0.5
    radius_sum = np.maximum(x_element[src, covalent_index] + x_element[dst, covalent_index], 0.5)
    ratio = distance / radius_sum
    pair_class = np.where(is_metal[src] & is_metal[dst], 3, np.where(is_metal[src] | is_metal[dst], 2, 1)).astype(np.int64)
    ratio_limit = np.where(pair_class == 2, 1.4, 1.25).astype(np.float32)
    sigmoid_input = np.clip((ratio_limit - ratio) / 0.08, -80.0, 80.0)
    bond_score = (1.0 / (1.0 + np.exp(-sigmoid_input))).astype(np.float32)
    bond_candidate = bond_score >= 0.5
    relation = np.where(bond_candidate, pair_class, 0).astype(np.uint8)
    n_nodes = len(z)
    coordination = np.zeros(n_nodes, dtype=np.float32)
    soft_coordination = np.zeros(n_nodes, dtype=np.float32)
    density_r3 = np.zeros(n_nodes, dtype=np.float32)
    density_r6 = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(coordination, src, bond_candidate.astype(np.float32))
    np.add.at(soft_coordination, src, bond_score)
    np.add.at(density_r3, src, np.exp(-np.square(distance / 3.0)))
    np.add.at(density_r6, src, np.exp(-np.square(distance / 6.0)))
    x_local = np.stack((coordination, soft_coordination, density_r3, density_r6), axis=1)
    return {'z': z, 'x_cont': np.concatenate((x_element, x_local), axis=1).astype(np.float32), 'x_cont_mask': np.concatenate((x_element_mask, np.ones_like(x_local, dtype=np.uint8)), axis=1), 'src': src.astype(np.uint16), 'dst': dst.astype(np.uint16), 'edge_vec': edge_vec.astype(np.float16), 'edge_distance': distance.astype(np.float16), 'edge_image': shifts.astype(np.int8), 'edge_relation': relation, 'edge_bond_score': bond_score.astype(np.float16)}

def safe_mean(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else 0.0

def graph_descriptors(mof_id, graph, base_row, assets):
    z = graph['z']
    x = graph['x_cont']
    relation = graph['edge_relation']
    score = graph['edge_bond_score'].astype(np.float32)
    src = graph['src'].astype(np.int64)
    dst = graph['dst'].astype(np.int64)
    is_metal = x[:, NODE_FEATURE_NAMES.index('metal_flag')] > 0.5
    soft_cn = x[:, NODE_FEATURE_NAMES.index('soft_coordination_number')]
    density_r3 = x[:, NODE_FEATURE_NAMES.index('local_density_r3')]
    density_r6 = x[:, NODE_FEATURE_NAMES.index('local_density_r6')]
    electronegativity = x[:, NODE_FEATURE_NAMES.index('electronegativity_pauling')]
    metal_indices = np.flatnonzero(is_metal)
    low_cn = np.zeros(len(z), dtype=bool)
    deficits = []
    for index in metal_indices:
        reference = assets.metal_reference.get(int(z[index]), assets.metal_fallback)
        low_cn[index] = soft_cn[index] <= reference['p10']
        deficits.append(max(reference['median'] - float(soft_cn[index]), 0.0))
    polar_atom = np.isin(z, list(POLAR_NONMETAL_Z))
    ligand_mask = ~is_metal
    ligand_edges = relation == 1
    if ligand_edges.any():
        delta = np.abs(electronegativity[src[ligand_edges]] - electronegativity[dst[ligand_edges]])
        ligand_en_difference = float(np.average(delta, weights=np.maximum(score[ligand_edges], 1e-06)))
    else:
        ligand_en_difference = 0.0
    pore_mask = density_r3 <= float(np.quantile(density_r3, 0.25))
    heteroatom = ~is_metal & (z != C_Z) & (z != H_Z)
    result = {'surface_area_m2g': float(base_row['surface_area_m2g']), 'void_fraction': float(base_row['void_fraction']), 'lcd': float(base_row['lcd']), 'pld': float(base_row['pld']), 'n_atoms_cif': len(z), 'metal_atom_fraction': safe_mean(is_metal.astype(float)), 'metal_species_count': len(set(z[is_metal].tolist())), 'open_metal_site_proxy_count': int(low_cn.sum()), 'open_metal_site_proxy_fraction': safe_mean(low_cn[metal_indices].astype(float)), 'metal_coordination_mean': safe_mean(soft_cn[metal_indices]), 'metal_coordination_deficit_mean': safe_mean(deficits), 'ligand_polar_atom_fraction': safe_mean(polar_atom[ligand_mask].astype(float)), 'ligand_bond_en_difference_mean': ligand_en_difference, 'pore_proxy_polar_atom_fraction': safe_mean(polar_atom[pore_mask].astype(float)), 'pore_proxy_heteroatom_fraction': safe_mean(heteroatom[pore_mask].astype(float)), 'pore_proxy_metal_fraction': safe_mean(is_metal[pore_mask].astype(float)), 'pore_proxy_electronegativity_mean': safe_mean(electronegativity[pore_mask]), 'local_density_r3_mean': safe_mean(density_r3), 'local_density_r3_std': float(np.std(density_r3)), 'local_density_r6_mean': safe_mean(density_r6), 'local_density_r6_std': float(np.std(density_r6)), 'spatial_only_edge_fraction': safe_mean((relation == 0).astype(float)), 'nonmetal_nonmetal_bond_fraction': safe_mean((relation == 1).astype(float)), 'metal_nonmetal_bond_fraction': safe_mean((relation == 2).astype(float)), 'metal_metal_bond_fraction': safe_mean((relation == 3).astype(float)), 'mean_edge_bond_score': safe_mean(score)}
    present_metals = set(z[is_metal].tolist())
    for metal_z in assets.metal_z_values:
        result[f'metal_type_{chemical_symbols[metal_z]}'] = int(metal_z in present_metals)
    missing = sorted(set(assets.descriptor_columns) - set(result))
    if missing:
        raise RuntimeError(f'Missing descriptors for {mof_id}: {missing}')
    return result
