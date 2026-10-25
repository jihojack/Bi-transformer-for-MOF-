#!/usr/bin/env python3
"""CPU validation and leakage-free external inference for the 50K MOF model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.data import atomic_numbers, chemical_symbols
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list as ase_neighbor_list
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings(
    "ignore",
    message="crystal system 'triclinic' is not interpreted",
)
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator .* from version",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_001 = REPO_ROOT / "data" / "processed"
ROOT_002 = ROOT_001
RESULTS = REPO_ROOT / "app" / "assets"
EXTERNAL_DIR = REPO_ROOT / "data" / "external_inputs"

CHECKPOINT = RESULTS / "best_dual_gas_alchemi_50k.pt"
DESCRIPTOR_PREPROCESSOR = RESULTS / "descriptor_preprocessing.joblib"
NODE_PREPROCESSOR = RESULTS / "node_feature_preprocessing.npz"
ELEMENT_TABLE = RESULTS / "element_property_table_50k.json"
METAL_REFERENCE = RESULTS / "metal_soft_cn_reference.json"
TRAINING_SOURCE = REPO_ROOT / "pipeline" / "15.DualGas_ALCHEMI_50K_L4_Training.py"
ORIGINAL_H5 = ROOT_001 / "periodic_graphs_50k_alchemi_compact.h5"
ORIGINAL_CSV = ROOT_002 / "MOF_CO2_CH4_50K_alchemi_enriched.csv"
ORIGINAL_CIF_ZIP = ROOT_002 / "MOF_CO2_CH4_50K_CIF.zip"
SAVED_TEST_PREDICTIONS = ROOT_001 / "independent_test_true_predictions.csv"
EXTERNAL_INPUTS = EXTERNAL_DIR / "external_candidate_pool_inputs.csv"
EXTERNAL_LABELS = EXTERNAL_DIR / "external_candidate_pool_labels_private.csv"
EXTERNAL_CIFS = EXTERNAL_DIR / "external_candidate_pool_cifs.zip"
EXTERNAL_PREDICTIONS = EXTERNAL_DIR / "external_10k_predictions.csv"
EXTERNAL_METRICS = EXTERNAL_DIR / "external_10k_metrics.json"
EXTERNAL_INFERENCE_MANIFEST = (
    EXTERNAL_DIR / "external_10k_inference_manifest.json"
)

ELEMENT_FEATURE_NAMES = [
    "atomic_number",
    "electronegativity_pauling",
    "atomic_mass_da",
    "period",
    "group",
    "metal_flag",
    "valence_electrons",
    "polarizability_bohr3",
    "vdw_radius_angstrom",
    "covalent_radius_angstrom",
]
LOCAL_FEATURE_NAMES = [
    "coordination_number",
    "soft_coordination_number",
    "local_density_r3",
    "local_density_r6",
]
NODE_FEATURE_NAMES = ELEMENT_FEATURE_NAMES + LOCAL_FEATURE_NAMES
NODE_FEATURE_INDICES = np.arange(1, len(NODE_FEATURE_NAMES), dtype=np.int64)
CO2_TARGET_COLUMNS = [
    "co2_q_0p01bar",
    "co2_q_0p05bar",
    "co2_adsorption_0.1bar",
    "co2_adsorption_0.5bar",
    "co2_adsorption_2.5bar",
]
CH4_TARGET_COLUMNS = [
    "ch4_adsorption_0.05bar",
    "ch4_adsorption_0.5bar",
    "ch4_adsorption_0.9bar",
    "ch4_adsorption_2.5bar",
    "ch4_adsorption_4.5bar",
]
TARGET_COLUMNS = CO2_TARGET_COLUMNS + CH4_TARGET_COLUMNS
PREDICTION_COLUMNS = [f"pred_{column}" for column in TARGET_COLUMNS]
PRESSURES_BAR = np.asarray(
    [0.01, 0.05, 0.1, 0.5, 2.5, 0.05, 0.5, 0.9, 2.5, 4.5],
    dtype=np.float32,
)
GAS_INDEX = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
POLAR_NONMETAL_Z = {
    atomic_numbers[symbol] for symbol in ("N", "O", "F", "P", "S", "Cl", "Br", "I")
}
C_Z = atomic_numbers["C"]
H_Z = atomic_numbers["H"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-original",
        type=int,
        default=0,
        help="Reproduce this many saved original-test predictions and CPU graphs.",
    )
    parser.add_argument(
        "--run-external",
        type=int,
        default=0,
        help="Predict this many external structures; -1 means the full 10,000.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Open private labels and evaluate completed external predictions.",
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-batch-nodes", type=int, default=600)
    parser.add_argument("--max-batch-edges", type=int, default=18_000)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--assets-dir", type=Path, default=RESULTS,
                        help="Frozen checkpoint and preprocessing assets directory.")
    parser.add_argument("--original-dir", type=Path, default=ROOT_001,
                        help="Original CSV, CIF ZIP, HDF5 and optional validation predictions.")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR,
                        help="External inputs, CIF ZIP and separately stored labels.")
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global CHECKPOINT, DESCRIPTOR_PREPROCESSOR, NODE_PREPROCESSOR
    global ELEMENT_TABLE, METAL_REFERENCE, ORIGINAL_H5, ORIGINAL_CSV
    global ORIGINAL_CIF_ZIP, SAVED_TEST_PREDICTIONS, EXTERNAL_DIR
    global EXTERNAL_INPUTS, EXTERNAL_LABELS, EXTERNAL_CIFS, EXTERNAL_PREDICTIONS
    global EXTERNAL_METRICS, EXTERNAL_INFERENCE_MANIFEST
    assets = args.assets_dir.resolve()
    original = args.original_dir.resolve()
    EXTERNAL_DIR = args.external_dir.resolve()
    CHECKPOINT = assets / "best_dual_gas_alchemi_50k.pt"
    DESCRIPTOR_PREPROCESSOR = assets / "descriptor_preprocessing.joblib"
    NODE_PREPROCESSOR = assets / "node_feature_preprocessing.npz"
    ELEMENT_TABLE = assets / "element_property_table_50k.json"
    METAL_REFERENCE = assets / "metal_soft_cn_reference.json"
    ORIGINAL_H5 = original / "periodic_graphs_50k_alchemi_compact.h5"
    ORIGINAL_CSV = original / "MOF_CO2_CH4_50K_alchemi_enriched.csv"
    ORIGINAL_CIF_ZIP = original / "MOF_CO2_CH4_50K_CIF.zip"
    SAVED_TEST_PREDICTIONS = original / "independent_test_true_predictions.csv"
    EXTERNAL_INPUTS = EXTERNAL_DIR / "external_candidate_pool_inputs.csv"
    EXTERNAL_LABELS = EXTERNAL_DIR / "external_candidate_pool_labels_private.csv"
    EXTERNAL_CIFS = EXTERNAL_DIR / "external_candidate_pool_cifs.zip"
    EXTERNAL_PREDICTIONS = EXTERNAL_DIR / "external_10k_predictions.csv"
    EXTERNAL_METRICS = EXTERNAL_DIR / "external_10k_metrics.json"
    EXTERNAL_INFERENCE_MANIFEST = EXTERNAL_DIR / "external_10k_inference_manifest.json"


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(map(str, missing)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_definitions(checkpoint: dict):
    wanted = {
        "segment_softmax",
        "GaussianRBF",
        "vector_linear",
        "EnrichedEquivariantAttentionBlock",
        "InvariantTokenPool",
        "AlchemiStructureTower",
        "DescriptorTower",
        "PressureAdapter",
        "positive_monotonic_curve",
        "positive_monotonic_dual_gas",
        "DualGasAlchemiModel",
    }
    tree = ast.parse(TRAINING_SOURCE.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    found = {node.name for node in selected}
    if found != wanted:
        raise RuntimeError(f"Missing model definitions: {sorted(wanted - found)}")
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "np": np,
        "NODE_FEATURE_INDICES": NODE_FEATURE_INDICES,
        "CO2_TARGET_COLUMNS": CO2_TARGET_COLUMNS,
        "PRESSURES_BAR": PRESSURES_BAR,
        "GAS_INDEX": GAS_INDEX,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAINING_SOURCE), "exec"), namespace)
    config = SimpleNamespace(**checkpoint["config"])
    model = namespace["DualGasAlchemiModel"](
        int(checkpoint["descriptor_dim"]), config
    )
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    model.eval()
    return model


class FrozenAssets:
    def __init__(self):
        require_files(
            [
                CHECKPOINT,
                DESCRIPTOR_PREPROCESSOR,
                NODE_PREPROCESSOR,
                ELEMENT_TABLE,
                METAL_REFERENCE,
                TRAINING_SOURCE,
            ]
        )
        self.checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=False
        )
        self.model = load_model_definitions(self.checkpoint)
        descriptor_payload = joblib.load(DESCRIPTOR_PREPROCESSOR)
        self.descriptor_pipeline = descriptor_payload["pipeline"]
        self.descriptor_columns = list(descriptor_payload["feature_columns"])
        if self.descriptor_columns != list(self.checkpoint["descriptor_columns"]):
            raise RuntimeError("Checkpoint and descriptor preprocessor columns differ")
        node_payload = np.load(NODE_PREPROCESSOR, allow_pickle=False)
        self.node_mean = node_payload["mean"].astype(np.float32)
        self.node_std = node_payload["std"].astype(np.float32)
        self.node_names = node_payload["names"].astype(str).tolist()
        if self.node_names != list(self.checkpoint["node_feature_names"]):
            raise RuntimeError("Checkpoint and node preprocessor features differ")
        self.elements = json.loads(ELEMENT_TABLE.read_text(encoding="utf-8"))
        metal_payload = json.loads(METAL_REFERENCE.read_text(encoding="utf-8"))
        self.metal_reference = {
            int(key): value
            for key, value in metal_payload["by_atomic_number"].items()
        }
        self.metal_fallback = metal_payload["fallback_all_training_metals"]
        self.metal_z_values = sorted(
            int(value["atomic_number"])
            for value in self.elements.values()
            if value["metal_flag"] > 0.5
        )


def parse_cif_file(path: Path, assets: FrozenAssets):
    try:
        atoms = ase_read(path, format="cif")
        if len(atoms) == 0 or atoms.cell.rank != 3:
            raise ValueError("ASE returned empty/non-3D structure")
        atoms.set_pbc((True, True, True))
        atoms.wrap(eps=1.0e-10)
        positions = np.asarray(atoms.positions, dtype=np.float32)
        cell = np.asarray(atoms.cell.array, dtype=np.float32)
        symbols = atoms.get_chemical_symbols()
    except Exception:
        structure = Structure.from_file(path, primitive=False)
        positions = np.asarray(structure.cart_coords, dtype=np.float32)
        cell = np.asarray(structure.lattice.matrix, dtype=np.float32)
        symbols = []
        for site in structure:
            species = max(site.species.items(), key=lambda item: float(item[1]))[0]
            symbols.append(species.symbol)
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if abs(float(np.linalg.det(cell))) < 1.0e-6:
        raise ValueError("Singular unit cell")
    unknown = sorted(set(symbols) - set(assets.elements))
    if unknown:
        raise KeyError(f"Missing frozen element properties: {unknown}")
    values, masks, z = [], [], []
    for symbol in symbols:
        record = assets.elements[symbol]
        row = [record[name] for name in ELEMENT_FEATURE_NAMES]
        mask = [value is not None and math.isfinite(value) for value in row]
        values.append(
            [float(value) if valid else 0.0 for value, valid in zip(row, mask)]
        )
        masks.append(mask)
        z.append(int(record["atomic_number"]))
    return (
        atoms,
        positions,
        cell,
        np.asarray(z, dtype=np.uint8),
        np.asarray(values, dtype=np.float32),
        np.asarray(masks, dtype=np.uint8),
    )


def build_cpu_graph(path: Path, assets: FrozenAssets) -> dict:
    atoms, positions, cell, z, x_element, x_element_mask = parse_cif_file(
        path, assets
    )
    src, dst, shifts = ase_neighbor_list(
        "ijS", atoms, cutoff=6.0, self_interaction=False
    )
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    edge_vec = (
        positions[dst]
        - positions[src]
        + np.einsum("ei,ij->ej", shifts.astype(np.float32), cell)
    ).astype(np.float32)
    distance = np.linalg.norm(edge_vec, axis=1).astype(np.float32)
    valid = (distance > 1.0e-5) & (distance <= 6.001)
    src, dst, shifts, edge_vec, distance = (
        src[valid],
        dst[valid],
        shifts[valid],
        edge_vec[valid],
        distance[valid],
    )
    order = np.argsort(src.astype(np.float64) * 7.0 + distance, kind="stable")
    kept = []
    count_by_src: dict[int, int] = {}
    for index in order:
        source = int(src[index])
        rank = count_by_src.get(source, 0)
        if rank < 32:
            kept.append(int(index))
        count_by_src[source] = rank + 1
    keep = np.asarray(kept, dtype=np.int64)
    src, dst, shifts, edge_vec, distance = (
        src[keep],
        dst[keep],
        shifts[keep],
        edge_vec[keep],
        distance[keep],
    )
    if len(src) == 0:
        raise RuntimeError("No periodic edges")

    metal_index = ELEMENT_FEATURE_NAMES.index("metal_flag")
    covalent_index = ELEMENT_FEATURE_NAMES.index("covalent_radius_angstrom")
    is_metal = x_element[:, metal_index] > 0.5
    radius_sum = np.maximum(
        x_element[src, covalent_index] + x_element[dst, covalent_index], 0.5
    )
    ratio = distance / radius_sum
    pair_class = np.where(
        is_metal[src] & is_metal[dst],
        3,
        np.where(is_metal[src] | is_metal[dst], 2, 1),
    ).astype(np.int64)
    ratio_limit = np.where(pair_class == 2, 1.40, 1.25).astype(np.float32)
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
    x_local = np.stack(
        (coordination, soft_coordination, density_r3, density_r6), axis=1
    )
    return {
        "z": z,
        "x_cont": np.concatenate((x_element, x_local), axis=1).astype(np.float32),
        "x_cont_mask": np.concatenate(
            (x_element_mask, np.ones_like(x_local, dtype=np.uint8)), axis=1
        ),
        "src": src.astype(np.uint16),
        "dst": dst.astype(np.uint16),
        "edge_vec": edge_vec.astype(np.float16),
        "edge_distance": distance.astype(np.float16),
        "edge_image": shifts.astype(np.int8),
        "edge_relation": relation,
        "edge_bond_score": bond_score.astype(np.float16),
    }


def safe_mean(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else 0.0


def graph_descriptors(
    mof_id: str, graph: dict, base_row: pd.Series, assets: FrozenAssets
) -> dict:
    z = graph["z"]
    x = graph["x_cont"]
    relation = graph["edge_relation"]
    score = graph["edge_bond_score"].astype(np.float32)
    src = graph["src"].astype(np.int64)
    dst = graph["dst"].astype(np.int64)
    is_metal = x[:, NODE_FEATURE_NAMES.index("metal_flag")] > 0.5
    soft_cn = x[:, NODE_FEATURE_NAMES.index("soft_coordination_number")]
    density_r3 = x[:, NODE_FEATURE_NAMES.index("local_density_r3")]
    density_r6 = x[:, NODE_FEATURE_NAMES.index("local_density_r6")]
    electronegativity = x[
        :, NODE_FEATURE_NAMES.index("electronegativity_pauling")
    ]
    metal_indices = np.flatnonzero(is_metal)
    low_cn = np.zeros(len(z), dtype=bool)
    deficits = []
    for index in metal_indices:
        reference = assets.metal_reference.get(
            int(z[index]), assets.metal_fallback
        )
        low_cn[index] = soft_cn[index] <= reference["p10"]
        deficits.append(max(reference["median"] - float(soft_cn[index]), 0.0))
    polar_atom = np.isin(z, list(POLAR_NONMETAL_Z))
    ligand_mask = ~is_metal
    ligand_edges = relation == 1
    if ligand_edges.any():
        delta = np.abs(
            electronegativity[src[ligand_edges]]
            - electronegativity[dst[ligand_edges]]
        )
        ligand_en_difference = float(
            np.average(delta, weights=np.maximum(score[ligand_edges], 1.0e-6))
        )
    else:
        ligand_en_difference = 0.0
    pore_mask = density_r3 <= float(np.quantile(density_r3, 0.25))
    heteroatom = (~is_metal) & (z != C_Z) & (z != H_Z)
    result = {
        "surface_area_m2g": float(base_row["surface_area_m2g"]),
        "void_fraction": float(base_row["void_fraction"]),
        "lcd": float(base_row["lcd"]),
        "pld": float(base_row["pld"]),
        "n_atoms_cif": len(z),
        "metal_atom_fraction": safe_mean(is_metal.astype(float)),
        "metal_species_count": len(set(z[is_metal].tolist())),
        "open_metal_site_proxy_count": int(low_cn.sum()),
        "open_metal_site_proxy_fraction": safe_mean(
            low_cn[metal_indices].astype(float)
        ),
        "metal_coordination_mean": safe_mean(soft_cn[metal_indices]),
        "metal_coordination_deficit_mean": safe_mean(deficits),
        "ligand_polar_atom_fraction": safe_mean(
            polar_atom[ligand_mask].astype(float)
        ),
        "ligand_bond_en_difference_mean": ligand_en_difference,
        "pore_proxy_polar_atom_fraction": safe_mean(
            polar_atom[pore_mask].astype(float)
        ),
        "pore_proxy_heteroatom_fraction": safe_mean(
            heteroatom[pore_mask].astype(float)
        ),
        "pore_proxy_metal_fraction": safe_mean(
            is_metal[pore_mask].astype(float)
        ),
        "pore_proxy_electronegativity_mean": safe_mean(
            electronegativity[pore_mask]
        ),
        "local_density_r3_mean": safe_mean(density_r3),
        "local_density_r3_std": float(np.std(density_r3)),
        "local_density_r6_mean": safe_mean(density_r6),
        "local_density_r6_std": float(np.std(density_r6)),
        "spatial_only_edge_fraction": safe_mean((relation == 0).astype(float)),
        "nonmetal_nonmetal_bond_fraction": safe_mean(
            (relation == 1).astype(float)
        ),
        "metal_nonmetal_bond_fraction": safe_mean(
            (relation == 2).astype(float)
        ),
        "metal_metal_bond_fraction": safe_mean((relation == 3).astype(float)),
        "mean_edge_bond_score": safe_mean(score),
    }
    present_metals = set(z[is_metal].tolist())
    for metal_z in assets.metal_z_values:
        result[f"metal_type_{chemical_symbols[metal_z]}"] = int(
            metal_z in present_metals
        )
    missing = sorted(set(assets.descriptor_columns) - set(result))
    if missing:
        raise RuntimeError(f"Missing descriptors for {mof_id}: {missing}")
    return result


def prepare_sample(
    mof_id: str,
    graph: dict,
    descriptor: dict,
    assets: FrozenAssets,
    graph_index: int,
) -> dict:
    values = graph["x_cont"][:, NODE_FEATURE_INDICES].astype(np.float32)
    mask = graph["x_cont_mask"][:, NODE_FEATURE_INDICES].astype(bool)
    normalized = np.where(
        mask, (values - assets.node_mean) / assets.node_std, 0.0
    ).astype(np.float32)
    descriptor_frame = pd.DataFrame(
        [{name: descriptor[name] for name in assets.descriptor_columns}]
    )
    transformed = assets.descriptor_pipeline.transform(descriptor_frame).astype(
        np.float32
    )[0]
    return {
        "mof_id": mof_id,
        "graph_index": graph_index,
        "z": torch.from_numpy(graph["z"].astype(np.int64)),
        "x_cont": torch.from_numpy(normalized),
        "x_cont_mask": torch.from_numpy(mask),
        "src": torch.from_numpy(graph["src"].astype(np.int64)),
        "dst": torch.from_numpy(graph["dst"].astype(np.int64)),
        "edge_vec": torch.from_numpy(graph["edge_vec"].astype(np.float32)),
        "edge_relation": torch.from_numpy(
            graph["edge_relation"].astype(np.int64)
        ),
        "edge_bond_score": torch.from_numpy(
            graph["edge_bond_score"].astype(np.float32)
        ),
        "descriptor": torch.from_numpy(transformed),
    }


def collate(samples: list[dict]) -> dict:
    output = {
        key: []
        for key in (
            "mof_id",
            "graph_index",
            "z",
            "x_cont",
            "x_cont_mask",
            "src",
            "dst",
            "edge_vec",
            "edge_relation",
            "edge_bond_score",
            "batch",
            "descriptor",
        )
    }
    node_offset = 0
    for graph_number, sample in enumerate(samples):
        n_nodes = len(sample["z"])
        output["mof_id"].append(sample["mof_id"])
        output["graph_index"].append(sample["graph_index"])
        for key in ("z", "x_cont", "x_cont_mask"):
            output[key].append(sample[key])
        output["src"].append(sample["src"] + node_offset)
        output["dst"].append(sample["dst"] + node_offset)
        for key in ("edge_vec", "edge_relation", "edge_bond_score"):
            output[key].append(sample[key])
        output["batch"].append(
            torch.full((n_nodes,), graph_number, dtype=torch.long)
        )
        output["descriptor"].append(sample["descriptor"])
        node_offset += n_nodes
    for key in (
        "z",
        "x_cont",
        "x_cont_mask",
        "src",
        "dst",
        "edge_vec",
        "edge_relation",
        "edge_bond_score",
        "batch",
    ):
        output[key] = torch.cat(output[key], dim=0)
    output["descriptor"] = torch.stack(output["descriptor"])
    output["graph_index"] = torch.tensor(output["graph_index"], dtype=torch.long)
    output["n_graphs"] = len(samples)
    return output


def predict_samples(samples: list[dict], assets: FrozenAssets) -> np.ndarray:
    with torch.inference_mode():
        return assets.model(collate(samples))["fusion"].cpu().numpy()


def graph_from_h5(handle: h5py.File, index: int) -> dict:
    ns, ne = handle["graph/node_offsets"][index : index + 2]
    es, ee = handle["graph/edge_offsets"][index : index + 2]
    return {
        "z": handle["node/z"][ns:ne],
        "x_cont": handle["node/x_cont"][ns:ne],
        "x_cont_mask": handle["node/x_cont_mask"][ns:ne],
        "src": handle["edge/src"][es:ee],
        "dst": handle["edge/dst"][es:ee],
        "edge_vec": handle["edge/vector"][es:ee],
        "edge_distance": handle["edge/distance"][es:ee],
        "edge_image": handle["edge/image"][es:ee],
        "edge_relation": handle["edge/relation"][es:ee],
        "edge_bond_score": handle["edge/bond_score"][es:ee],
    }


def zip_member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    result = {}
    for name in archive.namelist():
        match = re.search(r"hmof-(\d+)", Path(name).name, flags=re.IGNORECASE)
        if match and name.lower().endswith(".cif"):
            result[f"hmof-{int(match.group(1))}"] = name
    return result


def graph_distance_signature(graph: dict) -> np.ndarray:
    return np.sort(graph["edge_distance"].astype(np.float32))


def validate_original(count: int, assets: FrozenAssets) -> None:
    require_files(
        [ORIGINAL_H5, ORIGINAL_CSV, ORIGINAL_CIF_ZIP, SAVED_TEST_PREDICTIONS]
    )
    saved = pd.read_csv(SAVED_TEST_PREDICTIONS).iloc[:count].copy()
    enriched = pd.read_csv(ORIGINAL_CSV).set_index("mof_id")
    saved_pred_columns = [
        column for column in saved.columns if column.startswith(("co2_pred", "ch4_pred"))
    ]
    with h5py.File(ORIGINAL_H5, "r") as handle:
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["graph/mof_id"][:]
        ]
        index_by_id = {mof_id: index for index, mof_id in enumerate(ids)}
        samples = []
        for mof_id in saved["mof_id"]:
            index = index_by_id[mof_id]
            graph = graph_from_h5(handle, index)
            descriptor = enriched.loc[mof_id, assets.descriptor_columns].to_dict()
            samples.append(
                prepare_sample(mof_id, graph, descriptor, assets, index)
            )
        start = time.perf_counter()
        reproduced = predict_samples(samples, assets)
        inference_seconds = time.perf_counter() - start

        saved_values = saved[saved_pred_columns].to_numpy(dtype=np.float32)
        absolute = np.abs(reproduced - saved_values)
        print(
            "Original-H5 prediction reproduction:",
            f"n={count}, max_abs={absolute.max():.6g}, mean_abs={absolute.mean():.6g}, "
            f"seconds={inference_seconds:.3f}",
        )

        with zipfile.ZipFile(ORIGINAL_CIF_ZIP) as archive:
            members = zip_member_map(archive)
            cpu_prediction_errors = []
            descriptor_errors = []
            graph_reports = []
            for row_number, mof_id in enumerate(saved["mof_id"]):
                index = index_by_id[mof_id]
                original_graph = graph_from_h5(handle, index)
                data = archive.read(members[mof_id])
                temporary_name = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".cif", delete=False
                    ) as temporary:
                        temporary.write(data)
                        temporary_name = temporary.name
                    cpu_graph = build_cpu_graph(Path(temporary_name), assets)
                finally:
                    if temporary_name and Path(temporary_name).exists():
                        Path(temporary_name).unlink()
                original_sig = graph_distance_signature(original_graph)
                cpu_sig = graph_distance_signature(cpu_graph)
                same_edges = len(original_sig) == len(cpu_sig)
                distance_error = (
                    float(np.max(np.abs(original_sig - cpu_sig)))
                    if same_edges
                    else float("nan")
                )
                base = enriched.loc[mof_id]
                cpu_descriptor = graph_descriptors(
                    mof_id, cpu_graph, base, assets
                )
                expected_descriptor = base[assets.descriptor_columns].astype(float)
                descriptor_vector = np.asarray(
                    [cpu_descriptor[name] for name in assets.descriptor_columns],
                    dtype=float,
                )
                descriptor_error = float(
                    np.max(np.abs(descriptor_vector - expected_descriptor.to_numpy()))
                )
                descriptor_errors.append(descriptor_error)
                sample = prepare_sample(
                    mof_id, cpu_graph, cpu_descriptor, assets, index
                )
                cpu_prediction = predict_samples([sample], assets)[0]
                prediction_error = float(
                    np.max(np.abs(cpu_prediction - reproduced[row_number]))
                )
                cpu_prediction_errors.append(prediction_error)
                graph_reports.append(
                    {
                        "mof_id": mof_id,
                        "nodes": len(cpu_graph["z"]),
                        "original_edges": len(original_sig),
                        "cpu_edges": len(cpu_sig),
                        "distance_max_abs": distance_error,
                        "descriptor_max_abs": descriptor_error,
                        "prediction_max_abs": prediction_error,
                    }
                )
            print(pd.DataFrame(graph_reports).to_string(index=False))
            print(
                "CPU graph validation maxima:",
                f"descriptor={max(descriptor_errors):.6g}, "
                f"prediction={max(cpu_prediction_errors):.6g}",
            )


def flush_predictions(records: list[dict]) -> None:
    frame = pd.DataFrame(records)
    temporary = EXTERNAL_PREDICTIONS.with_suffix(".csv.partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, EXTERNAL_PREDICTIONS)


def run_external(
    requested_count: int,
    assets: FrozenAssets,
    max_batch_nodes: int,
    max_batch_edges: int,
    flush_every: int,
) -> None:
    require_files([EXTERNAL_INPUTS, EXTERNAL_CIFS])
    inputs = pd.read_csv(EXTERNAL_INPUTS).sort_values("selection_rank")
    target_count = 10_000 if requested_count < 0 else requested_count
    if not 1 <= target_count <= 10_000:
        raise ValueError("External count must be 1..10000 or -1")
    existing_records = []
    if EXTERNAL_PREDICTIONS.is_file():
        existing_records = pd.read_csv(EXTERNAL_PREDICTIONS).to_dict("records")
    initial_record_count = len(existing_records)
    existing_ids = {str(row["mof_id"]) for row in existing_records}
    if len(existing_records) >= target_count:
        print(f"Already have {len(existing_records):,} predictions; nothing to do")
        return

    failures = []
    pending_samples = []
    pending_meta = []
    records = existing_records
    total_nodes = 0
    total_edges = 0
    started = time.perf_counter()
    last_flush_count = len(records)

    def run_pending() -> None:
        nonlocal pending_samples, pending_meta, total_nodes, total_edges
        nonlocal last_flush_count
        if not pending_samples:
            return
        batch_started = time.perf_counter()
        predictions = predict_samples(pending_samples, assets)
        batch_seconds = time.perf_counter() - batch_started
        per_graph_seconds = batch_seconds / len(pending_samples)
        for meta, prediction in zip(pending_meta, predictions):
            record = {
                **meta,
                **{
                    column: float(value)
                    for column, value in zip(PREDICTION_COLUMNS, prediction)
                },
                "batch_inference_seconds_per_graph": per_graph_seconds,
            }
            records.append(record)
        pending_samples = []
        pending_meta = []
        total_nodes = 0
        total_edges = 0
        if len(records) - last_flush_count >= flush_every:
            flush_predictions(records)
            last_flush_count = len(records)
            elapsed = time.perf_counter() - started
            rate = (len(records) - initial_record_count) / max(elapsed, 1e-9)
            print(
                f"Predicted {len(records):,}/{target_count:,}; "
                f"new-run rate={rate:.3f} structures/s",
                flush=True,
            )

    with zipfile.ZipFile(EXTERNAL_CIFS) as archive:
        members = zip_member_map(archive)
        for row in inputs.itertuples(index=False):
            if len(records) >= target_count:
                break
            mof_id = str(row.mof_id)
            if mof_id in existing_ids:
                continue
            if row.selection_role == "reserve" and not failures:
                continue
            temporary_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".cif", delete=False
                ) as temporary:
                    temporary.write(archive.read(members[mof_id]))
                    temporary_name = temporary.name
                graph_started = time.perf_counter()
                graph = build_cpu_graph(Path(temporary_name), assets)
                graph_seconds = time.perf_counter() - graph_started
                base = pd.Series(row._asdict())
                descriptor = graph_descriptors(mof_id, graph, base, assets)
                sample = prepare_sample(
                    mof_id, graph, descriptor, assets, int(row.selection_rank)
                )
            except Exception as exc:
                failures.append(
                    {
                        "selection_rank": int(row.selection_rank),
                        "mof_id": mof_id,
                        "error": repr(exc),
                    }
                )
                print(f"Failed {mof_id}: {exc}", flush=True)
                continue
            finally:
                if temporary_name and Path(temporary_name).exists():
                    Path(temporary_name).unlink()
            n_nodes = len(graph["z"])
            n_edges = len(graph["src"])
            exceeds = pending_samples and (
                total_nodes + n_nodes > max_batch_nodes
                or total_edges + n_edges > max_batch_edges
            )
            if exceeds:
                run_pending()
                if len(records) >= target_count:
                    break
            pending_samples.append(sample)
            pending_meta.append(
                {
                    "selection_rank": int(row.selection_rank),
                    "selection_role": str(row.selection_role),
                    "mof_id": mof_id,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "graph_build_seconds": graph_seconds,
                }
            )
            total_nodes += n_nodes
            total_edges += n_edges
        run_pending()

    if len(records) > target_count:
        records = records[:target_count]
    flush_predictions(records)
    failure_path = EXTERNAL_DIR / "external_preprocessing_failures.json"
    failure_path.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    print(
        f"External inference finished: {len(records):,}/{target_count:,}, "
        f"failures={len(failures)}, run_seconds={elapsed:.1f}"
    )


def evaluate_external(assets: FrozenAssets) -> None:
    require_files([EXTERNAL_PREDICTIONS, EXTERNAL_LABELS])
    predictions = pd.read_csv(EXTERNAL_PREDICTIONS)
    if len(predictions) != 10_000:
        raise RuntimeError(
            f"Evaluation requires exactly 10,000 predictions; found {len(predictions):,}"
        )
    labels = pd.read_csv(EXTERNAL_LABELS)
    merged = predictions.merge(
        labels[["mof_id", *TARGET_COLUMNS]],
        on="mof_id",
        how="left",
        validate="one_to_one",
    )
    if merged[TARGET_COLUMNS].isna().any().any():
        raise RuntimeError("Missing private labels after merge")
    per_target = {}
    for target, prediction in zip(TARGET_COLUMNS, PREDICTION_COLUMNS):
        truth = merged[target].to_numpy(dtype=float)
        estimate = merged[prediction].to_numpy(dtype=float)
        per_target[target] = {
            "r2": float(r2_score(truth, estimate)),
            "rmse": float(mean_squared_error(truth, estimate) ** 0.5),
            "mae": float(mean_absolute_error(truth, estimate)),
        }
    overall_truth = merged[TARGET_COLUMNS].to_numpy(dtype=float)
    overall_prediction = merged[PREDICTION_COLUMNS].to_numpy(dtype=float)
    payload = {
        "n": len(merged),
        "selection_rank_min": int(merged["selection_rank"].min()),
        "selection_rank_max": int(merged["selection_rank"].max()),
        "reserve_rows_used": int((merged["selection_role"] == "reserve").sum()),
        "per_target": per_target,
        "macro_mean": {
            metric: float(
                np.mean([values[metric] for values in per_target.values()])
            )
            for metric in ("r2", "rmse", "mae")
        },
        "micro_all_10_outputs": {
            "r2": float(r2_score(overall_truth.ravel(), overall_prediction.ravel())),
            "rmse": float(
                mean_squared_error(
                    overall_truth.ravel(), overall_prediction.ravel()
                )
                ** 0.5
            ),
            "mae": float(
                mean_absolute_error(
                    overall_truth.ravel(), overall_prediction.ravel()
                )
            ),
        },
    }
    EXTERNAL_METRICS.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    inference_manifest = {
        "protocol": (
            "Predictions were completed from the frozen inputs CSV and CIF ZIP "
            "before the private-label CSV was opened for evaluation."
        ),
        "device": "CPU",
        "model_parameters": int(
            sum(parameter.numel() for parameter in assets.model.parameters())
        ),
        "n_predictions": len(predictions),
        "unique_mof_ids": int(predictions["mof_id"].nunique()),
        "selection_rank_range": [
            int(predictions["selection_rank"].min()),
            int(predictions["selection_rank"].max()),
        ],
        "reserve_rows_used": int(
            (predictions["selection_role"] == "reserve").sum()
        ),
        "all_predictions_finite": bool(
            np.isfinite(predictions[PREDICTION_COLUMNS].to_numpy(float)).all()
        ),
        "files_sha256": {
            "checkpoint": sha256_file(CHECKPOINT),
            "inputs": sha256_file(EXTERNAL_INPUTS),
            "cifs": sha256_file(EXTERNAL_CIFS),
            "predictions": sha256_file(EXTERNAL_PREDICTIONS),
            "private_labels": sha256_file(EXTERNAL_LABELS),
            "metrics": sha256_file(EXTERNAL_METRICS),
        },
        "software": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    EXTERNAL_INFERENCE_MANIFEST.write_text(
        json.dumps(inference_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    configure_paths(args)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(
        f"PyTorch={torch.__version__}, threads={torch.get_num_threads()}, "
        f"device=cpu"
    )
    assets = FrozenAssets()
    parameters = sum(parameter.numel() for parameter in assets.model.parameters())
    print(f"Loaded model parameters: {parameters:,}")
    if args.validate_original:
        validate_original(args.validate_original, assets)
    if args.run_external:
        run_external(
            args.run_external,
            assets,
            args.max_batch_nodes,
            args.max_batch_edges,
            args.flush_every,
        )
    if args.evaluate:
        evaluate_external(assets)
    if not (args.validate_original or args.run_external or args.evaluate):
        print("No action selected. Use --validate-original, --run-external, or --evaluate.")


if __name__ == "__main__":
    main()
