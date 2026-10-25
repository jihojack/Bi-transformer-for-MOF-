# %% [markdown]
# # MOFX-DB 50K ALCHEMI feature extraction
#
# This Colab notebook builds a periodic spatial graph for each CIF with NVIDIA
# ALCHEMI Toolkit-Ops on an L4 GPU. ALCHEMI finds periodic neighbors; it does
# not infer chemical bonds. Bond-like edge scores are therefore transparent
# covalent-radius priors, not experimental bond orders.
#
# Inputs:
# - `MOF_CO2_CH4_50K.csv`
# - `MOF_CO2_CH4_50K_CIF.zip`
# - `MOF_CO2_CH4_50K_split_70_10_20.json`
#
# Outputs:
# - `periodic_graphs_50k_alchemi_compact.h5`
# - `MOF_CO2_CH4_50K_alchemi_enriched.csv`
# - `element_property_table_50k.json`
# - `alchemi_50k_feature_manifest.json`

# %%
import subprocess
import sys


PACKAGES = [
    "nvalchemi-toolkit-ops[torch]==0.3.1",
    "warp-lang>=1.9,<2",
    "ase>=3.26,<4",
    "pymatgen>=2025.6,<2027",
    "mendeleev>=1.1,<2",
    "h5py>=3.11,<4",
    "joblib>=1.4,<2",
]
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", *PACKAGES]
)
print("Installation complete. Continue without restarting unless Colab requests it.")

# %%
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import ase
import h5py
import numpy as np
import pandas as pd
import pymatgen
import torch
import warp as wp
from ase.data import atomic_numbers, chemical_symbols
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii
from ase.io import read as ase_read
from mendeleev import element as mendeleev_element
from pymatgen.core import Element, Structure
from tqdm.auto import tqdm

import nvalchemiops
from nvalchemiops.torch.neighbors import neighbor_list

try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass

warnings.filterwarnings("ignore")


@dataclass
class Config:
    seed: int = 42
    expected_rows: int = 50_000
    cutoff_angstrom: float = 6.0
    max_neighbors: int = 32
    max_atoms_per_gpu_batch: int = 12_000
    max_structures_per_gpu_batch: int = 128
    parse_workers: int = 6
    parse_chunk_size: int = 512
    h5_gzip_level: int = 4
    bond_score_temperature: float = 0.08
    nonmetal_nonmetal_ratio_limit: float = 1.25
    metal_nonmetal_ratio_limit: float = 1.40
    metal_metal_ratio_limit: float = 1.25
    overwrite_local_h5: bool = False


CFG = Config()
CSV_NAME = "MOF_CO2_CH4_50K.csv"
CIF_ZIP_NAME = "MOF_CO2_CH4_50K_CIF.zip"
SPLIT_NAME = "MOF_CO2_CH4_50K_split_70_10_20.json"
H5_NAME = "periodic_graphs_50k_alchemi_compact.h5"
ENRICHED_CSV_NAME = "MOF_CO2_CH4_50K_alchemi_enriched.csv"
ELEMENT_TABLE_NAME = "element_property_table_50k.json"
MANIFEST_NAME = "alchemi_50k_feature_manifest.json"

LOCAL_ROOT = Path("/content/mof_50k_alchemi")
LOCAL_INPUT = LOCAL_ROOT / "input"
LOCAL_CIF_DIR = LOCAL_INPUT / "cifs"
LOCAL_OUTPUT = LOCAL_ROOT / "output"
for directory in (LOCAL_INPUT, LOCAL_CIF_DIR, LOCAL_OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before running this notebook.")

DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)

print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
print("PyTorch:", torch.__version__)
print("ALCHEMI Toolkit-Ops:", nvalchemiops.__version__)
print("Warp:", wp.__version__)
print("neighbor_list signature:", inspect.signature(neighbor_list))

# %% [markdown]
# ## 1. Load inputs from the fixed Drive folder and copy them to Colab storage
#
# Drive is used only for the three inputs and completed outputs. CIF extraction,
# graph generation, HDF5 construction, descriptors, and verification all run
# under `/content/mof_50k_alchemi` on the Colab runtime disk.

# %%
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/mf_hid/4th")
DRIVE_OUTPUT_DIR = DRIVE_DATA_DIR
DRIVE_CSV = DRIVE_DATA_DIR / CSV_NAME
DRIVE_ZIP = DRIVE_DATA_DIR / CIF_ZIP_NAME
DRIVE_SPLIT = DRIVE_DATA_DIR / SPLIT_NAME
for required_path in (DRIVE_CSV, DRIVE_ZIP, DRIVE_SPLIT):
    if not required_path.exists():
        raise FileNotFoundError(required_path)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_CSV = LOCAL_INPUT / CSV_NAME
LOCAL_ZIP = LOCAL_INPUT / CIF_ZIP_NAME
LOCAL_SPLIT = LOCAL_INPUT / SPLIT_NAME
for source, destination in (
    (DRIVE_CSV, LOCAL_CSV),
    (DRIVE_ZIP, LOCAL_ZIP),
    (DRIVE_SPLIT, LOCAL_SPLIT),
):
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        print(f"Copying {source.name} to local runtime storage...")
        shutil.copy2(source, destination)

df = pd.read_csv(LOCAL_CSV)
if len(df) != CFG.expected_rows:
    raise ValueError(f"Expected {CFG.expected_rows:,} rows, found {len(df):,}.")
if df["mof_id"].duplicated().any():
    raise ValueError("mof_id must be unique.")

split_payload = json.loads(LOCAL_SPLIT.read_text(encoding="utf-8"))
train_indices = np.asarray(split_payload["train_indices"], dtype=np.int64)
split_ids = set().union(
    *(set(split_payload[f"{name}_mof_ids"]) for name in ("train", "val", "test"))
)
if len(split_ids) != len(df) or split_ids != set(df["mof_id"]):
    raise RuntimeError("Split JSON does not cover exactly the 50K CSV IDs.")
if len(train_indices) != 35_000:
    raise RuntimeError(f"Expected 35,000 training indices, found {len(train_indices):,}.")

extract_marker = LOCAL_CIF_DIR / ".extract_complete"
if not extract_marker.exists():
    print("Extracting CIF files to local runtime storage...")
    with zipfile.ZipFile(LOCAL_ZIP) as archive:
        archive.extractall(LOCAL_CIF_DIR)
    extract_marker.write_text("complete\n", encoding="ascii")


def canonical_mof_id(value: str) -> str:
    match = re.search(r"hmof-(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse MOF id from {value!r}")
    return f"hmof-{int(match.group(1))}"


cif_by_id = {}
for path in LOCAL_CIF_DIR.rglob("*.cif"):
    mof_id = canonical_mof_id(path.name)
    if mof_id in cif_by_id:
        raise RuntimeError(f"Duplicate CIF for {mof_id}")
    cif_by_id[mof_id] = path

missing_cifs = sorted(set(df["mof_id"]) - set(cif_by_id))
extra_cifs = sorted(set(cif_by_id) - set(df["mof_id"]))
if missing_cifs or extra_cifs:
    raise RuntimeError(
        f"CSV/CIF mismatch: missing={missing_cifs[:10]}, extra={extra_cifs[:10]}"
    )

print("Input folder:", DRIVE_DATA_DIR)
print("Output folder:", DRIVE_OUTPUT_DIR)
print("Rows/CIFs:", f"{len(df):,}/{len(cif_by_id):,}")

# %% [markdown]
# ## 2. Build a versioned element-property table
#
# Missing continuous values receive `0` only in the numeric tensor and retain a
# separate validity mask. Atomic number is also stored independently as `z`.

# %%
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


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_element_symbols(series: pd.Series) -> list[str]:
    symbols = set()
    for value in series.dropna():
        for token in re.findall(r"[A-Z][a-z]?", str(value)):
            try:
                symbols.add(Element(token).symbol)
            except Exception:
                continue
    return sorted(symbols, key=lambda symbol: Element(symbol).Z)


def build_element_record(symbol: str) -> dict:
    pmg = Element(symbol)
    mend = mendeleev_element(symbol)
    z = int(pmg.Z)

    covalent_radius = finite_or_none(mend.covalent_radius_pyykko)
    vdw_radius = finite_or_none(mend.vdw_radius)
    polarizability = finite_or_none(mend.dipole_polarizability)
    try:
        valence_electrons = finite_or_none(mend.nvalence())
    except Exception:
        valence_electrons = None

    if covalent_radius is not None:
        covalent_radius /= 100.0
    elif z < len(ase_covalent_radii):
        covalent_radius = finite_or_none(ase_covalent_radii[z])

    if vdw_radius is not None:
        vdw_radius /= 100.0
    elif z < len(ase_vdw_radii):
        vdw_radius = finite_or_none(ase_vdw_radii[z])

    return {
        "atomic_number": float(z),
        "electronegativity_pauling": finite_or_none(pmg.X),
        "atomic_mass_da": finite_or_none(pmg.atomic_mass),
        "period": finite_or_none(pmg.row),
        "group": finite_or_none(pmg.group),
        "metal_flag": float(bool(pmg.is_metal)),
        "valence_electrons": valence_electrons,
        "polarizability_bohr3": polarizability,
        "vdw_radius_angstrom": vdw_radius,
        "covalent_radius_angstrom": covalent_radius,
    }


ELEMENT_SYMBOLS = parse_element_symbols(df["elements"])
ELEMENT_PROPERTIES = {
    symbol: build_element_record(symbol) for symbol in ELEMENT_SYMBOLS
}
Z_TO_SYMBOL = {
    int(record["atomic_number"]): symbol
    for symbol, record in ELEMENT_PROPERTIES.items()
}

ELEMENT_TABLE_PATH = LOCAL_OUTPUT / ELEMENT_TABLE_NAME
ELEMENT_TABLE_PATH.write_text(
    json.dumps(ELEMENT_PROPERTIES, indent=2, sort_keys=True), encoding="utf-8"
)
element_table_df = pd.DataFrame.from_dict(ELEMENT_PROPERTIES, orient="index")
display(element_table_df)
display(element_table_df.isna().sum().to_frame("missing_element_count"))

# %% [markdown]
# ## 3. Parse CIFs and define the ALCHEMI graph builder
#
# ASE is the fast primary parser. Pymatgen is used only as a fallback. The GPU
# neighbor graph contains every periodic neighbor inside 6 Angstrom, capped at
# the nearest 32 outgoing neighbors per atom.

# %%
@dataclass
class ParsedStructure:
    row_index: int
    mof_id: str
    positions: np.ndarray
    cell: np.ndarray
    z: np.ndarray
    x_element: np.ndarray
    x_element_mask: np.ndarray


def element_arrays(symbols: list[str]):
    unknown = sorted(set(symbols) - set(ELEMENT_PROPERTIES))
    if unknown:
        raise KeyError(f"Missing element properties for {unknown}")
    values = []
    masks = []
    z_values = []
    for symbol in symbols:
        record = ELEMENT_PROPERTIES[symbol]
        row = [record[name] for name in ELEMENT_FEATURE_NAMES]
        mask = [value is not None and math.isfinite(value) for value in row]
        values.append(
            [float(value) if valid else 0.0 for value, valid in zip(row, mask)]
        )
        masks.append(mask)
        z_values.append(int(record["atomic_number"]))
    return (
        np.asarray(z_values, dtype=np.uint8),
        np.asarray(values, dtype=np.float32),
        np.asarray(masks, dtype=np.uint8),
    )


def parse_cif(job) -> ParsedStructure:
    row_index, mof_id, path = job
    try:
        atoms = ase_read(path, format="cif")
        if len(atoms) == 0 or atoms.cell.rank != 3:
            raise ValueError("ASE returned an empty or non-3D structure")
        atoms.set_pbc((True, True, True))
        atoms.wrap(eps=1.0e-10)
        positions = np.asarray(atoms.positions, dtype=np.float32)
        cell = np.asarray(atoms.cell.array, dtype=np.float32)
        symbols = atoms.get_chemical_symbols()
    except Exception:
        structure = Structure.from_file(path, primitive=False)
        if len(structure) == 0:
            raise ValueError("Pymatgen returned an empty structure")
        positions = np.asarray(structure.cart_coords, dtype=np.float32)
        cell = np.asarray(structure.lattice.matrix, dtype=np.float32)
        symbols = []
        for site in structure:
            species = max(site.species.items(), key=lambda item: float(item[1]))[0]
            symbols.append(species.symbol)

    if abs(np.linalg.det(cell)) < 1.0e-6:
        raise ValueError("Singular unit cell")
    z, x_element, x_element_mask = element_arrays(symbols)
    return ParsedStructure(
        row_index=int(row_index),
        mof_id=mof_id,
        positions=positions,
        cell=cell,
        z=z,
        x_element=x_element,
        x_element_mask=x_element_mask,
    )


def alchemi_neighbor_list(positions, cells, batch_idx, batch_ptr):
    pbc = torch.ones((len(cells), 3), dtype=torch.bool, device=DEVICE)
    kwargs = {
        "cell": cells,
        "pbc": pbc,
        "batch_idx": batch_idx,
        "batch_ptr": batch_ptr,
        "method": "cell_list" if len(cells) == 1 else "batch_cell_list",
        "return_neighbor_list": True,
    }
    signature = inspect.signature(neighbor_list)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    result = neighbor_list(positions, CFG.cutoff_angstrom, **kwargs)
    if len(result) != 3:
        raise RuntimeError(f"Unexpected ALCHEMI neighbor_list output: {len(result)} items")
    edge_index, neighbor_ptr, shifts = result
    if edge_index.ndim != 2:
        raise RuntimeError(f"Unexpected edge_index shape: {tuple(edge_index.shape)}")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise RuntimeError(f"Unexpected edge_index shape: {tuple(edge_index.shape)}")
    return edge_index.long(), neighbor_ptr, shifts


def periodic_edge_geometry(positions, cells, batch_idx, edge_index, shifts):
    src, dst = edge_index
    graph_index = batch_idx[src]
    if shifts.ndim == 2 and shifts.shape == (3, edge_index.shape[1]):
        shifts = shifts.T
    shifts = shifts.reshape(-1, 3).to(dtype=torch.float32)
    if len(shifts) != edge_index.shape[1]:
        raise RuntimeError(
            f"Shift/edge count mismatch: {len(shifts)} != {edge_index.shape[1]}"
        )
    integer_like = bool(torch.all(torch.abs(shifts - shifts.round()) < 1.0e-4))
    if integer_like:
        unit_shifts = shifts.round()
        shift_cart = torch.einsum("ei,eij->ej", unit_shifts, cells[graph_index])
    else:
        # Kept for API compatibility if a future release returns Cartesian shifts.
        unit_shifts = torch.zeros_like(shifts)
        shift_cart = shifts

    base = positions[dst] - positions[src]
    plus = base + shift_cart
    minus = base - shift_cart
    plus_distance = plus.norm(dim=-1)
    minus_distance = minus.norm(dim=-1)

    def geometry_penalty(distance):
        return (
            torch.relu(distance - (CFG.cutoff_angstrom + 1.0e-3)).sum()
            + (distance < 1.0e-6).float().sum() * 1000.0
        )

    plus_penalty = geometry_penalty(plus_distance)
    minus_penalty = geometry_penalty(minus_distance)
    if float(plus_penalty) <= float(minus_penalty):
        return plus, plus_distance, unit_shifts.to(torch.int8), "+"
    return minus, minus_distance, (-unit_shifts).to(torch.int8), "-"


def nearest_neighbor_cap(src, distance):
    key = src.to(torch.float64) * (CFG.cutoff_angstrom + 1.0) + distance.to(torch.float64)
    order = torch.argsort(key)
    sorted_src = src[order]
    _, counts = torch.unique_consecutive(sorted_src, return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(len(order), device=DEVICE) - torch.repeat_interleave(starts, counts)
    return order[rank < CFG.max_neighbors]


def process_gpu_batch(items: list[ParsedStructure]):
    node_counts = [len(item.z) for item in items]
    batch_ptr = torch.tensor(
        np.cumsum([0, *node_counts]), dtype=torch.int32, device=DEVICE
    )
    batch_idx = torch.repeat_interleave(
        torch.arange(len(items), dtype=torch.int32, device=DEVICE),
        torch.tensor(node_counts, dtype=torch.long, device=DEVICE),
    )
    positions = torch.cat(
        [torch.from_numpy(item.positions) for item in items]
    ).to(DEVICE)
    cells = torch.stack(
        [torch.from_numpy(item.cell) for item in items]
    ).to(DEVICE)
    z = torch.cat([torch.from_numpy(item.z.astype(np.int64)) for item in items]).to(DEVICE)
    x_element = torch.cat(
        [torch.from_numpy(item.x_element) for item in items]
    ).to(DEVICE)
    x_element_mask = torch.cat(
        [torch.from_numpy(item.x_element_mask) for item in items]
    ).to(DEVICE)

    edge_index, _, shifts = alchemi_neighbor_list(
        positions, cells, batch_idx, batch_ptr
    )
    batch_idx_long = batch_idx.long()
    if edge_index.device.type != "cuda":
        raise RuntimeError("ALCHEMI neighbor output is not on the GPU.")
    src, dst = edge_index
    if not torch.equal(batch_idx_long[src], batch_idx_long[dst]):
        raise RuntimeError("ALCHEMI returned an edge crossing two structures.")

    edge_vec, distance, edge_image, shift_sign = periodic_edge_geometry(
        positions, cells, batch_idx_long, edge_index, shifts
    )
    valid = (distance > 1.0e-5) & (distance <= CFG.cutoff_angstrom + 1.0e-3)
    src, dst = src[valid], dst[valid]
    edge_vec, distance, edge_image = (
        edge_vec[valid],
        distance[valid],
        edge_image[valid],
    )
    keep = nearest_neighbor_cap(src, distance)
    src, dst = src[keep], dst[keep]
    edge_vec, distance, edge_image = (
        edge_vec[keep],
        distance[keep],
        edge_image[keep],
    )

    metal_index = ELEMENT_FEATURE_NAMES.index("metal_flag")
    covalent_index = ELEMENT_FEATURE_NAMES.index("covalent_radius_angstrom")
    is_metal = x_element[:, metal_index] > 0.5
    radius_sum = (
        x_element[src, covalent_index] + x_element[dst, covalent_index]
    ).clamp_min(0.5)
    ratio = distance / radius_sum
    pair_class = torch.where(
        is_metal[src] & is_metal[dst],
        torch.full_like(src, 3),
        torch.where(
            is_metal[src] | is_metal[dst],
            torch.full_like(src, 2),
            torch.ones_like(src),
        ),
    )
    ratio_limit = torch.where(
        pair_class == 1,
        torch.full_like(distance, CFG.nonmetal_nonmetal_ratio_limit),
        torch.where(
            pair_class == 2,
            torch.full_like(distance, CFG.metal_nonmetal_ratio_limit),
            torch.full_like(distance, CFG.metal_metal_ratio_limit),
        ),
    )
    bond_score = torch.sigmoid(
        (ratio_limit - ratio) / CFG.bond_score_temperature
    )
    bond_candidate = bond_score >= 0.5
    relation = torch.where(bond_candidate, pair_class, torch.zeros_like(pair_class))

    n_nodes = len(positions)
    coordination = torch.zeros(n_nodes, dtype=torch.float32, device=DEVICE)
    soft_coordination = torch.zeros_like(coordination)
    density_r3 = torch.zeros_like(coordination)
    density_r6 = torch.zeros_like(coordination)
    coordination.scatter_add_(0, src, bond_candidate.float())
    soft_coordination.scatter_add_(0, src, bond_score)
    density_r3.scatter_add_(0, src, torch.exp(-torch.square(distance / 3.0)))
    density_r6.scatter_add_(0, src, torch.exp(-torch.square(distance / 6.0)))
    x_local = torch.stack(
        (coordination, soft_coordination, density_r3, density_r6), dim=1
    )
    x_cont = torch.cat((x_element, x_local), dim=1)
    x_cont_mask = torch.cat(
        (
            x_element_mask,
            torch.ones_like(x_local, dtype=torch.uint8),
        ),
        dim=1,
    )

    graph_index_for_edge = batch_idx_long[src]
    graphs = []
    for graph_index, item in enumerate(items):
        node_start = int(batch_ptr[graph_index])
        node_end = int(batch_ptr[graph_index + 1])
        edge_mask = graph_index_for_edge == graph_index
        local_src = src[edge_mask] - node_start
        local_dst = dst[edge_mask] - node_start
        if len(local_src) == 0:
            raise RuntimeError(f"No periodic edges for {item.mof_id}")
        if max(int(local_src.max()), int(local_dst.max())) >= node_end - node_start:
            raise RuntimeError(f"Local edge index overflow for {item.mof_id}")
        graphs.append(
            {
                "row_index": item.row_index,
                "mof_id": item.mof_id,
                "z": z[node_start:node_end].to(torch.uint8).cpu().numpy(),
                "x_cont": x_cont[node_start:node_end].cpu().numpy().astype(np.float32),
                "x_cont_mask": x_cont_mask[node_start:node_end].cpu().numpy().astype(np.uint8),
                "src": local_src.cpu().numpy().astype(np.uint16),
                "dst": local_dst.cpu().numpy().astype(np.uint16),
                "edge_vec": edge_vec[edge_mask].cpu().numpy().astype(np.float16),
                "edge_distance": distance[edge_mask].cpu().numpy().astype(np.float16),
                "edge_image": edge_image[edge_mask].cpu().numpy().astype(np.int8),
                "edge_relation": relation[edge_mask].to(torch.uint8).cpu().numpy(),
                "edge_bond_score": bond_score[edge_mask].cpu().numpy().astype(np.float16),
                "shift_sign": shift_sign,
            }
        )
    return graphs


def process_gpu_batch_with_retry(items):
    try:
        return process_gpu_batch(items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            raise
        middle = len(items) // 2
        return process_gpu_batch_with_retry(items[:middle]) + process_gpu_batch_with_retry(items[middle:])

# %% [markdown]
# ## 4. ALCHEMI smoke test and rotation check
#
# The first call includes Warp kernel compilation. The test also verifies that
# rotating both the Cartesian coordinates and unit cell preserves the sorted
# neighbor-distance multiset.

# %%
first_id = str(df.iloc[0]["mof_id"])
sample = parse_cif((0, first_id, cif_by_id[first_id]))
sample_graph = process_gpu_batch([sample])[0]

rng = np.random.default_rng(CFG.seed)
q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
if np.linalg.det(q) < 0:
    q[:, 0] *= -1
rotated = ParsedStructure(
    row_index=sample.row_index,
    mof_id=sample.mof_id,
    positions=(sample.positions @ q.T).astype(np.float32),
    cell=(sample.cell @ q.T).astype(np.float32),
    z=sample.z,
    x_element=sample.x_element,
    x_element_mask=sample.x_element_mask,
)
rotated_graph = process_gpu_batch([rotated])[0]
original_distances = np.sort(sample_graph["edge_distance"].astype(np.float32))
rotated_distances = np.sort(rotated_graph["edge_distance"].astype(np.float32))
if len(original_distances) != len(rotated_distances):
    raise RuntimeError("Rotation changed the number of retained edges.")
rotation_distance_error = float(
    np.max(np.abs(original_distances - rotated_distances))
)
if rotation_distance_error > 2.0e-2:
    raise RuntimeError(
        f"Rotation distance stress test failed: {rotation_distance_error:.6f} Angstrom"
    )
print("Smoke-test atoms:", len(sample_graph["z"]))
print("Smoke-test edges:", len(sample_graph["src"]))
print("Rotation max distance difference:", rotation_distance_error, "Angstrom")
del sample_graph, rotated_graph
torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Stream all 50K graphs into one compact HDF5 file
#
# Inputs and the final HDF5 are kept on the Colab runtime during extraction.
# Only the completed, verified files are copied back to Drive.

# %%
LOCAL_H5 = LOCAL_OUTPUT / H5_NAME


def h5_build_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return bool(handle.attrs.get("build_complete", False)) and int(
                handle.attrs.get("graph_count", -1)
            ) == len(df)
    except OSError:
        return False


if LOCAL_H5.exists() and (CFG.overwrite_local_h5 or not h5_build_is_complete(LOCAL_H5)):
    print("Removing an incomplete or explicitly overwritten local HDF5:", LOCAL_H5)
    LOCAL_H5.unlink()
elif LOCAL_H5.exists():
    print("Verified complete local HDF5 found; extraction will be skipped:", LOCAL_H5)


def create_resizable_dataset(group, name, tail_shape, dtype, chunk_rows):
    return group.create_dataset(
        name,
        shape=(0, *tail_shape),
        maxshape=(None, *tail_shape),
        dtype=dtype,
        chunks=(chunk_rows, *tail_shape),
        compression="gzip",
        compression_opts=CFG.h5_gzip_level,
        shuffle=True,
    )


def append_rows(dataset, values):
    old_size = dataset.shape[0]
    new_size = old_size + len(values)
    dataset.resize(new_size, axis=0)
    dataset[old_size:new_size] = values


def parsed_batches():
    pending = []
    pending_atoms = 0
    jobs = [
        (index, str(row.mof_id), cif_by_id[str(row.mof_id)])
        for index, row in enumerate(df.itertuples(index=False))
    ]
    with ThreadPoolExecutor(max_workers=CFG.parse_workers) as executor:
        for start in range(0, len(jobs), CFG.parse_chunk_size):
            chunk = jobs[start : start + CFG.parse_chunk_size]
            parsed = list(executor.map(parse_cif, chunk))
            for item in parsed:
                n_atoms = len(item.z)
                should_flush = pending and (
                    pending_atoms + n_atoms > CFG.max_atoms_per_gpu_batch
                    or len(pending) >= CFG.max_structures_per_gpu_batch
                )
                if should_flush:
                    yield pending
                    pending = []
                    pending_atoms = 0
                pending.append(item)
                pending_atoms += n_atoms
    if pending:
        yield pending


def write_graph_batch(graphs, datasets, node_offsets, edge_offsets, graph_cursor):
    graphs = sorted(graphs, key=lambda graph: graph["row_index"])
    node_values = np.concatenate([graph["z"] for graph in graphs])
    x_values = np.concatenate([graph["x_cont"] for graph in graphs])
    mask_values = np.concatenate([graph["x_cont_mask"] for graph in graphs])
    src_values = np.concatenate([graph["src"] for graph in graphs])
    dst_values = np.concatenate([graph["dst"] for graph in graphs])
    vector_values = np.concatenate([graph["edge_vec"] for graph in graphs])
    distance_values = np.concatenate([graph["edge_distance"] for graph in graphs])
    image_values = np.concatenate([graph["edge_image"] for graph in graphs])
    relation_values = np.concatenate([graph["edge_relation"] for graph in graphs])
    score_values = np.concatenate([graph["edge_bond_score"] for graph in graphs])

    append_rows(datasets["z"], node_values)
    append_rows(datasets["x_cont"], x_values)
    append_rows(datasets["x_cont_mask"], mask_values)
    append_rows(
        datasets["partial_charge_ddec6"],
        np.full(len(node_values), np.nan, dtype=np.float32),
    )
    append_rows(
        datasets["partial_charge_mask"],
        np.zeros(len(node_values), dtype=np.uint8),
    )
    append_rows(datasets["src"], src_values)
    append_rows(datasets["dst"], dst_values)
    append_rows(datasets["vector"], vector_values)
    append_rows(datasets["distance"], distance_values)
    append_rows(datasets["image"], image_values)
    append_rows(datasets["relation"], relation_values)
    append_rows(datasets["bond_score"], score_values)

    node_cursor = int(node_offsets[graph_cursor])
    edge_cursor = int(edge_offsets[graph_cursor])
    for graph in graphs:
        expected_id = str(df.iloc[graph_cursor]["mof_id"])
        if graph["mof_id"] != expected_id or graph["row_index"] != graph_cursor:
            raise RuntimeError(
                f"Graph order mismatch at row {graph_cursor}: {graph['mof_id']} != {expected_id}"
            )
        node_cursor += len(graph["z"])
        edge_cursor += len(graph["src"])
        graph_cursor += 1
        node_offsets[graph_cursor] = node_cursor
        edge_offsets[graph_cursor] = edge_cursor
    return graph_cursor


if not LOCAL_H5.exists():
    started = time.perf_counter()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(LOCAL_H5, "w") as handle:
        handle.attrs["format"] = "mof_50k_alchemi_graph_hdf5"
        handle.attrs["format_version"] = 2
        handle.attrs["graph_count"] = len(df)
        handle.attrs["build_complete"] = False
        handle.attrs["cutoff_angstrom"] = CFG.cutoff_angstrom
        handle.attrs["max_neighbors"] = CFG.max_neighbors
        handle.attrs["node_feature_names_json"] = json.dumps(NODE_FEATURE_NAMES)
        handle.attrs["edge_relation_labels_json"] = json.dumps(
            {
                0: "spatial_only",
                1: "nonmetal_nonmetal_bond_candidate",
                2: "metal_nonmetal_bond_candidate",
                3: "metal_metal_bond_candidate",
            }
        )
        handle.attrs["partial_charge_status"] = (
            "unavailable; values are NaN and masks are zero"
        )
        handle.attrs["neighbor_backend"] = "nvalchemiops.torch.neighbors.neighbor_list"
        handle.attrs["nvalchemiops_version"] = str(nvalchemiops.__version__)

        graph_group = handle.create_group("graph")
        node_group = handle.create_group("node")
        edge_group = handle.create_group("edge")
        graph_group.create_dataset(
            "mof_id",
            data=np.asarray(df["mof_id"].astype(str), dtype=object),
            dtype=string_dtype,
        )
        graph_group.create_dataset(
            "row_index", data=np.arange(len(df), dtype=np.int32)
        )
        node_offsets = graph_group.create_dataset(
            "node_offsets", shape=(len(df) + 1,), dtype=np.int64
        )
        edge_offsets = graph_group.create_dataset(
            "edge_offsets", shape=(len(df) + 1,), dtype=np.int64
        )
        node_offsets[0] = 0
        edge_offsets[0] = 0

        datasets = {
            "z": create_resizable_dataset(node_group, "z", (), np.uint8, 250_000),
            "x_cont": create_resizable_dataset(
                node_group, "x_cont", (len(NODE_FEATURE_NAMES),), np.float32, 25_000
            ),
            "x_cont_mask": create_resizable_dataset(
                node_group, "x_cont_mask", (len(NODE_FEATURE_NAMES),), np.uint8, 50_000
            ),
            "partial_charge_ddec6": create_resizable_dataset(
                node_group, "partial_charge_ddec6", (), np.float32, 250_000
            ),
            "partial_charge_mask": create_resizable_dataset(
                node_group, "partial_charge_mask", (), np.uint8, 250_000
            ),
            "src": create_resizable_dataset(edge_group, "src", (), np.uint16, 500_000),
            "dst": create_resizable_dataset(edge_group, "dst", (), np.uint16, 500_000),
            "vector": create_resizable_dataset(edge_group, "vector", (3,), np.float16, 125_000),
            "distance": create_resizable_dataset(edge_group, "distance", (), np.float16, 500_000),
            "image": create_resizable_dataset(edge_group, "image", (3,), np.int8, 250_000),
            "relation": create_resizable_dataset(edge_group, "relation", (), np.uint8, 500_000),
            "bond_score": create_resizable_dataset(edge_group, "bond_score", (), np.float16, 500_000),
        }

        graph_cursor = 0
        progress = tqdm(total=len(df), desc="ALCHEMI graph extraction")
        for batch in parsed_batches():
            graphs = process_gpu_batch_with_retry(batch)
            before = graph_cursor
            graph_cursor = write_graph_batch(
                graphs, datasets, node_offsets, edge_offsets, graph_cursor
            )
            progress.update(graph_cursor - before)
            if graph_cursor % 1_000 < len(graphs):
                handle.flush()
        progress.close()
        if graph_cursor != len(df):
            raise RuntimeError(f"Only wrote {graph_cursor:,}/{len(df):,} graphs")
        handle.attrs["total_nodes"] = int(node_offsets[-1])
        handle.attrs["total_edges"] = int(edge_offsets[-1])
        handle.attrs["extraction_seconds"] = float(time.perf_counter() - started)
        handle.attrs["build_complete"] = True
        handle.flush()

    print("Created:", LOCAL_H5)
    print("Size:", f"{LOCAL_H5.stat().st_size / 2**30:.2f} GiB")

# %% [markdown]
# ## 6. Build descriptor-tower features
#
# OMS remains a structural proxy: a metal atom is flagged when its soft
# coordination is at or below the element-specific 10th percentile. Pore-facing
# atoms are the lowest local-density quartile inside each unit cell.

# %%
POLAR_NONMETAL_Z = {
    atomic_numbers[symbol] for symbol in ("N", "O", "F", "P", "S", "Cl", "Br", "I")
}
C_Z = atomic_numbers["C"]
H_Z = atomic_numbers["H"]

with h5py.File(LOCAL_H5, "r") as handle:
    node_offsets = handle["graph/node_offsets"][:]
    edge_offsets = handle["graph/edge_offsets"][:]
    metal_cn_by_z = defaultdict(list)
    z_dataset = handle["node/z"]
    x_dataset = handle["node/x_cont"]
    metal_index = NODE_FEATURE_NAMES.index("metal_flag")
    soft_cn_index = NODE_FEATURE_NAMES.index("soft_coordination_number")
    for graph_index in tqdm(train_indices, desc="Train-only metal CN reference"):
        start, end = node_offsets[graph_index : graph_index + 2]
        z_graph = z_dataset[start:end]
        x_graph = x_dataset[start:end]
        metal_mask = x_graph[:, metal_index] > 0.5
        for z_value in np.unique(z_graph[metal_mask]):
            selected = metal_mask & (z_graph == z_value)
            metal_cn_by_z[int(z_value)].extend(
                x_graph[selected, soft_cn_index].astype(float).tolist()
            )

metal_cn_reference = {
    int(z_value): {
        "symbol": chemical_symbols[int(z_value)],
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "n_atoms": len(values),
    }
    for z_value, values in sorted(metal_cn_by_z.items())
}
all_train_metal_cn = [
    value for values in metal_cn_by_z.values() for value in values
]
if not all_train_metal_cn:
    raise RuntimeError("No metal atoms were found in the training split.")
metal_cn_fallback = {
    "symbol": "TRAIN_ALL_METALS",
    "p10": float(np.percentile(all_train_metal_cn, 10)),
    "median": float(np.median(all_train_metal_cn)),
    "n_atoms": len(all_train_metal_cn),
}
METAL_Z_VALUES = sorted(
    int(record["atomic_number"])
    for record in ELEMENT_PROPERTIES.values()
    if record["metal_flag"] > 0.5
)


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else 0.0


def descriptor_row(handle, graph_index):
    node_start, node_end = handle["graph/node_offsets"][graph_index : graph_index + 2]
    edge_start, edge_end = handle["graph/edge_offsets"][graph_index : graph_index + 2]
    z = handle["node/z"][node_start:node_end]
    x = handle["node/x_cont"][node_start:node_end]
    relation = handle["edge/relation"][edge_start:edge_end]
    score = handle["edge/bond_score"][edge_start:edge_end].astype(np.float32)
    src = handle["edge/src"][edge_start:edge_end].astype(np.int64)
    dst = handle["edge/dst"][edge_start:edge_end].astype(np.int64)

    is_metal = x[:, NODE_FEATURE_NAMES.index("metal_flag")] > 0.5
    soft_cn = x[:, NODE_FEATURE_NAMES.index("soft_coordination_number")]
    density_r3 = x[:, NODE_FEATURE_NAMES.index("local_density_r3")]
    density_r6 = x[:, NODE_FEATURE_NAMES.index("local_density_r6")]
    electronegativity = x[:, NODE_FEATURE_NAMES.index("electronegativity_pauling")]
    metal_indices = np.flatnonzero(is_metal)

    low_cn = np.zeros(len(z), dtype=bool)
    deficits = []
    for index in metal_indices:
        reference = metal_cn_reference.get(int(z[index]), metal_cn_fallback)
        low_cn[index] = soft_cn[index] <= reference["p10"]
        deficits.append(max(reference["median"] - float(soft_cn[index]), 0.0))

    polar_atom = np.isin(z, list(POLAR_NONMETAL_Z))
    ligand_mask = ~is_metal
    ligand_edges = relation == 1
    if ligand_edges.any():
        ligand_en_delta = np.abs(
            electronegativity[src[ligand_edges]]
            - electronegativity[dst[ligand_edges]]
        )
        ligand_en_difference = float(
            np.average(ligand_en_delta, weights=np.maximum(score[ligand_edges], 1.0e-6))
        )
    else:
        ligand_en_difference = 0.0

    pore_threshold = float(np.quantile(density_r3, 0.25))
    pore_mask = density_r3 <= pore_threshold
    heteroatom = (~is_metal) & (z != C_Z) & (z != H_Z)
    partial_mask = handle["node/partial_charge_mask"][node_start:node_end].astype(bool)

    row = {
        "mof_id": str(df.iloc[graph_index]["mof_id"]),
        "n_atoms_cif": len(z),
        "neighbor_backend": "alchemi_cell_list",
        "metal_atom_fraction": safe_mean(is_metal.astype(float)),
        "metal_species_count": len(set(z[is_metal].tolist())),
        "open_metal_site_proxy_count": int(low_cn.sum()),
        "open_metal_site_proxy_fraction": safe_mean(low_cn[metal_indices].astype(float)),
        "metal_coordination_mean": safe_mean(soft_cn[metal_indices]),
        "metal_coordination_deficit_mean": safe_mean(deficits),
        "ligand_polar_atom_fraction": safe_mean(polar_atom[ligand_mask].astype(float)),
        "ligand_bond_en_difference_mean": ligand_en_difference,
        "pore_proxy_polar_atom_fraction": safe_mean(polar_atom[pore_mask].astype(float)),
        "pore_proxy_heteroatom_fraction": safe_mean(heteroatom[pore_mask].astype(float)),
        "pore_proxy_metal_fraction": safe_mean(is_metal[pore_mask].astype(float)),
        "pore_proxy_electronegativity_mean": safe_mean(electronegativity[pore_mask]),
        "local_density_r3_mean": safe_mean(density_r3),
        "local_density_r3_std": float(np.std(density_r3)),
        "local_density_r6_mean": safe_mean(density_r6),
        "local_density_r6_std": float(np.std(density_r6)),
        "spatial_only_edge_fraction": safe_mean((relation == 0).astype(float)),
        "nonmetal_nonmetal_bond_fraction": safe_mean((relation == 1).astype(float)),
        "metal_nonmetal_bond_fraction": safe_mean((relation == 2).astype(float)),
        "metal_metal_bond_fraction": safe_mean((relation == 3).astype(float)),
        "mean_edge_bond_score": safe_mean(score),
        "partial_charge_available_fraction": safe_mean(partial_mask.astype(float)),
    }
    present_metals = set(z[is_metal].tolist())
    for metal_z in METAL_Z_VALUES:
        row[f"metal_type_{chemical_symbols[metal_z]}"] = int(metal_z in present_metals)
    return row


with h5py.File(LOCAL_H5, "r") as handle:
    descriptor_rows = [
        descriptor_row(handle, index)
        for index in tqdm(range(len(df)), desc="Descriptor features")
    ]

descriptor_df = pd.DataFrame(descriptor_rows)
if descriptor_df.drop(columns=["neighbor_backend"]).isna().any().any():
    raise RuntimeError("Unexpected missing value in descriptor features")
enriched_df = df.merge(descriptor_df, on="mof_id", how="left", validate="one_to_one")
LOCAL_ENRICHED_CSV = LOCAL_OUTPUT / ENRICHED_CSV_NAME
enriched_df.to_csv(LOCAL_ENRICHED_CSV, index=False)

METAL_REFERENCE_PATH = LOCAL_OUTPUT / "metal_soft_cn_reference.json"
METAL_REFERENCE_PATH.write_text(
    json.dumps(
        {
            "scope": "training split only",
            "by_atomic_number": metal_cn_reference,
            "fallback_all_training_metals": metal_cn_fallback,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("Descriptor columns added:", len(descriptor_df.columns) - 1)
display(descriptor_df.head())

# %% [markdown]
# ## 7. Verify HDF5 integrity and feature ranges

# %%
rng = np.random.default_rng(CFG.seed)
sample_indices = rng.choice(len(df), size=50, replace=False)
with h5py.File(LOCAL_H5, "r") as handle:
    assert int(handle.attrs["graph_count"]) == len(df)
    node_offsets = handle["graph/node_offsets"][:]
    edge_offsets = handle["graph/edge_offsets"][:]
    assert node_offsets[0] == 0 and edge_offsets[0] == 0
    assert np.all(np.diff(node_offsets) > 0)
    assert np.all(np.diff(edge_offsets) > 0)
    assert node_offsets[-1] == len(handle["node/z"])
    assert edge_offsets[-1] == len(handle["edge/src"])
    for index in sample_indices:
        ns, ne = node_offsets[index : index + 2]
        es, ee = edge_offsets[index : index + 2]
        n_nodes = int(ne - ns)
        src = handle["edge/src"][es:ee]
        dst = handle["edge/dst"][es:ee]
        distance = handle["edge/distance"][es:ee].astype(np.float32)
        vector = handle["edge/vector"][es:ee].astype(np.float32)
        assert src.max() < n_nodes and dst.max() < n_nodes
        assert np.isfinite(vector).all()
        assert np.max(np.abs(np.linalg.norm(vector, axis=1) - distance)) < 0.03
        assert distance.min() > 0 and distance.max() <= CFG.cutoff_angstrom + 0.03

    verification = {
        "graph_count": len(df),
        "total_nodes": int(node_offsets[-1]),
        "total_edges": int(edge_offsets[-1]),
        "node_feature_dim": len(NODE_FEATURE_NAMES),
        "descriptor_column_count": len(descriptor_df.columns) - 1,
        "rotation_distance_max_difference_angstrom": rotation_distance_error,
        "h5_size_gib": LOCAL_H5.stat().st_size / 2**30,
        "random_graphs_checked": len(sample_indices),
    }

manifest = {
    "dataset": "MOFX-DB paired CO2/CH4 50K ALCHEMI features",
    "config": asdict(CFG),
    "software": {
        "python": sys.version,
        "torch": torch.__version__,
        "nvalchemiops": str(nvalchemiops.__version__),
        "warp": str(wp.__version__),
        "ase": ase.__version__,
        "pymatgen": str(getattr(pymatgen, "__version__", "unknown")),
    },
    "node_feature_names": NODE_FEATURE_NAMES,
    "edge_features": [
        "periodic displacement vector",
        "distance",
        "periodic image",
        "relation class",
        "soft covalent-radius bond score",
    ],
    "edge_relation_labels": {
        0: "spatial_only",
        1: "nonmetal_nonmetal_bond_candidate",
        2: "metal_nonmetal_bond_candidate",
        3: "metal_metal_bond_candidate",
    },
    "bond_semantics": (
        "Bond scores are covalent-radius priors on ALCHEMI spatial edges, not formal bond orders."
    ),
    "partial_charge_status": (
        "No reliable charge labels were available. HDF5 charge values are NaN with zero masks; "
        "the rejected electronegativity charge proxy is not used."
    ),
    "descriptor_columns": descriptor_df.columns.tolist(),
    "metal_soft_cn_reference": metal_cn_reference,
    "metal_soft_cn_fallback": metal_cn_fallback,
    "metal_soft_cn_reference_scope": "training split only",
    "verification": verification,
    "outputs": {
        "h5": H5_NAME,
        "enriched_csv": ENRICHED_CSV_NAME,
        "element_property_table": ELEMENT_TABLE_NAME,
        "metal_soft_cn_reference": METAL_REFERENCE_PATH.name,
    },
}
LOCAL_MANIFEST = LOCAL_OUTPUT / MANIFEST_NAME
LOCAL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(verification, indent=2))

# %% [markdown]
# ## 8. Copy only completed outputs to Google Drive

# %%
def atomic_copy_to_drive(source: Path, destination: Path):
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"Copy size mismatch: {destination}")
    print(f"Saved: {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")


for source in (
    LOCAL_H5,
    LOCAL_ENRICHED_CSV,
    ELEMENT_TABLE_PATH,
    METAL_REFERENCE_PATH,
    LOCAL_MANIFEST,
):
    atomic_copy_to_drive(source, DRIVE_OUTPUT_DIR / source.name)

print("All verified outputs were saved to:", DRIVE_OUTPUT_DIR)

# %% [markdown]
# ## 9. Optional local cleanup
#
# Run only after confirming the five output files in Drive. This removes runtime
# copies but leaves Drive files untouched.

# %%
# shutil.rmtree(LOCAL_ROOT)
# print("Removed local runtime files.")
