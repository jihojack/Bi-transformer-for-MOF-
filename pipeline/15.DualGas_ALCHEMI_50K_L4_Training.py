# %% [markdown]
# # CO2 + CH4 50K ALCHEMI Dual-Tower Training (L4)
#
# Structure tower:
# periodic equivariant graph attention using atomic properties, ALCHEMI spatial
# edges, transparent soft bond scores, and local crystal descriptors.
#
# Descriptor tower:
# FT-Transformer-style scalar feature tokenization.
#
# The towers are gas-pressure-conditioned independently and interact only in the
# fusion transformer. CO2 and CH4 predictions are monotonic within each gas.

# %%
import gc
import json
import math
import os
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

try:
    from google.colab import drive

    drive.mount("/content/drive")
except ImportError:
    pass

warnings.filterwarnings("ignore")


@dataclass
class Config:
    seed: int = 42
    drive_data_dir: str = "/content/drive/MyDrive/mf_hid/4th"
    work_dir: str = "/content/dual_gas_50k_training"
    output_dir_name: str = "\uacb0\uacfc"
    h5_name: str = "periodic_graphs_50k_alchemi_compact.h5"
    csv_name: str = "MOF_CO2_CH4_50K_alchemi_enriched.csv"
    split_name: str = "MOF_CO2_CH4_50K_split_70_10_20.json"
    feature_manifest_name: str = "alchemi_50k_feature_manifest.json"
    expected_rows: int = 50_000
    cutoff_angstrom: float = 6.0
    max_atomic_number: int = 118
    n_rbf: int = 64
    relation_dim: int = 24
    d_model: int = 256
    vector_channels: int = 48
    num_heads: int = 8
    structure_layers: int = 6
    descriptor_layers: int = 4
    tower_pressure_layers: int = 2
    fusion_layers: int = 4
    structure_tokens: int = 8
    dropout: float = 0.10
    modality_dropout: float = 0.05
    max_nodes_per_batch: int = 4_200
    max_edges_per_batch: int = 130_000
    max_graphs_per_batch: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.8e-4
    weight_decay: float = 1.0e-4
    max_epochs: int = 80
    minimum_epochs: int = 25
    early_stopping_patience: int = 14
    early_stopping_min_delta: float = 5.0e-4
    warmup_epochs: int = 4
    auxiliary_loss_start: float = 0.12
    auxiliary_loss_end: float = 0.05
    auxiliary_decay_epochs: int = 10
    scaled_loss_weight: float = 0.20
    relative_loss_weight: float = 0.08
    relative_error_floor_mmol_g: float = 0.05
    num_workers: int = 4
    resume_training: bool = True


CFG = Config()
DRIVE_DATA_DIR = Path(CFG.drive_data_dir)
WORK_DIR = Path(CFG.work_dir)
OUTPUT_DIR = DRIVE_DATA_DIR / CFG.output_dir_name
LOCAL_H5 = WORK_DIR / CFG.h5_name
LOCAL_CSV = WORK_DIR / CFG.csv_name
LOCAL_SPLIT = WORK_DIR / CFG.split_name
SOURCE_H5 = DRIVE_DATA_DIR / CFG.h5_name
SOURCE_CSV = DRIVE_DATA_DIR / CFG.csv_name
SOURCE_SPLIT = DRIVE_DATA_DIR / CFG.split_name
SOURCE_FEATURE_MANIFEST = DRIVE_DATA_DIR / CFG.feature_manifest_name

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CO2_PRESSURES_BAR = np.array([0.01, 0.05, 0.1, 0.5, 2.5], dtype=np.float32)
CH4_PRESSURES_BAR = np.array([0.05, 0.5, 0.9, 2.5, 4.5], dtype=np.float32)
PRESSURES_BAR = np.concatenate([CO2_PRESSURES_BAR, CH4_PRESSURES_BAR])
GAS_NAMES = np.array(["CO2"] * 5 + ["CH4"] * 5)
GAS_INDEX = np.array([0] * 5 + [1] * 5, dtype=np.int64)
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
EXPECTED_NODE_FEATURE_NAMES = [
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
    "coordination_number",
    "soft_coordination_number",
    "local_density_r3",
    "local_density_r6",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


seed_everything(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Select an L4 GPU runtime before training.")

DEVICE = torch.device("cuda")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
)
print("AMP dtype:", AMP_DTYPE)
print("Results:", OUTPUT_DIR)

# %% [markdown]
# ## 1. Copy the compact dataset to the runtime and verify alignment

# %%
for source in (SOURCE_H5, SOURCE_CSV, SOURCE_SPLIT):
    if not source.exists():
        raise FileNotFoundError(source)

for source, destination in (
    (SOURCE_H5, LOCAL_H5),
    (SOURCE_CSV, LOCAL_CSV),
    (SOURCE_SPLIT, LOCAL_SPLIT),
):
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        print("Copying to runtime:", source.name)
        shutil.copy2(source, destination)

df = pd.read_csv(LOCAL_CSV)
if len(df) != CFG.expected_rows:
    raise ValueError(f"Expected {CFG.expected_rows:,} CSV rows, found {len(df):,}.")
if df["mof_id"].duplicated().any():
    raise ValueError("Duplicate mof_id values were found.")

with h5py.File(LOCAL_H5, "r") as handle:
    graph_count = int(handle.attrs["graph_count"])
    h5_ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in handle["graph/mof_id"][:]
    ]
    h5_row_indices = handle["graph/row_index"][:].astype(np.int64)
    node_offsets = handle["graph/node_offsets"][:].astype(np.int64)
    edge_offsets = handle["graph/edge_offsets"][:].astype(np.int64)
    h5_node_feature_names = json.loads(handle.attrs["node_feature_names_json"])
    has_bond_score = "edge/bond_score" in handle

if graph_count != CFG.expected_rows:
    raise ValueError(
        f"Expected {CFG.expected_rows:,} HDF5 graphs, found {graph_count:,}."
    )
if h5_node_feature_names != EXPECTED_NODE_FEATURE_NAMES:
    raise ValueError(
        "Unexpected HDF5 node-feature schema.\n"
        f"Expected: {EXPECTED_NODE_FEATURE_NAMES}\n"
        f"Found:    {h5_node_feature_names}"
    )
if not has_bond_score:
    raise ValueError("HDF5 is missing the required edge/bond_score dataset.")
if not np.array_equal(h5_row_indices, np.arange(CFG.expected_rows)):
    raise ValueError(f"HDF5 row_index must be ordered 0..{CFG.expected_rows - 1}.")
if h5_ids != df["mof_id"].astype(str).tolist():
    raise ValueError("HDF5 and CSV mof_id ordering does not match.")
if node_offsets[-1] <= 0 or edge_offsets[-1] <= 0:
    raise ValueError("The HDF5 node or edge arrays are empty.")

targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
if not np.isfinite(targets).all() or (targets < 0).any():
    raise ValueError("Targets contain invalid values.")
co2_monotonic_violations = np.any(
    np.diff(targets[:, : len(CO2_TARGET_COLUMNS)], axis=1) < -1.0e-8,
    axis=1,
)
ch4_monotonic_violations = np.any(
    np.diff(targets[:, len(CO2_TARGET_COLUMNS) :], axis=1) < -1.0e-8,
    axis=1,
)
NODE_FEATURE_NAMES = h5_node_feature_names
NODE_FEATURE_INDICES = np.arange(1, len(NODE_FEATURE_NAMES), dtype=np.int64)
USED_NODE_FEATURE_NAMES = [NODE_FEATURE_NAMES[i] for i in NODE_FEATURE_INDICES]

print("Rows:", len(df))
print("Nodes:", f"{node_offsets[-1]:,}")
print("Edges:", f"{edge_offsets[-1]:,}")
print("CO2 target monotonic violations:", int(co2_monotonic_violations.sum()))
print("CH4 target monotonic violations:", int(ch4_monotonic_violations.sum()))

# %% [markdown]
# ## 2. Load the fixed 70/10/20 split and preprocess descriptor features

# %%
NON_FEATURE_COLUMNS = {
    "mof_id",
    "elements",
    "has_cif",
    "neighbor_backend",
    "partial_charge_available_fraction",
    *TARGET_COLUMNS,
}
NON_FEATURE_PREFIXES = (
    "partial_charge_",
)

DESCRIPTOR_COLUMNS = [
    column
    for column in df.select_dtypes(include=[np.number, "bool"]).columns
    if column not in NON_FEATURE_COLUMNS
    and not any(column.startswith(prefix) for prefix in NON_FEATURE_PREFIXES)
    and column not in {"row_index"}
]
if not DESCRIPTOR_COLUMNS:
    raise ValueError("No numeric descriptor columns were selected.")

split_payload = json.loads(LOCAL_SPLIT.read_text(encoding="utf-8"))
train_indices = np.sort(np.asarray(split_payload["train_indices"], dtype=np.int64))
val_indices = np.sort(np.asarray(split_payload["val_indices"], dtype=np.int64))
test_indices = np.sort(np.asarray(split_payload["test_indices"], dtype=np.int64))

expected_split_sizes = (35_000, 5_000, 10_000)
actual_split_sizes = (len(train_indices), len(val_indices), len(test_indices))
if actual_split_sizes != expected_split_sizes:
    raise ValueError(
        f"Expected split sizes {expected_split_sizes}, found {actual_split_sizes}."
    )
all_split_indices = np.concatenate([train_indices, val_indices, test_indices])
if not np.array_equal(np.sort(all_split_indices), np.arange(CFG.expected_rows)):
    raise ValueError("Split indices must cover every row exactly once.")

csv_ids = df["mof_id"].astype(str).to_numpy()
for split_name, split_indices in (
    ("train", train_indices),
    ("val", val_indices),
    ("test", test_indices),
):
    payload_ids = np.asarray(split_payload[f"{split_name}_mof_ids"], dtype=str)
    if not np.array_equal(csv_ids[split_indices], payload_ids):
        raise ValueError(f"{split_name} split MOF IDs do not match CSV row indices.")

shutil.copy2(LOCAL_SPLIT, OUTPUT_DIR / "fixed_split_70_10_20.json")

descriptor_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
    ]
)
descriptor_raw = df[DESCRIPTOR_COLUMNS].replace([np.inf, -np.inf], np.nan)
descriptor_pipeline.fit(descriptor_raw.iloc[train_indices])
descriptor_all = descriptor_pipeline.transform(descriptor_raw).astype(np.float32)
joblib.dump(
    {
        "pipeline": descriptor_pipeline,
        "feature_columns": DESCRIPTOR_COLUMNS,
    },
    OUTPUT_DIR / "descriptor_preprocessing.joblib",
)

target_log = np.log1p(targets)
target_log_mean = target_log[train_indices].mean(axis=0).astype(np.float32)
target_log_std = target_log[train_indices].std(axis=0).clip(1.0e-4).astype(np.float32)
target_scale = targets[train_indices].std(axis=0).clip(0.05).astype(np.float32)

print(
    f"Split sizes: train={len(train_indices):,}, "
    f"validation={len(val_indices):,}, test={len(test_indices):,}"
)
print("Descriptor columns:", len(DESCRIPTOR_COLUMNS))
print("Transformed descriptor dimension:", descriptor_all.shape[1])
print("Excluded true partial-charge columns until PACMAN values are available.")

run_metadata = {
    "config": asdict(CFG),
    "target_columns": TARGET_COLUMNS,
    "gas_names": GAS_NAMES.tolist(),
    "pressures_bar": PRESSURES_BAR.tolist(),
    "node_feature_names": USED_NODE_FEATURE_NAMES,
    "descriptor_columns": DESCRIPTOR_COLUMNS,
    "split_sizes": {
        "train": len(train_indices),
        "validation": len(val_indices),
        "test": len(test_indices),
    },
}
(OUTPUT_DIR / "training_configuration.json").write_text(
    json.dumps(run_metadata, indent=2), encoding="utf-8"
)
if SOURCE_FEATURE_MANIFEST.exists():
    shutil.copy2(SOURCE_FEATURE_MANIFEST, OUTPUT_DIR / SOURCE_FEATURE_MANIFEST.name)

# %% [markdown]
# ## 3. Fit node-feature statistics on train graphs only

# %%
NODE_STATS_PATH = OUTPUT_DIR / "node_feature_preprocessing.npz"
node_stats_signature = (
    f"rows={len(df)}|first={h5_ids[0]}|last={h5_ids[-1]}|"
    f"train_n={len(train_indices)}|train_index_sum={int(train_indices.sum())}"
)

reuse_node_stats = False
if NODE_STATS_PATH.exists():
    node_stats = np.load(NODE_STATS_PATH, allow_pickle=False)
    cached_names = node_stats["names"].astype(str).tolist()
    cached_signature = (
        str(node_stats["dataset_signature"].item())
        if "dataset_signature" in node_stats.files
        else ""
    )
    reuse_node_stats = (
        cached_names == USED_NODE_FEATURE_NAMES
        and cached_signature == node_stats_signature
    )

if reuse_node_stats:
    node_mean = node_stats["mean"].astype(np.float32)
    node_std = node_stats["std"].astype(np.float32)
    print("Loaded compatible cached node preprocessing.")
else:
    feature_count = len(NODE_FEATURE_INDICES)
    sums = np.zeros(feature_count, dtype=np.float64)
    sums_sq = np.zeros(feature_count, dtype=np.float64)
    counts = np.zeros(feature_count, dtype=np.float64)
    with h5py.File(LOCAL_H5, "r") as handle:
        x_dataset = handle["node/x_cont"]
        mask_dataset = handle["node/x_cont_mask"]
        for graph_index in tqdm(train_indices, desc="Node statistics"):
            start, end = node_offsets[graph_index : graph_index + 2]
            values = x_dataset[start:end, NODE_FEATURE_INDICES].astype(np.float64)
            mask = mask_dataset[start:end, NODE_FEATURE_INDICES].astype(bool)
            sums += np.where(mask, values, 0.0).sum(axis=0)
            sums_sq += np.where(mask, values * values, 0.0).sum(axis=0)
            counts += mask.sum(axis=0)
    node_mean = (sums / np.maximum(counts, 1.0)).astype(np.float32)
    variance = sums_sq / np.maximum(counts, 1.0) - node_mean.astype(np.float64) ** 2
    node_std = np.sqrt(np.maximum(variance, 1.0e-8)).astype(np.float32)
    node_std[node_std < 1.0e-4] = 1.0
    np.savez(
        NODE_STATS_PATH,
        mean=node_mean,
        std=node_std,
        names=np.asarray(USED_NODE_FEATURE_NAMES),
        dataset_signature=np.asarray(node_stats_signature),
    )

print(
    pd.DataFrame(
        {
            "feature": USED_NODE_FEATURE_NAMES,
            "mean": node_mean,
            "std": node_std,
        }
    ).to_string(index=False)
)

# %% [markdown]
# ## 4. HDF5 random-access dataset and edge-budget batching

# %%
class EnrichedH5Dataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.h5_path = str(LOCAL_H5)
        self._handle = None

    def __len__(self):
        return len(self.indices)

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r", swmr=True)
        return self._handle

    def __getitem__(self, local_index):
        graph_index = int(self.indices[local_index])
        handle = self._file()
        node_start, node_end = node_offsets[graph_index : graph_index + 2]
        edge_start, edge_end = edge_offsets[graph_index : graph_index + 2]

        x = handle["node/x_cont"][node_start:node_end, NODE_FEATURE_INDICES].astype(
            np.float32
        )
        mask = handle["node/x_cont_mask"][
            node_start:node_end, NODE_FEATURE_INDICES
        ].astype(bool)
        x = np.where(mask, (x - node_mean) / node_std, 0.0).astype(np.float32)

        return {
            "graph_index": graph_index,
            "mof_id": h5_ids[graph_index],
            "z": torch.from_numpy(handle["node/z"][node_start:node_end]).long(),
            "x_cont": torch.from_numpy(x),
            "x_cont_mask": torch.from_numpy(mask),
            "src": torch.from_numpy(
                handle["edge/src"][edge_start:edge_end].astype(np.int64)
            ),
            "dst": torch.from_numpy(
                handle["edge/dst"][edge_start:edge_end].astype(np.int64)
            ),
            "edge_vec": torch.from_numpy(
                handle["edge/vector"][edge_start:edge_end].astype(np.float32)
            ),
            "edge_relation": torch.from_numpy(
                handle["edge/relation"][edge_start:edge_end].astype(np.int64)
            ),
            "edge_bond_score": torch.from_numpy(
                handle["edge/bond_score"][edge_start:edge_end].astype(np.float32)
            ),
            "descriptor": torch.from_numpy(descriptor_all[graph_index]),
            "target": torch.from_numpy(targets[graph_index]),
        }


class EdgeBudgetBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        max_nodes,
        max_edges,
        max_graphs,
        shuffle,
        seed,
    ):
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_graphs = max_graphs
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        global_indices = dataset.indices
        self.node_counts = np.diff(node_offsets)[global_indices]
        self.edge_counts = np.diff(edge_offsets)[global_indices]
        self._length = self._count_batches(np.arange(len(dataset)))

    def _ordered_indices(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)
        return indices

    def _batches(self, indices):
        batch, nodes, edges = [], 0, 0
        for index in indices:
            n_nodes = int(self.node_counts[index])
            n_edges = int(self.edge_counts[index])
            exceeds = batch and (
                nodes + n_nodes > self.max_nodes
                or edges + n_edges > self.max_edges
                or len(batch) >= self.max_graphs
            )
            if exceeds:
                yield batch
                batch, nodes, edges = [], 0, 0
            batch.append(int(index))
            nodes += n_nodes
            edges += n_edges
        if batch:
            yield batch

    def _count_batches(self, indices):
        return sum(1 for _ in self._batches(indices))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        yield from self._batches(self._ordered_indices())

    def __len__(self):
        return self._length


def collate_mofs(samples):
    output = {
        "mof_id": [],
        "graph_index": [],
        "z": [],
        "x_cont": [],
        "x_cont_mask": [],
        "src": [],
        "dst": [],
        "edge_vec": [],
        "edge_relation": [],
        "edge_bond_score": [],
        "batch": [],
        "descriptor": [],
        "target": [],
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
        output["batch"].append(torch.full((n_nodes,), graph_number, dtype=torch.long))
        output["descriptor"].append(sample["descriptor"])
        output["target"].append(sample["target"])
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
    output["target"] = torch.stack(output["target"])
    output["graph_index"] = torch.tensor(output["graph_index"], dtype=torch.long)
    output["n_graphs"] = len(samples)
    return output


def make_loader(indices, shuffle):
    dataset = EnrichedH5Dataset(indices)
    sampler = EdgeBudgetBatchSampler(
        dataset,
        max_nodes=CFG.max_nodes_per_batch,
        max_edges=CFG.max_edges_per_batch,
        max_graphs=CFG.max_graphs_per_batch,
        shuffle=shuffle,
        seed=CFG.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_mofs,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=CFG.num_workers > 0,
        prefetch_factor=2 if CFG.num_workers > 0 else None,
    )
    return dataset, sampler, loader


train_dataset, train_sampler, train_loader = make_loader(train_indices, True)
val_dataset, val_sampler, val_loader = make_loader(val_indices, False)
test_dataset, test_sampler, test_loader = make_loader(test_indices, False)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

# %% [markdown]
# ## 5. Enriched equivariant dual-tower graph transformer

# %%
def segment_softmax(scores, index, n_segments):
    original_dtype = scores.dtype
    stable_scores = scores.float()
    expanded_index = index.view(-1, *([1] * (scores.ndim - 1))).expand_as(
        stable_scores
    )
    max_values = torch.full(
        (n_segments, *scores.shape[1:]),
        -torch.inf,
        device=scores.device,
        dtype=torch.float32,
    )
    max_values.scatter_reduce_(
        0, expanded_index, stable_scores, reduce="amax", include_self=True
    )
    exponentials = torch.exp(stable_scores - max_values[index])
    denominators = torch.zeros_like(max_values)
    denominators.index_add_(0, index, exponentials)
    return (exponentials / denominators[index].clamp_min(1.0e-12)).to(
        original_dtype
    )


class GaussianRBF(nn.Module):
    def __init__(self, cutoff, n_rbf):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = float(n_rbf / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, distances):
        rbf = torch.exp(
            -self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2
        )
        envelope = 0.5 * (
            torch.cos(math.pi * distances / self.cutoff) + 1.0
        )
        envelope = envelope * (distances < self.cutoff)
        return rbf * envelope.unsqueeze(-1)


def vector_linear(vectors, layer):
    return layer(vectors.transpose(1, 2)).transpose(1, 2)


class EnrichedEquivariantAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        vector_channels,
        num_heads,
        edge_dim,
        dropout,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.vector_channels = vector_channels
        self.scalar_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.scalar_value = nn.Linear(d_model, d_model, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_heads),
        )
        self.edge_scalar_value = nn.Linear(edge_dim, d_model, bias=False)
        self.direction_from_scalar = nn.Linear(
            d_model, vector_channels, bias=False
        )
        self.direction_from_edge = nn.Sequential(
            nn.Linear(edge_dim, vector_channels),
            nn.SiLU(),
            nn.Linear(vector_channels, vector_channels),
        )
        self.vector_value = nn.Linear(
            vector_channels, vector_channels, bias=False
        )
        self.scalar_out = nn.Linear(d_model, d_model)
        self.vector_out = nn.Linear(
            vector_channels, vector_channels, bias=False
        )
        self.update = nn.Sequential(
            nn.Linear(d_model + vector_channels, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.vector_gate = nn.Linear(d_model, vector_channels)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, scalar, vector, src, dst, unit_vectors, edge_context):
        n_nodes = scalar.shape[0]
        normalized = self.scalar_norm(scalar)
        query = self.query(normalized[dst]).view(
            -1, self.num_heads, self.head_dim
        )
        key = self.key(normalized[src]).view(
            -1, self.num_heads, self.head_dim
        )
        scores = (query * key).sum(-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_context)
        attention = segment_softmax(scores, dst, n_nodes)

        scalar_value = (
            self.scalar_value(normalized[src])
            + self.edge_scalar_value(edge_context)
        ).view(-1, self.num_heads, self.head_dim)
        scalar_message = (attention.unsqueeze(-1) * scalar_value).reshape(
            -1, self.num_heads * self.head_dim
        )
        scalar_aggregate = torch.zeros(
            n_nodes,
            scalar_message.shape[-1],
            device=scalar.device,
            dtype=scalar_message.dtype,
        )
        scalar_aggregate.index_add_(0, dst, scalar_message)

        neighbor_vector = vector_linear(vector[src], self.vector_value)
        direction_coefficient = self.direction_from_scalar(
            normalized[src]
        ) * self.direction_from_edge(edge_context)
        vector_message = neighbor_vector + (
            direction_coefficient.unsqueeze(-1) * unit_vectors.unsqueeze(1)
        )
        vector_message = vector_message * attention.mean(-1)[:, None, None]
        vector_aggregate = torch.zeros(
            n_nodes,
            self.vector_channels,
            3,
            device=vector.device,
            dtype=vector_message.dtype,
        )
        vector_aggregate.index_add_(0, dst, vector_message)

        scalar_context = scalar + self.dropout(
            self.scalar_out(scalar_aggregate)
        )
        vector_context = vector + vector_linear(
            vector_aggregate, self.vector_out
        )
        vector_norm = torch.sqrt(
            vector_context.float().pow(2).sum(-1) + 1.0e-8
        ).to(vector_context.dtype)
        scalar_update = self.update(
            torch.cat([scalar_context, vector_norm], dim=-1)
        )
        scalar = self.output_norm(scalar_context + scalar_update)
        vector = vector_context * torch.sigmoid(
            self.vector_gate(scalar)
        ).unsqueeze(-1)
        return scalar, vector


class InvariantTokenPool(nn.Module):
    def __init__(self, d_model, n_tokens):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, scalar, batch, n_graphs):
        keys = self.key(scalar)
        scores = torch.einsum("nd,kd->nk", keys, self.queries)
        scores = scores / math.sqrt(keys.shape[-1])
        weights = segment_softmax(scores, batch, n_graphs)
        weighted = weights.unsqueeze(-1) * self.value(scalar).unsqueeze(1)
        output = torch.zeros(
            n_graphs,
            self.queries.shape[0],
            scalar.shape[-1],
            device=scalar.device,
            dtype=weighted.dtype,
        )
        output.index_add_(0, batch, weighted)
        return output


class AlchemiStructureTower(nn.Module):
    def __init__(self, config, node_feature_dim):
        super().__init__()
        self.atom_embedding = nn.Embedding(
            config.max_atomic_number + 1, config.d_model
        )
        self.node_projector = nn.Sequential(
            nn.Linear(2 * node_feature_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.rbf = GaussianRBF(config.cutoff_angstrom, config.n_rbf)
        self.relation_embedding = nn.Embedding(4, config.relation_dim)
        edge_input_dim = config.n_rbf + config.relation_dim + 1
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_input_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                EnrichedEquivariantAttentionBlock(
                    config.d_model,
                    config.vector_channels,
                    config.num_heads,
                    config.d_model,
                    config.dropout,
                )
                for _ in range(config.structure_layers)
            ]
        )
        self.pool = InvariantTokenPool(
            config.d_model, config.structure_tokens
        )
        self.vector_channels = config.vector_channels

    def forward(self, batch):
        node_input = torch.cat(
            [batch["x_cont"], batch["x_cont_mask"].to(batch["x_cont"].dtype)],
            dim=-1,
        )
        scalar = self.atom_embedding(batch["z"]) + self.node_projector(node_input)
        vector = torch.zeros(
            len(batch["z"]),
            self.vector_channels,
            3,
            device=scalar.device,
            dtype=scalar.dtype,
        )
        distances = batch["edge_vec"].float().norm(dim=-1).clamp_min(1.0e-8)
        unit_vectors = (
            batch["edge_vec"].float() / distances.unsqueeze(-1)
        ).to(scalar.dtype)
        rbf = self.rbf(distances).to(scalar.dtype)
        relation = self.relation_embedding(batch["edge_relation"])
        edge_input = torch.cat(
            [
                rbf,
                relation,
                batch["edge_bond_score"].unsqueeze(-1).to(rbf.dtype),
            ],
            dim=-1,
        )
        edge_context = self.edge_projector(edge_input)
        for block in self.blocks:
            scalar, vector = block(
                scalar,
                vector,
                batch["src"],
                batch["dst"],
                unit_vectors,
                edge_context,
            )
        return self.pool(scalar, batch["batch"], batch["n_graphs"])


class DescriptorTower(nn.Module):
    def __init__(self, n_features, config):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, config.d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, config.d_model))
        self.feature_embedding = nn.Parameter(
            torch.randn(n_features, config.d_model) * 0.02
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.descriptor_layers
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, descriptor):
        tokens = (
            descriptor.unsqueeze(-1) * self.weight.unsqueeze(0)
            + self.bias.unsqueeze(0)
            + self.feature_embedding.unsqueeze(0)
        )
        cls = self.cls_token.expand(len(descriptor), -1, -1)
        return self.norm(self.encoder(torch.cat([cls, tokens], dim=1)))


class PressureAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.tower_pressure_layers
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, pressure_tokens, tower_tokens):
        combined = torch.cat([pressure_tokens, tower_tokens], dim=1)
        encoded = self.norm(self.encoder(combined))
        return encoded[:, : pressure_tokens.shape[1]]


def positive_monotonic_curve(raw_values):
    log_curve = torch.cumsum(F.softplus(raw_values), dim=-1)
    return torch.expm1(log_curve.clamp_max(6.0))


def positive_monotonic_dual_gas(raw_values):
    split = len(CO2_TARGET_COLUMNS)
    return torch.cat(
        [
            positive_monotonic_curve(raw_values[:, :split]),
            positive_monotonic_curve(raw_values[:, split:]),
        ],
        dim=-1,
    )


class DualGasAlchemiModel(nn.Module):
    def __init__(self, n_descriptor_features, config):
        super().__init__()
        self.config = config
        self.structure_tower = AlchemiStructureTower(
            config, len(NODE_FEATURE_INDICES)
        )
        self.descriptor_tower = DescriptorTower(
            n_descriptor_features, config
        )

        log_pressure = torch.log10(torch.as_tensor(PRESSURES_BAR).float())
        pressure_values = (log_pressure - log_pressure.mean()) / log_pressure.std()
        self.register_buffer("pressure_values", pressure_values)
        self.register_buffer("gas_index", torch.as_tensor(GAS_INDEX).long())

        # Molecular weight, kinetic diameter, polarizability, quadrupole flag,
        # critical temperature, and acentric factor for CO2 and CH4.
        gas_properties = torch.tensor(
            [
                [44.0095, 3.30, 2.507, 1.0, 304.13, 0.224],
                [16.0430, 3.80, 2.593, 0.0, 190.56, 0.011],
            ],
            dtype=torch.float32,
        )
        gas_properties = (gas_properties - gas_properties.mean(0)) / gas_properties.std(
            0, unbiased=False
        ).clamp_min(1.0e-6)
        self.register_buffer("gas_properties", gas_properties)
        self.pressure_mlp = nn.Sequential(
            nn.Linear(5, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.gas_property_mlp = nn.Sequential(
            nn.Linear(gas_properties.shape[1], config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.gas_embedding = nn.Embedding(2, config.d_model)
        self.condition_embedding = nn.Parameter(
            torch.randn(len(PRESSURES_BAR), config.d_model) * 0.02
        )
        self.structure_adapter = PressureAdapter(config)
        self.descriptor_adapter = PressureAdapter(config)

        self.fusion_queries = nn.Parameter(
            torch.randn(len(PRESSURES_BAR), config.d_model) * 0.02
        )
        self.type_embedding = nn.Parameter(
            torch.randn(3, config.d_model) * 0.02
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=config.fusion_layers
        )
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.fusion_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.descriptor_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        for head in (
            self.fusion_head,
            self.structure_head,
            self.descriptor_head,
        ):
            nn.init.constant_(head[-1].bias, -2.5)

    def forward(self, batch):
        structure_tokens = self.structure_tower(batch)
        descriptor_tokens = self.descriptor_tower(batch["descriptor"])
        batch_size = batch["n_graphs"]
        pressure = self.pressure_values
        pressure_features = torch.stack(
            [
                pressure,
                torch.sin(math.pi * pressure),
                torch.cos(math.pi * pressure),
                torch.sin(2.0 * math.pi * pressure),
                torch.cos(2.0 * math.pi * pressure),
            ],
            dim=-1,
        )
        condition_tokens = (
            self.pressure_mlp(pressure_features)
            + self.gas_property_mlp(self.gas_properties[self.gas_index])
            + self.gas_embedding(self.gas_index)
            + self.condition_embedding
        ).unsqueeze(0).expand(batch_size, -1, -1)

        structure_pressure = self.structure_adapter(
            condition_tokens, structure_tokens
        )
        descriptor_pressure = self.descriptor_adapter(
            condition_tokens, descriptor_tokens
        )

        structure_aux = positive_monotonic_dual_gas(
            self.structure_head(structure_pressure).squeeze(-1)
        )
        descriptor_aux = positive_monotonic_dual_gas(
            self.descriptor_head(descriptor_pressure).squeeze(-1)
        )

        if self.training and self.config.modality_dropout > 0:
            choices = torch.rand(batch_size, device=condition_tokens.device)
            drop_structure = choices < self.config.modality_dropout / 2
            drop_descriptor = (
                choices >= self.config.modality_dropout / 2
            ) & (choices < self.config.modality_dropout)
            structure_pressure = structure_pressure.masked_fill(
                drop_structure[:, None, None], 0.0
            )
            descriptor_pressure = descriptor_pressure.masked_fill(
                drop_descriptor[:, None, None], 0.0
            )

        queries = self.fusion_queries.unsqueeze(0).expand(batch_size, -1, -1)
        fusion_input = torch.cat(
            [
                queries + condition_tokens + self.type_embedding[0],
                structure_pressure + self.type_embedding[1],
                descriptor_pressure + self.type_embedding[2],
            ],
            dim=1,
        )
        fused = self.fusion_norm(self.fusion(fusion_input))
        fusion_prediction = positive_monotonic_dual_gas(
            self.fusion_head(fused[:, : len(PRESSURES_BAR)]).squeeze(-1)
        )
        return {
            "fusion": fusion_prediction,
            "structure": structure_aux,
            "descriptor": descriptor_aux,
        }


model = DualGasAlchemiModel(descriptor_all.shape[1], CFG).to(DEVICE)
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print(f"Trainable parameters: {trainable_parameters:,}")

# %% [markdown]
# ## 6. One-batch forward/backward smoke test

# %%
def move_batch_to_device(batch):
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(DEVICE, non_blocking=True)
        else:
            output[key] = value
    return output


TARGET_LOG_MEAN = torch.as_tensor(target_log_mean, device=DEVICE)
TARGET_LOG_STD = torch.as_tensor(target_log_std, device=DEVICE)
TARGET_SCALE = torch.as_tensor(target_scale, device=DEVICE)


def adsorption_loss(prediction, target):
    prediction_float = prediction.float()
    target_float = target.float()
    prediction_log_z = (
        torch.log1p(prediction_float) - TARGET_LOG_MEAN
    ) / TARGET_LOG_STD
    target_log_z = (
        torch.log1p(target_float) - TARGET_LOG_MEAN
    ) / TARGET_LOG_STD
    log_loss = F.smooth_l1_loss(
        prediction_log_z, target_log_z, beta=0.20
    )
    scaled_loss = F.smooth_l1_loss(
        prediction_float / TARGET_SCALE,
        target_float / TARGET_SCALE,
        beta=0.20,
    )
    relative_error = (
        (prediction_float - target_float).abs()
        / (target_float + CFG.relative_error_floor_mmol_g)
    ).clamp_max(5.0).mean()
    return (
        (1.0 - CFG.scaled_loss_weight - CFG.relative_loss_weight) * log_loss
        + CFG.scaled_loss_weight * scaled_loss
        + CFG.relative_loss_weight * relative_error
    )


smoke_batch = move_batch_to_device(next(iter(train_loader)))
model.train()
with torch.autocast("cuda", dtype=AMP_DTYPE):
    smoke_output = model(smoke_batch)
    smoke_loss = adsorption_loss(smoke_output["fusion"], smoke_batch["target"])
smoke_loss.backward()
model.zero_grad(set_to_none=True)
print("Smoke batch graphs:", smoke_batch["n_graphs"])
print("Smoke batch nodes:", len(smoke_batch["z"]))
print("Smoke batch edges:", len(smoke_batch["src"]))
print("Smoke prediction shape:", tuple(smoke_output["fusion"].shape))
print("Smoke loss:", float(smoke_loss))
del smoke_batch, smoke_output, smoke_loss
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Mixed-precision training, resume, and early stopping

# %%
@torch.no_grad()
def evaluate_loader(loader, include_towers=False):
    model.eval()
    predictions = {"fusion": [], "structure": [], "descriptor": []}
    observations = []
    graph_indices = []
    mof_ids = []
    total_loss, total_samples = 0.0, 0
    for batch in tqdm(loader, desc="Evaluation", leave=False):
        mof_ids.extend(batch["mof_id"])
        graph_indices.extend(batch["graph_index"].numpy().tolist())
        batch = move_batch_to_device(batch)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            output = model(batch)
            loss = adsorption_loss(output["fusion"], batch["target"])
        batch_size = batch["n_graphs"]
        total_loss += float(loss) * batch_size
        total_samples += batch_size
        observations.append(batch["target"].float().cpu().numpy())
        for tower in predictions:
            if include_towers or tower == "fusion":
                predictions[tower].append(output[tower].float().cpu().numpy())

    observations = np.concatenate(observations)
    for tower in predictions:
        if predictions[tower]:
            predictions[tower] = np.concatenate(predictions[tower])
    fusion = predictions["fusion"]
    pressure_rmse = np.sqrt(np.mean((observations - fusion) ** 2, axis=0))
    normalized_rmse = pressure_rmse / target_scale
    gas_split = len(CO2_TARGET_COLUMNS)
    return {
        "loss": total_loss / total_samples,
        "rmse": float(np.sqrt(np.mean((observations - fusion) ** 2))),
        "mean_pressure_rmse": float(pressure_rmse.mean()),
        "mean_normalized_rmse": float(normalized_rmse.mean()),
        "co2_mean_normalized_rmse": float(normalized_rmse[:gas_split].mean()),
        "ch4_mean_normalized_rmse": float(normalized_rmse[gas_split:].mean()),
        "pressure_rmse": pressure_rmse,
        "predictions": predictions,
        "observations": observations,
        "graph_indices": np.asarray(graph_indices),
        "mof_ids": mof_ids,
    }


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    fused=True,
)
steps_per_epoch = math.ceil(
    len(train_loader) / CFG.gradient_accumulation_steps
)
total_steps = CFG.max_epochs * steps_per_epoch
warmup_steps = CFG.warmup_epochs * steps_per_epoch


def lr_lambda(step):
    if step < warmup_steps:
        return max(step + 1, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
use_fp16_scaler = AMP_DTYPE == torch.float16
gradient_scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)

best_checkpoint_path = OUTPUT_DIR / "best_dual_gas_alchemi_50k.pt"
last_checkpoint_path = OUTPUT_DIR / "last_dual_gas_alchemi_50k_checkpoint.pt"
history = []
best_val_score = float("inf")
early_stopping_reference = float("inf")
epochs_without_improvement = 0
start_epoch = 1
global_step = 0

if CFG.resume_training and last_checkpoint_path.exists():
    checkpoint = torch.load(
        last_checkpoint_path, map_location=DEVICE, weights_only=False
    )
    if checkpoint.get("descriptor_columns") != DESCRIPTOR_COLUMNS:
        raise RuntimeError("Checkpoint descriptor columns do not match.")
    if checkpoint.get("node_feature_names") != USED_NODE_FEATURE_NAMES:
        raise RuntimeError("Checkpoint node-feature schema does not match.")
    if checkpoint.get("target_columns") != TARGET_COLUMNS:
        raise RuntimeError("Checkpoint target schema does not match.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if use_fp16_scaler and checkpoint.get("scaler_state_dict"):
        gradient_scaler.load_state_dict(checkpoint["scaler_state_dict"])
    history = checkpoint.get("history", [])
    best_val_score = float(checkpoint["best_val_score"])
    early_stopping_reference = float(
        checkpoint.get("early_stopping_reference", best_val_score)
    )
    epochs_without_improvement = int(
        checkpoint["epochs_without_improvement"]
    )
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint.get("global_step", 0))
    print("Resuming from epoch:", start_epoch)

for epoch in range(start_epoch, CFG.max_epochs + 1):
    epoch_start = time.time()
    train_sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, seen_samples = 0.0, 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
    auxiliary_progress = min(
        max(epoch - 1, 0) / max(CFG.auxiliary_decay_epochs, 1), 1.0
    )
    auxiliary_weight = (
        CFG.auxiliary_loss_start
        + auxiliary_progress * (CFG.auxiliary_loss_end - CFG.auxiliary_loss_start)
    )

    for batch_number, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            output = model(batch)
            fusion_loss = adsorption_loss(output["fusion"], batch["target"])
            structure_loss = adsorption_loss(
                output["structure"], batch["target"]
            )
            descriptor_loss = adsorption_loss(
                output["descriptor"], batch["target"]
            )
            loss = fusion_loss + auxiliary_weight * (
                structure_loss + descriptor_loss
            )
            backward_loss = loss / CFG.gradient_accumulation_steps

        gradient_scaler.scale(backward_loss).backward()
        should_step = (
            batch_number % CFG.gradient_accumulation_steps == 0
            or batch_number == len(train_loader)
        )
        if should_step:
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        batch_size = batch["n_graphs"]
        running_loss += float(loss.detach()) * batch_size
        seen_samples += batch_size
        progress.set_postfix(loss=f"{running_loss / seen_samples:.4f}")

    train_loss = running_loss / seen_samples
    validation = evaluate_loader(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]
    epoch_seconds = time.time() - epoch_start
    val_score = validation["mean_normalized_rmse"]
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_rmse_mmol_g": validation["rmse"],
            "val_mean_pressure_rmse_mmol_g": validation[
                "mean_pressure_rmse"
            ],
            "val_mean_normalized_rmse": val_score,
            "val_co2_mean_normalized_rmse": validation[
                "co2_mean_normalized_rmse"
            ],
            "val_ch4_mean_normalized_rmse": validation[
                "ch4_mean_normalized_rmse"
            ],
            "auxiliary_loss_weight": auxiliary_weight,
            "learning_rate": current_lr,
            "seconds": epoch_seconds,
        }
    )
    print(
        f"Epoch {epoch:03d} | train {train_loss:.5f} | "
        f"val RMSE {validation['rmse']:.5f} mmol/g | "
        f"norm all/CO2/CH4 {val_score:.5f}/"
        f"{validation['co2_mean_normalized_rmse']:.5f}/"
        f"{validation['ch4_mean_normalized_rmse']:.5f} | "
        f"lr {current_lr:.2e} | "
        f"{epoch_seconds:.1f} s"
    )

    improved = val_score < best_val_score - 1.0e-6
    if improved:
        best_val_score = val_score
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(CFG),
                "descriptor_columns": DESCRIPTOR_COLUMNS,
                "descriptor_dim": descriptor_all.shape[1],
                "node_feature_names": USED_NODE_FEATURE_NAMES,
                "target_columns": TARGET_COLUMNS,
                "gas_names": GAS_NAMES.tolist(),
                "target_log_mean": target_log_mean,
                "target_log_std": target_log_std,
                "target_scale": target_scale,
                "pressures_bar": PRESSURES_BAR,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "best_val_score": best_val_score,
                "epoch": epoch,
            },
            best_checkpoint_path,
        )
    significant_improvement = (
        val_score < early_stopping_reference - CFG.early_stopping_min_delta
    )
    if significant_improvement:
        early_stopping_reference = val_score
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": gradient_scaler.state_dict(),
            "config": asdict(CFG),
            "descriptor_columns": DESCRIPTOR_COLUMNS,
            "node_feature_names": USED_NODE_FEATURE_NAMES,
            "target_columns": TARGET_COLUMNS,
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "best_val_score": best_val_score,
            "early_stopping_reference": early_stopping_reference,
            "epochs_without_improvement": epochs_without_improvement,
        },
        last_checkpoint_path,
    )
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )

    if (
        epoch >= CFG.minimum_epochs
        and epochs_without_improvement >= CFG.early_stopping_patience
    ):
        print("Early stopping triggered.")
        break

print("Best validation mean normalized RMSE:", best_val_score)

# %% [markdown]
# ## 8. Training curves

# %%
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
axes[0].set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend()
axes[1].plot(history_df["epoch"], history_df["val_rmse_mmol_g"])
axes[1].set(
    title="Validation RMSE",
    xlabel="Epoch",
    ylabel="RMSE (mmol g$^{-1}$)",
)
axes[2].plot(
    history_df["epoch"], history_df["val_mean_normalized_rmse"], label="All"
)
axes[2].plot(
    history_df["epoch"],
    history_df["val_co2_mean_normalized_rmse"],
    label="CO2",
)
axes[2].plot(
    history_df["epoch"],
    history_df["val_ch4_mean_normalized_rmse"],
    label="CH4",
)
axes[2].set(
    title="Mean Pressure-Normalized RMSE",
    xlabel="Epoch",
    ylabel="Normalized RMSE",
)
axes[2].legend()
for axis in axes:
    axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_history.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Independent test evaluation and tower ablation

# %%
best_checkpoint = torch.load(
    best_checkpoint_path, map_location=DEVICE, weights_only=False
)
model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()
test_result = evaluate_loader(test_loader, include_towers=True)


CONDITIONS = [
    {
        "index": index,
        "gas": str(GAS_NAMES[index]),
        "pressure_bar": float(PRESSURES_BAR[index]),
        "target_column": TARGET_COLUMNS[index],
    }
    for index in range(len(TARGET_COLUMNS))
]


def calculate_metric_row(y_true, y_pred, tower, gas, pressure_bar, scope):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    absolute_error = np.abs(y_true - y_pred)
    smape_denominator = np.abs(y_true) + np.abs(y_pred) + 1.0e-8
    eligible = y_true >= CFG.relative_error_floor_mmol_g
    ape = 100.0 * absolute_error[eligible] / np.maximum(y_true[eligible], 1.0e-8)
    return {
        "tower": tower,
        "gas": gas,
        "pressure_bar": pressure_bar,
        "scope": scope,
        "r2": r2_score(y_true, y_pred),
        "rmse_mmol_g": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae_mmol_g": mean_absolute_error(y_true, y_pred),
        "wape_pct": 100.0 * absolute_error.sum() / np.maximum(y_true.sum(), 1.0e-8),
        "smape_pct": 200.0 * np.mean(absolute_error / smape_denominator),
        "median_ape_pct_q_ge_0p05": np.median(ape) if len(ape) else np.nan,
        "mean_ape_pct_q_ge_0p05": np.mean(ape) if len(ape) else np.nan,
        "p90_ape_pct_q_ge_0p05": np.percentile(ape, 90) if len(ape) else np.nan,
        "within_10pct_q_ge_0p05_pct": 100.0 * np.mean(ape <= 10.0) if len(ape) else np.nan,
        "within_20pct_q_ge_0p05_pct": 100.0 * np.mean(ape <= 20.0) if len(ape) else np.nan,
        "within_50pct_q_ge_0p05_pct": 100.0 * np.mean(ape <= 50.0) if len(ape) else np.nan,
        "underprediction_rate_pct": 100.0 * np.mean(y_pred < y_true),
        "n_test": len(y_true),
    }


def metric_table(observed, predicted, tower):
    rows = []
    for condition in CONDITIONS:
        index = condition["index"]
        rows.append(
            calculate_metric_row(
                observed[:, index],
                predicted[:, index],
                tower,
                condition["gas"],
                condition["pressure_bar"],
                "condition",
            )
        )
    for gas, indices in (("CO2", slice(0, 5)), ("CH4", slice(5, 10))):
        rows.append(
            calculate_metric_row(
                observed[:, indices],
                predicted[:, indices],
                tower,
                gas,
                np.nan,
                "gas_flattened",
            )
        )
    rows.append(
        calculate_metric_row(
            observed,
            predicted,
            tower,
            "ALL",
            np.nan,
            "overall_flattened",
        )
    )
    return rows


metric_rows = []
for tower, prediction in test_result["predictions"].items():
    metric_rows.extend(
        metric_table(test_result["observations"], prediction, tower)
    )
metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(OUTPUT_DIR / "test_metrics_all_towers.csv", index=False)
display(metrics_df)

recovery_rows = []
for tower, prediction in test_result["predictions"].items():
    for condition in CONDITIONS:
        pressure_index = condition["index"]
        y_true = test_result["observations"][:, pressure_index]
        y_pred = prediction[:, pressure_index]
        for fraction in (0.01, 0.05, 0.10):
            k = max(1, int(math.ceil(len(y_true) * fraction)))
            true_top = set(np.argpartition(y_true, -k)[-k:])
            predicted_top = set(np.argpartition(y_pred, -k)[-k:])
            recovery_rows.append(
                {
                    "tower": tower,
                    "gas": condition["gas"],
                    "pressure_bar": condition["pressure_bar"],
                    "true_top_fraction": fraction,
                    "k": k,
                    "recall_at_k": len(true_top & predicted_top) / k,
                }
            )
recovery_df = pd.DataFrame(recovery_rows)
recovery_df.to_csv(
    OUTPUT_DIR / "high_uptake_candidate_recovery.csv", index=False
)
display(recovery_df[recovery_df["tower"] == "fusion"])

monotonic_rows = []
for tower, prediction in test_result["predictions"].items():
    for gas, indices in (("CO2", slice(0, 5)), ("CH4", slice(5, 10))):
        violations = np.any(np.diff(prediction[:, indices], axis=1) < -1.0e-7, axis=1)
        monotonic_rows.append(
            {
                "tower": tower,
                "gas": gas,
                "violation_count": int(violations.sum()),
                "violation_rate_pct": 100.0 * float(violations.mean()),
            }
        )
pd.DataFrame(monotonic_rows).to_csv(
    OUTPUT_DIR / "prediction_monotonicity.csv", index=False
)

prediction_table = pd.DataFrame(
    {
        "graph_index": test_result["graph_indices"],
        "mof_id": test_result["mof_ids"],
    }
)
for condition in CONDITIONS:
    pressure_index = condition["index"]
    label = str(condition["pressure_bar"]).replace(".", "p")
    prefix = f"{condition['gas'].lower()}_{label}bar"
    prediction_table[f"observed_{prefix}_mmol_g"] = test_result[
        "observations"
    ][:, pressure_index]
    for tower, prediction in test_result["predictions"].items():
        prediction_table[f"{tower}_{prefix}_mmol_g"] = prediction[
            :, pressure_index
        ]
prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

error_decile_rows = []
relative_error_rows = []
fusion_for_errors = test_result["predictions"]["fusion"]
for condition in CONDITIONS:
    index = condition["index"]
    y_true = test_result["observations"][:, index]
    y_pred = fusion_for_errors[:, index]
    absolute_error = np.abs(y_true - y_pred)
    deciles = pd.qcut(
        pd.Series(y_true).rank(method="first"), q=10, labels=False
    ).to_numpy()
    for decile in range(10):
        mask = deciles == decile
        error_decile_rows.append(
            {
                "gas": condition["gas"],
                "pressure_bar": condition["pressure_bar"],
                "measured_uptake_decile": decile + 1,
                "n": int(mask.sum()),
                "mean_observed_mmol_g": float(y_true[mask].mean()),
                "mae_mmol_g": float(absolute_error[mask].mean()),
                "median_ape_pct_q_ge_0p05": float(
                    np.median(
                        100.0
                        * absolute_error[mask & (y_true >= 0.05)]
                        / y_true[mask & (y_true >= 0.05)]
                    )
                )
                if np.any(mask & (y_true >= 0.05))
                else np.nan,
            }
        )
    eligible_indices = np.flatnonzero(y_true >= 0.05)
    for sample_index in eligible_indices:
        relative_error_rows.append(
            {
                "mof_id": test_result["mof_ids"][sample_index],
                "gas": condition["gas"],
                "pressure_bar": condition["pressure_bar"],
                "observed_mmol_g": float(y_true[sample_index]),
                "predicted_mmol_g": float(y_pred[sample_index]),
                "absolute_error_mmol_g": float(absolute_error[sample_index]),
                "ape_pct": float(100.0 * absolute_error[sample_index] / y_true[sample_index]),
            }
        )

error_decile_df = pd.DataFrame(error_decile_rows)
error_decile_df.to_csv(OUTPUT_DIR / "error_by_uptake_decile.csv", index=False)
largest_relative_errors_df = (
    pd.DataFrame(relative_error_rows)
    .sort_values("ape_pct", ascending=False)
    .head(200)
)
largest_relative_errors_df.to_csv(
    OUTPUT_DIR / "largest_relative_errors_q_ge_0p05.csv", index=False
)

# %% [markdown]
# ## 10. Scatter plots and representative isotherms

# %%
fusion_prediction = test_result["predictions"]["fusion"]
observations = test_result["observations"]

for gas, gas_pressures, condition_slice, color in (
    ("CO2", CO2_PRESSURES_BAR, slice(0, 5), "#3B6FB6"),
    ("CH4", CH4_PRESSURES_BAR, slice(5, 10), "#238B57"),
):
    condition_indices = np.arange(len(PRESSURES_BAR))[condition_slice]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()
    for local_index, (condition_index, pressure) in enumerate(
        zip(condition_indices, gas_pressures)
    ):
        axis = axes[local_index]
        y_true = observations[:, condition_index]
        y_pred = fusion_prediction[:, condition_index]
        limit = max(float(y_true.max()), float(y_pred.max()), 1.0e-3) * 1.03
        axis.scatter(y_true, y_pred, s=9, alpha=0.30, color=color)
        axis.plot([0, limit], [0, limit], "k--", linewidth=1.3)
        axis.set(
            title=f"{gas} Uptake at {pressure:g} bar",
            xlabel=f"Measured {gas} uptake (mmol g$^{{-1}}$)",
            ylabel=f"Predicted {gas} uptake (mmol g$^{{-1}}$)",
            xlim=(0, limit),
            ylim=(0, limit),
        )
        row = metrics_df[
            (metrics_df["tower"] == "fusion")
            & (metrics_df["gas"] == gas)
            & (metrics_df["scope"] == "condition")
            & np.isclose(metrics_df["pressure_bar"].astype(float), float(pressure))
        ].iloc[0]
        axis.text(
            0.03,
            0.96,
            f"R$^2$ = {row.r2:.4f}\n"
            f"RMSE = {row.rmse_mmol_g:.4f} mmol g$^{{-1}}$\n"
            f"MAE = {row.mae_mmol_g:.4f} mmol g$^{{-1}}$",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
        )
        axis.grid(alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle(
        f"Measured vs Predicted {gas} Uptake on the Independent Test Set",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"test_scatter_{gas.lower()}_all_pressures.png", dpi=220)
    plt.show()

    gas_observed = observations[:, condition_slice]
    gas_predicted = fusion_prediction[:, condition_slice]
    sample_order = np.argsort(gas_observed[:, 2])
    sample_positions = np.linspace(0, len(sample_order) - 1, 9, dtype=int)
    sample_indices = sample_order[sample_positions]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for axis, sample_index in zip(axes.ravel(), sample_indices):
        axis.plot(gas_pressures, gas_observed[sample_index], "o-", label="Measured", linewidth=2)
        axis.plot(gas_pressures, gas_predicted[sample_index], "s--", label="Predicted", linewidth=2)
        axis.set_xscale("log")
        axis.set(
            title=test_result["mof_ids"][sample_index],
            xlabel="Pressure (bar)",
            ylabel=f"{gas} uptake (mmol g$^{{-1}}$)",
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(f"Representative Measured and Predicted {gas} Isotherms", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"representative_{gas.lower()}_isotherms.png", dpi=220)
    plt.show()

condition_labels = [f"{item['gas']}\n{item['pressure_bar']:g} bar" for item in CONDITIONS]
x_positions = np.arange(len(CONDITIONS))
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for tower in ("fusion", "structure", "descriptor"):
    rows = metrics_df[(metrics_df["tower"] == tower) & (metrics_df["scope"] == "condition")]
    axes[0].plot(x_positions, rows["r2"], "o-", label=tower)
    axes[1].plot(x_positions, rows["mae_mmol_g"], "o-", label=tower)
axes[0].set(ylabel="R$^2$", title="Tower Ablation Across Gas-Pressure Conditions")
axes[1].set(ylabel="MAE (mmol g$^{-1}$)", xlabel="Gas and pressure")
axes[1].set_xticks(x_positions, condition_labels)
for axis in axes:
    axis.axvline(4.5, color="0.55", linestyle=":")
    axis.grid(alpha=0.25)
    axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "tower_ablation_metrics.png", dpi=220)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fusion_recovery = recovery_df[recovery_df["tower"] == "fusion"]
for axis, gas in zip(axes, ("CO2", "CH4")):
    gas_rows = fusion_recovery[fusion_recovery["gas"] == gas]
    for fraction in (0.01, 0.05, 0.10):
        rows = gas_rows[gas_rows["true_top_fraction"] == fraction]
        axis.plot(rows["pressure_bar"], rows["recall_at_k"], "o-", label=f"Top {fraction:.0%}")
    axis.set_xscale("log")
    axis.set(title=f"{gas} High-Uptake Candidate Recovery", xlabel="Pressure (bar)")
    axis.grid(alpha=0.25)
    axis.legend()
axes[0].set_ylabel("Recall@K")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "high_uptake_candidate_recovery.png", dpi=220)
plt.show()

# %% [markdown]
# ## 11. Rotation-invariance stress test

# %%
def random_rotation_matrix(device):
    matrix = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    q = q * signs
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


rotation_dataset = EnrichedH5Dataset(test_indices[:100])
rotation_differences = []
model.eval()
for sample_index in tqdm(range(len(rotation_dataset)), desc="Rotation test"):
    original = collate_mofs([rotation_dataset[sample_index]])
    rotated = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in original.items()
    }
    rotation = random_rotation_matrix(torch.device("cpu"))
    rotated["edge_vec"] = rotated["edge_vec"] @ rotation.T
    original = move_batch_to_device(original)
    rotated = move_batch_to_device(rotated)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE):
        original_prediction = model(original)["fusion"]
        rotated_prediction = model(rotated)["fusion"]
    rotation_differences.append(
        float(
            (original_prediction.float() - rotated_prediction.float())
            .abs()
            .max()
            .cpu()
        )
    )

rotation_df = pd.DataFrame(
    {"maximum_absolute_difference_mmol_g": rotation_differences}
)
rotation_df.to_csv(OUTPUT_DIR / "rotation_invariance_test.csv", index=False)
print(rotation_df.describe())

fig, axis = plt.subplots(figsize=(8, 5))
axis.hist(rotation_differences, bins=20, color="#4C78A8", edgecolor="black")
axis.set(
    title="Rotation-Invariance Stress Test",
    xlabel="Maximum absolute prediction difference (mmol g$^{-1}$)",
    ylabel="Number of MOFs",
)
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "rotation_invariance_test.png", dpi=220)
plt.show()

evaluation_summary = {
    "best_epoch": int(best_checkpoint["epoch"]),
    "best_validation_mean_normalized_rmse": float(
        best_checkpoint["best_val_score"]
    ),
    "trainable_parameters": int(trainable_parameters),
    "test_rows": int(len(test_indices)),
    "target_units": "mmol g^-1",
    "fusion_condition_metrics": metrics_df[
        (metrics_df["tower"] == "fusion")
        & (metrics_df["scope"] == "condition")
    ].to_dict(orient="records"),
    "rotation_maximum_absolute_difference_mmol_g": float(
        rotation_df["maximum_absolute_difference_mmol_g"].max()
    ),
}
(OUTPUT_DIR / "evaluation_summary.json").write_text(
    json.dumps(evaluation_summary, indent=2, allow_nan=True), encoding="utf-8"
)
print("All checkpoints, tables, and figures were saved to:", OUTPUT_DIR)

# %% [markdown]
# ## 12. Optional runtime disconnect (run only after checking saved outputs)

# %%
DISCONNECT_RUNTIME = False

if DISCONNECT_RUNTIME:
    from google.colab import runtime

    runtime.unassign()
