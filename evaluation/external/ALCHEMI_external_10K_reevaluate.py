#!/usr/bin/env python3
"""Colab GPU ALCHEMI preprocessing, frozen-model inference, and external 10K evaluation.

Scientific protocol:
1. Read only the input CSV and CIF ZIP.
2. Rebuild periodic graphs with NVIDIA ALCHEMI Toolkit-Ops on CUDA.
3. Apply only the preprocessing objects frozen on the original 50K training set.
4. Save and SHA-256 seal all 10,000 predictions.
5. Only then read the private labels and calculate metrics.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import ase
import joblib
import numpy as np
import pandas as pd
import pymatgen
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp
from ase.data import atomic_numbers, chemical_symbols
from ase.io import read as ase_read
from nvalchemiops.torch.neighbors import neighbor_list
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

import nvalchemiops


warnings.filterwarnings("ignore")


@dataclass
class Config:
    seed: int = 42
    expected_rows: int = 10_000
    cutoff_angstrom: float = 6.0
    max_neighbors: int = 32
    max_atoms_per_alchemi_batch: int = 12_000
    max_structures_per_alchemi_batch: int = 128
    parse_workers: int = 6
    parse_chunk_size: int = 512
    max_nodes_per_model_batch: int = 4_200
    max_edges_per_model_batch: int = 130_000
    max_graphs_per_model_batch: int = 64
    flush_every: int = 250
    bootstrap_replicates: int = 1_000
    bond_score_temperature: float = 0.08
    nonmetal_nonmetal_ratio_limit: float = 1.25
    metal_nonmetal_ratio_limit: float = 1.40
    metal_metal_ratio_limit: float = 1.25


CFG = Config()
DEVICE = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("Colab 런타임을 GPU로 변경한 뒤 다시 실행하세요.")

torch.set_float32_matmul_precision("high")
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)

BASE_DIR = Path(os.environ.get("MOF_BASE_DIR", "/content/drive/MyDrive/mf_hid/4th"))
EVAL_DIR = BASE_DIR / "재평가"
INPUT_CSV = EVAL_DIR / "MOF_external_10K_inputs.csv"
CIF_ZIP = EVAL_DIR / "MOF_external_10K_CIF.zip"
LABEL_CSV = EVAL_DIR / "MOF_external_10K_labels_private.csv"
SOURCE_MANIFEST = EVAL_DIR / "MOF_external_10K_manifest.json"

WORK_DIR = Path("/content/alchemi_external_10k")
CIF_DIR = WORK_DIR / "cifs"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CIF_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_CSV = EVAL_DIR / "external_10K_alchemi_progress.csv"
PREDICTIONS_CSV = EVAL_DIR / "external_10K_alchemi_predictions.csv"
DESCRIPTORS_CSV = EVAL_DIR / "external_10K_alchemi_descriptors.csv"
INFERENCE_MANIFEST = EVAL_DIR / "external_10K_alchemi_inference_manifest.json"
EVALUATION_ROWS_CSV = EVAL_DIR / "external_10K_alchemi_evaluation_rows.csv"
PER_TARGET_METRICS_CSV = EVAL_DIR / "external_10K_alchemi_per_target_metrics.csv"
METRICS_JSON = EVAL_DIR / "external_10K_alchemi_metrics.json"

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
PREDICTION_COLUMNS = [f"pred_{name}" for name in TARGET_COLUMNS]
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_artifact(name: str) -> Path:
    candidates = sorted(BASE_DIR.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"기존 50K 학습 산출물을 찾을 수 없습니다: {name}")
    if len(candidates) > 1:
        exact_result = [path for path in candidates if path.parent.name == "결과"]
        if len(exact_result) == 1:
            return exact_result[0]
        raise RuntimeError(f"동일 이름 파일이 여러 개입니다: {name}: {candidates}")
    return candidates[0]


CHECKPOINT = resolve_artifact("best_dual_gas_alchemi_50k.pt")
DESCRIPTOR_PREPROCESSOR = resolve_artifact("descriptor_preprocessing.joblib")
NODE_PREPROCESSOR = resolve_artifact("node_feature_preprocessing.npz")
ELEMENT_TABLE = resolve_artifact("element_property_table_50k.json")
METAL_REFERENCE = resolve_artifact("metal_soft_cn_reference.json")
ORIGINAL_50K_CSV = resolve_artifact("MOF_CO2_CH4_50K.csv")

for required in (INPUT_CSV, CIF_ZIP, CHECKPOINT, DESCRIPTOR_PREPROCESSOR,
                 NODE_PREPROCESSOR, ELEMENT_TABLE, METAL_REFERENCE,
                 ORIGINAL_50K_CSV):
    if not required.is_file():
        raise FileNotFoundError(required)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("ALCHEMI Toolkit-Ops:", nvalchemiops.__version__)
print("Warp:", wp.__version__)
print("Evaluation folder:", EVAL_DIR)
print("Checkpoint:", CHECKPOINT)


# ---- Frozen model architecture copied verbatim from the 50K training code ----

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


# ---- NVIDIA ALCHEMI graph construction copied from the 50K feature code ----

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


# ---- Descriptor construction and tensor collation ----

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


class FrozenAssets:
    def __init__(self):
        self.checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=False
        )
        config = SimpleNamespace(**self.checkpoint["config"])
        self.model = DualGasAlchemiModel(
            int(self.checkpoint["descriptor_dim"]), config
        )
        result = self.model.load_state_dict(
            self.checkpoint["model_state_dict"], strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(str(result))
        self.model.to(DEVICE).eval()

        descriptor_payload = joblib.load(DESCRIPTOR_PREPROCESSOR)
        self.descriptor_pipeline = descriptor_payload["pipeline"]
        self.descriptor_columns = list(descriptor_payload["feature_columns"])
        if self.descriptor_columns != list(self.checkpoint["descriptor_columns"]):
            raise RuntimeError("Checkpoint와 descriptor 전처리기의 열 순서가 다릅니다.")

        node_payload = np.load(NODE_PREPROCESSOR, allow_pickle=False)
        self.node_mean = node_payload["mean"].astype(np.float32)
        self.node_std = node_payload["std"].astype(np.float32)
        self.node_names = node_payload["names"].astype(str).tolist()
        if self.node_names != list(self.checkpoint["node_feature_names"]):
            raise RuntimeError("Checkpoint와 node 전처리기의 특성 순서가 다릅니다.")

        self.elements = json.loads(ELEMENT_TABLE.read_text(encoding="utf-8"))
        metal_payload = json.loads(METAL_REFERENCE.read_text(encoding="utf-8"))
        self.metal_reference = {
            int(key): value
            for key, value in metal_payload["by_atomic_number"].items()
        }
        self.metal_fallback = metal_payload["fallback_all_training_metals"]
        self.metal_z_values = sorted(
            int(record["atomic_number"])
            for record in self.elements.values()
            if record["metal_flag"] > 0.5
        )


def move_batch_to_device(batch: dict) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(DEVICE, non_blocking=True) if torch.is_tensor(value) else value
    return result


def predict_samples_gpu(samples: list[dict], assets: FrozenAssets) -> np.ndarray:
    batch = move_batch_to_device(collate(samples))
    with torch.inference_mode():
        predictions = assets.model(batch)["fusion"].float().cpu().numpy()
    del batch
    return predictions


def validate_and_extract_inputs() -> pd.DataFrame:
    inputs = pd.read_csv(INPUT_CSV)
    required_columns = {
        "selection_rank", "mof_id", "surface_area_m2g",
        "void_fraction", "lcd", "pld", "elements",
    }
    missing = sorted(required_columns - set(inputs.columns))
    if missing:
        raise RuntimeError(f"입력 CSV 필수 열 누락: {missing}")
    if len(inputs) != CFG.expected_rows:
        raise RuntimeError(f"입력 행 수 {len(inputs):,}; 예상 {CFG.expected_rows:,}")
    if inputs["mof_id"].duplicated().any():
        raise RuntimeError("입력 CSV에 중복 mof_id가 있습니다.")
    if set(TARGET_COLUMNS) & set(inputs.columns):
        raise RuntimeError("입력 CSV에 정답 열이 들어 있습니다.")
    inputs = inputs.sort_values("selection_rank").reset_index(drop=True)
    if not np.array_equal(
        inputs["selection_rank"].to_numpy(dtype=int),
        np.arange(CFG.expected_rows),
    ):
        raise RuntimeError("selection_rank가 0..9999 순서가 아닙니다.")
    if not np.isfinite(
        inputs[["surface_area_m2g", "void_fraction", "lcd", "pld"]]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError("입력 descriptor에 결측값 또는 무한값이 있습니다.")

    original_50k_ids = set(
        pd.read_csv(ORIGINAL_50K_CSV, usecols=["mof_id"])["mof_id"].astype(str)
    )
    overlap = sorted(set(inputs["mof_id"].astype(str)) & original_50k_ids)
    if overlap:
        raise RuntimeError(
            f"외부 10K와 기존 학습 50K가 {len(overlap):,}개 겹칩니다: {overlap[:10]}"
        )

    if SOURCE_MANIFEST.is_file():
        source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        expected_hashes = source_manifest.get("files_sha256", {})
        for path in (INPUT_CSV, CIF_ZIP):
            expected = expected_hashes.get(path.name)
            if expected and sha256_file(path) != expected:
                raise RuntimeError(f"원본 manifest와 SHA-256이 다릅니다: {path.name}")

    with zipfile.ZipFile(CIF_ZIP) as archive:
        names = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".cif")
        ]
        zip_ids = [Path(name).stem for name in names]
        if len(zip_ids) != CFG.expected_rows or len(set(zip_ids)) != CFG.expected_rows:
            raise RuntimeError(
                f"CIF ZIP은 중복 없는 10,000개여야 합니다: {len(zip_ids):,}"
            )
        if set(zip_ids) != set(inputs["mof_id"]):
            raise RuntimeError("입력 CSV와 CIF ZIP의 mof_id 집합이 다릅니다.")
        marker = CIF_DIR / ".complete.json"
        marker_ok = False
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                marker_ok = (
                    payload.get("zip_sha256") == sha256_file(CIF_ZIP)
                    and payload.get("cif_count") == CFG.expected_rows
                )
            except Exception:
                marker_ok = False
        if not marker_ok:
            if CIF_DIR.resolve() != (WORK_DIR / "cifs").resolve():
                raise RuntimeError("안전하지 않은 CIF 임시 경로입니다.")
            shutil.rmtree(CIF_DIR)
            CIF_DIR.mkdir(parents=True)
            archive.extractall(CIF_DIR)
            marker.write_text(
                json.dumps(
                    {
                        "zip_sha256": sha256_file(CIF_ZIP),
                        "cif_count": CFG.expected_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    cif_by_id = {}
    for path in CIF_DIR.rglob("*.cif"):
        mof_id = Path(path.name).stem
        if mof_id in cif_by_id:
            raise RuntimeError(f"중복 CIF: {mof_id}")
        cif_by_id[mof_id] = path
    if set(cif_by_id) != set(inputs["mof_id"]):
        raise RuntimeError("압축 해제 후 CSV/CIF mof_id가 일치하지 않습니다.")
    return inputs


def alchemi_batches(inputs: pd.DataFrame, completed_ids: set[str]):
    jobs = [
        (int(row.selection_rank), str(row.mof_id), CIF_DIR / f"{row.mof_id}.cif")
        for row in inputs.itertuples(index=False)
        if str(row.mof_id) not in completed_ids
    ]
    pending = []
    pending_atoms = 0
    with ThreadPoolExecutor(max_workers=CFG.parse_workers) as executor:
        for start in range(0, len(jobs), CFG.parse_chunk_size):
            chunk = jobs[start : start + CFG.parse_chunk_size]
            parsed = list(executor.map(parse_cif, chunk))
            for item in parsed:
                n_atoms = len(item.z)
                should_flush = pending and (
                    pending_atoms + n_atoms > CFG.max_atoms_per_alchemi_batch
                    or len(pending) >= CFG.max_structures_per_alchemi_batch
                )
                if should_flush:
                    yield pending
                    pending = []
                    pending_atoms = 0
                pending.append(item)
                pending_atoms += n_atoms
    if pending:
        yield pending


def predict_graph_group(
    graphs: list[dict],
    inputs_by_id: pd.DataFrame,
    assets: FrozenAssets,
) -> list[dict]:
    records = []
    pending_samples = []
    pending_meta = []
    total_nodes = 0
    total_edges = 0

    def flush_model_batch():
        nonlocal pending_samples, pending_meta, total_nodes, total_edges
        if not pending_samples:
            return
        predictions = predict_samples_gpu(pending_samples, assets)
        for meta, prediction in zip(pending_meta, predictions):
            records.append(
                {
                    **meta,
                    **{
                        name: float(value)
                        for name, value in zip(PREDICTION_COLUMNS, prediction)
                    },
                }
            )
        pending_samples = []
        pending_meta = []
        total_nodes = 0
        total_edges = 0

    for graph in sorted(graphs, key=lambda item: item["row_index"]):
        mof_id = str(graph["mof_id"])
        base_row = inputs_by_id.loc[mof_id]
        descriptor = graph_descriptors(mof_id, graph, base_row, assets)
        sample = prepare_sample(
            mof_id, graph, descriptor, assets, int(graph["row_index"])
        )
        n_nodes = len(graph["z"])
        n_edges = len(graph["src"])
        exceeds = pending_samples and (
            total_nodes + n_nodes > CFG.max_nodes_per_model_batch
            or total_edges + n_edges > CFG.max_edges_per_model_batch
            or len(pending_samples) >= CFG.max_graphs_per_model_batch
        )
        if exceeds:
            flush_model_batch()
        pending_samples.append(sample)
        pending_meta.append(
            {
                "selection_rank": int(graph["row_index"]),
                "mof_id": mof_id,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                **{name: descriptor[name] for name in assets.descriptor_columns},
            }
        )
        total_nodes += n_nodes
        total_edges += n_edges
    flush_model_batch()
    return records


def run_alchemi_and_inference(inputs: pd.DataFrame, assets: FrozenAssets) -> str:
    global ELEMENT_PROPERTIES
    ELEMENT_PROPERTIES = assets.elements
    inputs_by_id = inputs.set_index("mof_id", drop=False)

    if PROGRESS_CSV.is_file():
        progress = pd.read_csv(PROGRESS_CSV)
        required = {"selection_rank", "mof_id", *assets.descriptor_columns, *PREDICTION_COLUMNS}
        if not required.issubset(progress.columns):
            raise RuntimeError("기존 progress CSV의 스키마가 현재 모델과 다릅니다.")
        if progress["mof_id"].duplicated().any():
            raise RuntimeError("기존 progress CSV에 중복 mof_id가 있습니다.")
        if not set(progress["mof_id"]).issubset(set(inputs["mof_id"])):
            raise RuntimeError("기존 progress CSV에 이번 10K 이외의 ID가 있습니다.")
        records = progress.to_dict("records")
    else:
        records = []

    completed_ids = {str(record["mof_id"]) for record in records}
    print(f"재개 지점: {len(completed_ids):,}/{CFG.expected_rows:,}")
    started = time.perf_counter()
    last_flush = len(records)
    progress_bar = tqdm(total=CFG.expected_rows, initial=len(records), desc="ALCHEMI + inference")

    for parsed_batch in alchemi_batches(inputs, completed_ids):
        graphs = process_gpu_batch_with_retry(parsed_batch)
        new_records = predict_graph_group(graphs, inputs_by_id, assets)
        records.extend(new_records)
        progress_bar.update(len(new_records))
        del graphs, parsed_batch, new_records
        gc.collect()
        torch.cuda.empty_cache()

        if len(records) - last_flush >= CFG.flush_every:
            frame = pd.DataFrame(records).sort_values("selection_rank")
            atomic_write_csv(frame, PROGRESS_CSV)
            last_flush = len(records)

    progress_bar.close()
    frame = pd.DataFrame(records).sort_values("selection_rank").reset_index(drop=True)
    if len(frame) != CFG.expected_rows:
        raise RuntimeError(f"예측 수가 10,000개가 아닙니다: {len(frame):,}")
    if frame["mof_id"].duplicated().any():
        raise RuntimeError("최종 예측에 중복 mof_id가 있습니다.")
    if list(frame["mof_id"]) != list(inputs["mof_id"]):
        raise RuntimeError("최종 예측 순서가 입력 순서와 다릅니다.")
    if not np.isfinite(frame[PREDICTION_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("예측값에 결측값 또는 무한값이 있습니다.")

    atomic_write_csv(frame, PROGRESS_CSV)
    predictions = frame[
        ["selection_rank", "mof_id", "n_nodes", "n_edges", *PREDICTION_COLUMNS]
    ].copy()
    descriptors = frame[
        ["selection_rank", "mof_id", "n_nodes", "n_edges", *assets.descriptor_columns]
    ].copy()
    atomic_write_csv(predictions, PREDICTIONS_CSV)
    atomic_write_csv(descriptors, DESCRIPTORS_CSV)
    prediction_hash = sha256_file(PREDICTIONS_CSV)

    inference_payload = {
        "protocol": (
            "Private labels were not read. Predictions were completed and SHA-256 "
            "sealed before the evaluation phase."
        ),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0),
        "neighbor_backend": "nvalchemiops.torch.neighbors.neighbor_list",
        "alchemi_method": "cell_list/batch_cell_list",
        "config": asdict(CFG),
        "n_predictions": len(predictions),
        "unique_mof_ids": int(predictions["mof_id"].nunique()),
        "model_parameters": int(sum(p.numel() for p in assets.model.parameters())),
        "total_nodes": int(predictions["n_nodes"].sum()),
        "total_edges": int(predictions["n_edges"].sum()),
        "run_seconds_this_session": float(time.perf_counter() - started),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "nvalchemiops": str(nvalchemiops.__version__),
            "warp": str(wp.__version__),
            "ase": ase.__version__,
            "pymatgen": str(getattr(pymatgen, "__version__", "unknown")),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "files_sha256": {
            "input_csv": sha256_file(INPUT_CSV),
            "cif_zip": sha256_file(CIF_ZIP),
            "checkpoint": sha256_file(CHECKPOINT),
            "descriptor_preprocessor": sha256_file(DESCRIPTOR_PREPROCESSOR),
            "node_preprocessor": sha256_file(NODE_PREPROCESSOR),
            "element_table": sha256_file(ELEMENT_TABLE),
            "metal_reference": sha256_file(METAL_REFERENCE),
            "original_50k_csv": sha256_file(ORIGINAL_50K_CSV),
            "predictions": prediction_hash,
            "descriptors": sha256_file(DESCRIPTORS_CSV),
        },
    }
    atomic_write_json(inference_payload, INFERENCE_MANIFEST)
    print("예측 봉인 완료:", PREDICTIONS_CSV)
    print("prediction SHA-256:", prediction_hash)
    return prediction_hash


def bootstrap_intervals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rng = np.random.default_rng(CFG.seed)
    n = len(y_true)
    values = {"r2": [], "rmse": [], "mae": []}
    for _ in range(CFG.bootstrap_replicates):
        index = rng.integers(0, n, size=n)
        true_sample = y_true[index]
        pred_sample = y_pred[index]
        values["r2"].append(r2_score(true_sample, pred_sample))
        values["rmse"].append(
            mean_squared_error(true_sample, pred_sample) ** 0.5
        )
        values["mae"].append(mean_absolute_error(true_sample, pred_sample))
    return {
        f"{metric}_{bound}": float(np.percentile(samples, percentile))
        for metric, samples in values.items()
        for bound, percentile in (("ci95_low", 2.5), ("ci95_high", 97.5))
    }


def evaluate_after_prediction_seal(prediction_hash: str) -> None:
    # 중요: private label 파일은 이 함수에 들어온 뒤 처음 읽는다.
    if sha256_file(PREDICTIONS_CSV) != prediction_hash:
        raise RuntimeError("라벨 개봉 전후 prediction 파일 해시가 달라졌습니다.")
    if not LABEL_CSV.is_file():
        raise FileNotFoundError(LABEL_CSV)

    predictions = pd.read_csv(PREDICTIONS_CSV)
    labels = pd.read_csv(LABEL_CSV)
    if len(labels) != CFG.expected_rows or labels["mof_id"].duplicated().any():
        raise RuntimeError("비공개 라벨 CSV는 중복 없는 10,000행이어야 합니다.")
    merged = predictions.merge(
        labels[["selection_rank", "mof_id", *TARGET_COLUMNS]],
        on=["selection_rank", "mof_id"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != CFG.expected_rows or merged[TARGET_COLUMNS].isna().any().any():
        raise RuntimeError("예측과 라벨의 one-to-one 병합에 실패했습니다.")

    per_target = []
    for target, prediction in zip(TARGET_COLUMNS, PREDICTION_COLUMNS):
        y_true = merged[target].to_numpy(dtype=float)
        y_pred = merged[prediction].to_numpy(dtype=float)
        row = {
            "gas": "CO2" if target.startswith("co2") else "CH4",
            "target": target,
            "prediction_column": prediction,
            "n": len(y_true),
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mean_true": float(y_true.mean()),
            "std_true": float(y_true.std(ddof=0)),
            **bootstrap_intervals(y_true, y_pred),
        }
        per_target.append(row)

    metrics = pd.DataFrame(per_target)
    atomic_write_csv(metrics, PER_TARGET_METRICS_CSV)
    atomic_write_csv(merged, EVALUATION_ROWS_CSV)
    summary = {
        "protocol": (
            "Predictions were SHA-256 sealed before the private labels were read."
        ),
        "n": CFG.expected_rows,
        "prediction_sha256_before_label_access": prediction_hash,
        "prediction_sha256_after_evaluation": sha256_file(PREDICTIONS_CSV),
        "private_labels_sha256": sha256_file(LABEL_CSV),
        "bootstrap_replicates": CFG.bootstrap_replicates,
        "macro_all_targets": {
            "r2": float(metrics["r2"].mean()),
            "rmse": float(metrics["rmse"].mean()),
            "mae": float(metrics["mae"].mean()),
        },
        "macro_by_gas": {
            gas: {
                "r2": float(group["r2"].mean()),
                "rmse": float(group["rmse"].mean()),
                "mae": float(group["mae"].mean()),
            }
            for gas, group in metrics.groupby("gas")
        },
        "per_target": per_target,
        "output_files": {
            "predictions": PREDICTIONS_CSV.name,
            "descriptors": DESCRIPTORS_CSV.name,
            "inference_manifest": INFERENCE_MANIFEST.name,
            "evaluation_rows": EVALUATION_ROWS_CSV.name,
            "per_target_metrics": PER_TARGET_METRICS_CSV.name,
        },
    }
    atomic_write_json(summary, METRICS_JSON)
    print("\n외부 10K ALCHEMI 재평가 완료")
    print(metrics[["gas", "target", "r2", "rmse", "mae"]].to_string(index=False))
    print("\nMacro:", json.dumps(summary["macro_all_targets"], indent=2))
    print("Saved:", METRICS_JSON)


def main() -> None:
    inputs = validate_and_extract_inputs()
    assets = FrozenAssets()
    print("Frozen model parameters:", f"{sum(p.numel() for p in assets.model.parameters()):,}")
    prediction_hash = run_alchemi_and_inference(inputs, assets)
    evaluate_after_prediction_seal(prediction_hash)


if __name__ == "__main__":
    main()
