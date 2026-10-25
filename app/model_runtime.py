from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from pathlib import Path

import pandas as pd
import torch

import core_inference as core


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"

# Rebind the validated inference module to the portable app assets.
core.RESULTS = ASSET_DIR
core.CHECKPOINT = Path(os.environ.get('MOF_CHECKPOINT_PATH',str(ASSET_DIR / "best_dual_gas_alchemi_50k.pt")))
core.DESCRIPTOR_PREPROCESSOR = ASSET_DIR / "descriptor_preprocessing.joblib"
core.NODE_PREPROCESSOR = ASSET_DIR / "node_feature_preprocessing.npz"
core.ELEMENT_TABLE = ASSET_DIR / "element_property_table_50k.json"
core.METAL_REFERENCE = ASSET_DIR / "metal_soft_cn_reference.json"
core.TRAINING_SOURCE = ASSET_DIR / "15.DualGas_ALCHEMI_50K_L4_Training.py"


PRESSURES = {
    "CO2": [0.01, 0.05, 0.10, 0.50, 2.50],
    "CH4": [0.05, 0.50, 0.90, 2.50, 4.50],
}


class Predictor:
    def __init__(self) -> None:
        self._assets = None
        self._load_error: str | None = None
        self._loading = False
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loaded_at: float | None = None

    @property
    def ready(self) -> bool:
        return self._assets is not None

    def status(self) -> dict:
        files = {
            "checkpoint": core.CHECKPOINT,
            "descriptor_preprocessor": core.DESCRIPTOR_PREPROCESSOR,
            "node_preprocessor": core.NODE_PREPROCESSOR,
            "element_table": core.ELEMENT_TABLE,
            "metal_reference": core.METAL_REFERENCE,
            "training_source": core.TRAINING_SOURCE,
        }
        return {
            "ready": self.ready,
            "loading": self._loading,
            "error": self._load_error,
            "device": "CPU",
            "torch": torch.__version__,
            "files": {name: path.is_file() for name, path in files.items()},
            "loaded_at": self._loaded_at,
        }

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            self._loading = True
            self._load_error = None
            try:
                torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
                self._assets = core.FrozenAssets()
                self._loaded_at = time.time()
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._loading = False

    def predict(self, cif_path: Path, descriptors: dict[str, float]) -> dict:
        self.load()
        for key in ("surface_area_m2g", "void_fraction", "lcd", "pld"):
            value = descriptors.get(key)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{key} 값이 유효한 숫자가 아닙니다.")
        if not 0 <= float(descriptors["void_fraction"]) <= 1:
            raise ValueError("공극률은 0과 1 사이여야 합니다.")
        if min(float(descriptors["surface_area_m2g"]), float(descriptors["lcd"]), float(descriptors["pld"])) < 0:
            raise ValueError("표면적, LCD, PLD는 음수가 될 수 없습니다.")

        started = time.perf_counter()
        graph_started = time.perf_counter()
        graph = core.build_cpu_graph(cif_path, self._assets)
        graph_seconds = time.perf_counter() - graph_started

        mof_key = f"upload-{hashlib.sha256(cif_path.read_bytes()).hexdigest()[:12]}"
        descriptor = core.graph_descriptors(
            mof_key,
            graph,
            pd.Series(descriptors, dtype=float),
            self._assets,
        )
        sample = core.prepare_sample(mof_key, graph, descriptor, self._assets, 0)
        inference_started = time.perf_counter()
        with self._lock:
            values = core.predict_samples([sample], self._assets)[0]
        inference_seconds = time.perf_counter() - inference_started

        co2 = [float(value) for value in values[:5]]
        ch4 = [float(value) for value in values[5:]]
        return {
            "mof_key": mof_key,
            "predictions": {
                "CO2": [
                    {"pressure_bar": pressure, "uptake_mmol_g": value}
                    for pressure, value in zip(PRESSURES["CO2"], co2)
                ],
                "CH4": [
                    {"pressure_bar": pressure, "uptake_mmol_g": value}
                    for pressure, value in zip(PRESSURES["CH4"], ch4)
                ],
            },
            "structure": {
                "atoms": int(len(graph["z"])),
                "periodic_edges": int(len(graph["src"])),
                "elements": sorted({core.chemical_symbols[int(z)] for z in graph["z"]}),
            },
            "timing": {
                "graph_seconds": graph_seconds,
                "inference_seconds": inference_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        }


predictor = Predictor()
