# %% [markdown]
# # Descriptor baselines for the fixed MOF 50K split
#
# This script trains two independent descriptor-only baselines on the exact
# 35K/5K/10K split used by the final model:
# - CatBoost with MultiRMSE on normalized log1p targets
# - A compact MLP with the same normalized log1p targets
#
# Existing structure-head, descriptor-head, and fusion predictions are loaded
# from the saved final-model test_predictions.csv and re-evaluated on the same
# test MOFs. The two tower-head outputs are auxiliary heads from joint training,
# not independently retrained single-tower models.

# %%
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

if importlib.util.find_spec("catboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])

import catboost

import joblib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from catboost import CatBoostError, CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from google.colab import drive

drive.mount("/content/drive")


SEED = 42
BASE_DIR = Path("/content/drive/MyDrive/mf_hid/4th")
RESULT_DIR = BASE_DIR / "결과"
OUTPUT_DIR = RESULT_DIR / "descriptor_baseline_comparison"
LOCAL_DIR = Path("/content/descriptor_baseline_comparison")

SOURCE_CSV = BASE_DIR / "MOF_CO2_CH4_50K_alchemi_enriched.csv"
SOURCE_SPLIT = BASE_DIR / "MOF_CO2_CH4_50K_split_70_10_20.json"
LOCAL_CSV = LOCAL_DIR / SOURCE_CSV.name

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "co2_q_0p01bar",
    "co2_q_0p05bar",
    "co2_adsorption_0.1bar",
    "co2_adsorption_0.5bar",
    "co2_adsorption_2.5bar",
    "ch4_adsorption_0.05bar",
    "ch4_adsorption_0.5bar",
    "ch4_adsorption_0.9bar",
    "ch4_adsorption_2.5bar",
    "ch4_adsorption_4.5bar",
]
CONDITIONS = [
    ("CO2", 0.01),
    ("CO2", 0.05),
    ("CO2", 0.10),
    ("CO2", 0.50),
    ("CO2", 2.50),
    ("CH4", 0.05),
    ("CH4", 0.50),
    ("CH4", 0.90),
    ("CH4", 2.50),
    ("CH4", 4.50),
]
FALLBACK_DESCRIPTOR_COLUMNS = [
    "surface_area_m2g",
    "void_fraction",
    "lcd",
    "pld",
    "n_atoms_cif",
    "metal_atom_fraction",
    "metal_species_count",
    "open_metal_site_proxy_count",
    "open_metal_site_proxy_fraction",
    "metal_coordination_mean",
    "metal_coordination_deficit_mean",
    "ligand_polar_atom_fraction",
    "ligand_bond_en_difference_mean",
    "pore_proxy_polar_atom_fraction",
    "pore_proxy_heteroatom_fraction",
    "pore_proxy_metal_fraction",
    "pore_proxy_electronegativity_mean",
    "local_density_r3_mean",
    "local_density_r3_std",
    "local_density_r6_mean",
    "local_density_r6_std",
    "spatial_only_edge_fraction",
    "nonmetal_nonmetal_bond_fraction",
    "metal_nonmetal_bond_fraction",
    "metal_metal_bond_fraction",
    "mean_edge_bond_score",
    "metal_type_V",
    "metal_type_Cu",
    "metal_type_Zn",
    "metal_type_Zr",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def find_result_file(name):
    direct = RESULT_DIR / name
    if direct.exists():
        return direct
    matches = sorted(RESULT_DIR.rglob(name), key=lambda path: (len(path.parts), str(path)))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} below {RESULT_DIR}")
    print(f"Using {name}: {matches[0]}")
    return matches[0]


seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

if not LOCAL_CSV.exists() or LOCAL_CSV.stat().st_size != SOURCE_CSV.stat().st_size:
    print("Copying enriched CSV to the Colab runtime disk...")
    shutil.copy2(SOURCE_CSV, LOCAL_CSV)

df = pd.read_csv(LOCAL_CSV)
split_payload = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
train_indices = np.sort(np.asarray(split_payload["train_indices"], dtype=np.int64))
val_indices = np.sort(np.asarray(split_payload["val_indices"], dtype=np.int64))
test_indices = np.sort(np.asarray(split_payload["test_indices"], dtype=np.int64))

if (len(train_indices), len(val_indices), len(test_indices)) != (35_000, 5_000, 10_000):
    raise RuntimeError("The fixed split is not 35K/5K/10K.")
if not np.array_equal(
    np.sort(np.concatenate([train_indices, val_indices, test_indices])),
    np.arange(len(df)),
):
    raise RuntimeError("Split indices do not cover the 50K CSV exactly once.")

configuration_matches = sorted(
    RESULT_DIR.rglob("training_configuration.json"),
    key=lambda path: (len(path.parts), str(path)),
)
if configuration_matches:
    training_configuration = json.loads(
        configuration_matches[0].read_text(encoding="utf-8")
    )
    descriptor_columns = training_configuration.get(
        "descriptor_columns", FALLBACK_DESCRIPTOR_COLUMNS
    )
else:
    descriptor_columns = FALLBACK_DESCRIPTOR_COLUMNS

if descriptor_columns != FALLBACK_DESCRIPTOR_COLUMNS:
    print("Descriptor list differs from the expected fallback; using the saved training list.")
if len(descriptor_columns) != 30:
    raise RuntimeError(f"Expected 30 descriptor columns, found {len(descriptor_columns)}")
missing_columns = [column for column in descriptor_columns if column not in df.columns]
if missing_columns:
    raise KeyError(f"Missing descriptor columns: {missing_columns}")

raw_features = df[descriptor_columns].replace([np.inf, -np.inf], np.nan)
targets = df[TARGET_COLUMNS].to_numpy(np.float32)
if not np.isfinite(targets).all() or (targets < 0).any():
    raise RuntimeError("Target table contains invalid values.")

# Fit every preprocessing statistic on the training split only.
feature_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
    ]
)
feature_pipeline.fit(raw_features.iloc[train_indices])
features_scaled = feature_pipeline.transform(raw_features).astype(np.float32)
joblib.dump(feature_pipeline, OUTPUT_DIR / "baseline_feature_preprocessing.joblib")

target_log = np.log1p(targets)
target_log_mean = target_log[train_indices].mean(axis=0).astype(np.float32)
target_log_std = target_log[train_indices].std(axis=0).clip(1.0e-4).astype(np.float32)
targets_normalized = (target_log - target_log_mean) / target_log_std
target_raw_std = targets[train_indices].std(axis=0).clip(0.05)


def invert_target_transform(values):
    log_values = values * target_log_std + target_log_mean
    return np.clip(np.expm1(log_values), 0.0, None).astype(np.float32)


print("Rows:", len(df))
print("Descriptor columns:", len(descriptor_columns))
print("Split:", len(train_indices), len(val_indices), len(test_indices))

# %% [markdown]
# ## 1. CatBoost baseline

# %%
catboost_model_path = OUTPUT_DIR / "catboost_30_descriptor_multitarget.cbm"
catboost_metadata_path = OUTPUT_DIR / "catboost_baseline_metadata.json"

catboost_parameters = {
    "iterations": 3_000,
    "depth": 8,
    "learning_rate": 0.03,
    "loss_function": "MultiRMSE",
    "eval_metric": "MultiRMSE",
    "l2_leaf_reg": 5.0,
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": 100,
    "task_type": "GPU" if torch.cuda.is_available() else "CPU",
    "devices": "0" if torch.cuda.is_available() else None,
}
catboost_parameters = {
    key: value for key, value in catboost_parameters.items() if value is not None
}

catboost_model = CatBoostRegressor(**catboost_parameters)
catboost_started = time.time()
try:
    catboost_model.fit(
        raw_features.iloc[train_indices],
        targets_normalized[train_indices],
        eval_set=(raw_features.iloc[val_indices], targets_normalized[val_indices]),
        use_best_model=True,
        early_stopping_rounds=200,
    )
except CatBoostError as error:
    if catboost_parameters["task_type"] != "GPU":
        raise
    print("GPU CatBoost failed; retrying on CPU:", error)
    catboost_parameters.pop("devices", None)
    catboost_parameters["task_type"] = "CPU"
    catboost_parameters["thread_count"] = max(1, os.cpu_count() or 1)
    catboost_model = CatBoostRegressor(**catboost_parameters)
    catboost_model.fit(
        raw_features.iloc[train_indices],
        targets_normalized[train_indices],
        eval_set=(raw_features.iloc[val_indices], targets_normalized[val_indices]),
        use_best_model=True,
        early_stopping_rounds=200,
    )

catboost_seconds = time.time() - catboost_started
catboost_model.save_model(catboost_model_path)
catboost_predictions = invert_target_transform(
    np.asarray(catboost_model.predict(raw_features.iloc[test_indices]), dtype=np.float32)
)
(catboost_metadata_path).write_text(
    json.dumps(
        {
            "parameters": catboost_parameters,
            "tree_count": int(catboost_model.tree_count_),
            "training_seconds": catboost_seconds,
            "catboost_version": catboost.__version__,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("CatBoost trees:", catboost_model.tree_count_)
print("CatBoost seconds:", round(catboost_seconds, 1))

# %% [markdown]
# ## 2. MLP baseline

# %%
class DescriptorMLP(nn.Module):
    def __init__(self, input_dim=30, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def make_loader(indices, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features_scaled[indices]),
        torch.from_numpy(targets_normalized[indices].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=DEVICE.type == "cuda",
        generator=generator if shuffle else None,
    )


train_loader = make_loader(train_indices, batch_size=512, shuffle=True)
val_loader = make_loader(val_indices, batch_size=1024, shuffle=False)
test_loader = make_loader(test_indices, batch_size=1024, shuffle=False)

mlp_model = DescriptorMLP().to(DEVICE)
mlp_parameter_count = sum(parameter.numel() for parameter in mlp_model.parameters())
optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=8, min_lr=1.0e-6
)
loss_function = nn.MSELoss()
amp_enabled = DEVICE.type == "cuda"
amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

best_val_loss = float("inf")
epochs_without_improvement = 0
patience = 30
maximum_epochs = 300
mlp_history = []
mlp_checkpoint_path = OUTPUT_DIR / "mlp_30_descriptor.pt"
mlp_started = time.time()

for epoch in range(1, maximum_epochs + 1):
    mlp_model.train()
    train_loss_sum = 0.0
    train_count = 0
    for features_batch, targets_batch in train_loader:
        features_batch = features_batch.to(DEVICE, non_blocking=True)
        targets_batch = targets_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = mlp_model(features_batch)
            loss = loss_function(output, targets_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), 5.0)
        optimizer.step()
        train_loss_sum += float(loss.detach()) * len(features_batch)
        train_count += len(features_batch)

    mlp_model.eval()
    val_loss_sum = 0.0
    val_count = 0
    with torch.no_grad():
        for features_batch, targets_batch in val_loader:
            features_batch = features_batch.to(DEVICE, non_blocking=True)
            targets_batch = targets_batch.to(DEVICE, non_blocking=True)
            output = mlp_model(features_batch)
            loss = loss_function(output, targets_batch)
            val_loss_sum += float(loss) * len(features_batch)
            val_count += len(features_batch)

    train_loss = train_loss_sum / train_count
    val_loss = val_loss_sum / val_count
    scheduler.step(val_loss)
    learning_rate = optimizer.param_groups[0]["lr"]
    mlp_history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
        }
    )

    if val_loss < best_val_loss - 1.0e-5:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(
            {
                "model_state_dict": mlp_model.state_dict(),
                "descriptor_columns": descriptor_columns,
                "parameter_count": mlp_parameter_count,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            },
            mlp_checkpoint_path,
        )
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"MLP epoch {epoch:03d} | train {train_loss:.5f} | "
            f"val {val_loss:.5f} | lr {learning_rate:.2e}"
        )
    if epochs_without_improvement >= patience:
        print("MLP early stopping at epoch", epoch)
        break

pd.DataFrame(mlp_history).to_csv(OUTPUT_DIR / "mlp_training_history.csv", index=False)
mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location=DEVICE, weights_only=False)
mlp_model.load_state_dict(mlp_checkpoint["model_state_dict"])
mlp_model.eval()

mlp_normalized_parts = []
with torch.no_grad():
    for features_batch, _ in test_loader:
        output = mlp_model(features_batch.to(DEVICE, non_blocking=True))
        mlp_normalized_parts.append(output.float().cpu().numpy())
mlp_predictions = invert_target_transform(np.concatenate(mlp_normalized_parts, axis=0))
mlp_seconds = time.time() - mlp_started
print("MLP parameters:", f"{mlp_parameter_count:,}")
print("MLP seconds:", round(mlp_seconds, 1))

# %% [markdown]
# ## 3. Load the jointly trained tower-head and fusion predictions

# %%
saved_predictions_path = find_result_file("test_predictions.csv")
saved_predictions = pd.read_csv(saved_predictions_path)


def find_saved_column(prefix, gas, target_pressure):
    candidates = []
    leading = f"{prefix}_{gas.lower()}_"
    trailing = "bar_mmol_g"
    for column in saved_predictions.columns:
        if column.startswith(leading) and column.endswith(trailing):
            pressure_text = column[len(leading) : -len(trailing)]
            try:
                pressure = float(pressure_text.replace("p", "."))
            except ValueError:
                continue
            candidates.append((abs(pressure - target_pressure), column, pressure))
    if not candidates:
        raise KeyError(f"No saved {prefix} column found for {gas}.")
    distance, column, pressure = min(candidates)
    if distance > 1.0e-4:
        raise KeyError(f"No {prefix} column matches {gas} {target_pressure:g} bar.")
    return column


saved_ids = saved_predictions["mof_id"].astype(str).to_numpy()
expected_test_ids = df.iloc[test_indices]["mof_id"].astype(str).to_numpy()
saved_order = {mof_id: index for index, mof_id in enumerate(saved_ids)}
if set(saved_ids) != set(expected_test_ids):
    raise RuntimeError("Saved final-model test IDs do not match the fixed test split.")
reorder = np.asarray([saved_order[mof_id] for mof_id in expected_test_ids])

existing_predictions = {}
for output_name, saved_prefix in (
    ("Structure auxiliary head", "structure"),
    ("Descriptor Transformer auxiliary head", "descriptor"),
    ("Final fusion model", "fusion"),
):
    matrix = []
    for gas, pressure in CONDITIONS:
        column = find_saved_column(saved_prefix, gas, pressure)
        matrix.append(saved_predictions[column].to_numpy(np.float32)[reorder])
    existing_predictions[output_name] = np.stack(matrix, axis=1)

test_observations = targets[test_indices]
for condition_index, (gas, pressure) in enumerate(CONDITIONS):
    observed_column = find_saved_column("observed", gas, pressure)
    saved_observed = saved_predictions[observed_column].to_numpy(np.float32)[reorder]
    if not np.allclose(saved_observed, test_observations[:, condition_index], atol=1.0e-5):
        raise RuntimeError(f"Saved observations disagree with the 50K CSV for {gas} {pressure:g} bar")

all_predictions = {
    "CatBoost (30 descriptors)": catboost_predictions,
    "MLP (30 descriptors)": mlp_predictions,
    **existing_predictions,
}

# %% [markdown]
# ## 4. Common test-set metrics

# %%
metric_rows = []
summary_rows = []
for model_name, prediction in all_predictions.items():
    normalized_rmse_values = []
    for condition_index, (gas, pressure) in enumerate(CONDITIONS):
        observed = test_observations[:, condition_index]
        predicted = prediction[:, condition_index]
        rmse = math.sqrt(mean_squared_error(observed, predicted))
        mae = mean_absolute_error(observed, predicted)
        normalized_rmse = rmse / target_raw_std[condition_index]
        normalized_rmse_values.append(normalized_rmse)
        metric_rows.append(
            {
                "model": model_name,
                "gas": gas,
                "pressure_bar": pressure,
                "r2": r2_score(observed, predicted),
                "rmse_mmol_g": rmse,
                "mae_mmol_g": mae,
                "wape_pct": 100.0 * np.abs(observed - predicted).sum() / max(observed.sum(), 1.0e-12),
                "normalized_rmse": normalized_rmse,
                "n_test": len(observed),
            }
        )

    summary_rows.append(
        {
            "model": model_name,
            "overall_flattened_r2": r2_score(test_observations.ravel(), prediction.ravel()),
            "overall_flattened_rmse_mmol_g": math.sqrt(
                mean_squared_error(test_observations.ravel(), prediction.ravel())
            ),
            "overall_flattened_mae_mmol_g": mean_absolute_error(
                test_observations.ravel(), prediction.ravel()
            ),
            "mean_pressure_normalized_rmse": float(np.mean(normalized_rmse_values)),
        }
    )

metrics_df = pd.DataFrame(metric_rows)
summary_df = pd.DataFrame(summary_rows).sort_values("mean_pressure_normalized_rmse")
metrics_df.to_csv(OUTPUT_DIR / "baseline_comparison_metrics.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "baseline_comparison_summary.csv", index=False)
display(summary_df)
display(metrics_df.pivot_table(index=["gas", "pressure_bar"], columns="model", values="r2"))

prediction_output = pd.DataFrame(
    {
        "row_index": test_indices,
        "mof_id": expected_test_ids,
    }
)
for condition_index, (gas, pressure) in enumerate(CONDITIONS):
    pressure_label = f"{pressure:.2f}".replace(".", "p")
    prefix = f"{gas.lower()}_{pressure_label}bar"
    prediction_output[f"true_{prefix}_mmol_g"] = test_observations[:, condition_index]
    for model_name, prediction in all_predictions.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        prediction_output[f"{safe_name}_{prefix}_mmol_g"] = prediction[:, condition_index]
prediction_output.to_csv(OUTPUT_DIR / "baseline_comparison_predictions.csv", index=False)

# %% [markdown]
# ## 5. Paired bootstrap: baseline RMSE minus fusion RMSE

# %%
rng = np.random.default_rng(SEED)
bootstrap_rows = []
fusion_prediction = all_predictions["Final fusion model"]
for baseline_name in ("CatBoost (30 descriptors)", "MLP (30 descriptors)"):
    baseline_prediction = all_predictions[baseline_name]
    for condition_index, (gas, pressure) in enumerate(CONDITIONS):
        observed = test_observations[:, condition_index]
        differences = []
        for _ in range(1_000):
            indices = rng.integers(0, len(observed), size=len(observed))
            baseline_rmse = np.sqrt(
                np.mean((baseline_prediction[indices, condition_index] - observed[indices]) ** 2)
            )
            fusion_rmse = np.sqrt(
                np.mean((fusion_prediction[indices, condition_index] - observed[indices]) ** 2)
            )
            differences.append(baseline_rmse - fusion_rmse)
        lower, median, upper = np.percentile(differences, [2.5, 50.0, 97.5])
        bootstrap_rows.append(
            {
                "baseline": baseline_name,
                "gas": gas,
                "pressure_bar": pressure,
                "rmse_difference_baseline_minus_fusion_median": median,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "fusion_significantly_better_at_95pct": bool(lower > 0),
            }
        )

bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(OUTPUT_DIR / "paired_bootstrap_rmse_difference.csv", index=False)
display(bootstrap_df)

# %% [markdown]
# ## 6. Figures and run metadata

# %%
condition_labels = [f"{gas}\n{pressure:g} bar" for gas, pressure in CONDITIONS]
model_order = list(all_predictions)
colors = plt.cm.tab10(np.linspace(0, 1, len(model_order)))

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for color, model_name in zip(colors, model_order):
    selected = metrics_df[metrics_df["model"] == model_name]
    axes[0].plot(condition_labels, selected["r2"], marker="o", label=model_name, color=color)
    axes[1].plot(
        condition_labels,
        selected["normalized_rmse"],
        marker="o",
        label=model_name,
        color=color,
    )
axes[0].set_ylabel("R2")
axes[0].set_title("Baseline Comparison on the Fixed Independent Test Set")
axes[0].grid(alpha=0.25)
axes[0].legend(ncol=2)
axes[1].set_ylabel("Pressure-normalized RMSE")
axes[1].set_xlabel("Gas and pressure condition")
axes[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "baseline_comparison_by_condition.png", dpi=220)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ordered = summary_df.sort_values("mean_pressure_normalized_rmse", ascending=True)
ax.barh(ordered["model"], ordered["mean_pressure_normalized_rmse"], color="#3877B2")
ax.set_xlabel("Mean pressure-normalized RMSE")
ax.set_title("Overall Baseline Comparison")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "baseline_comparison_overall.png", dpi=220)
plt.show()

# %% [markdown]
# ## 7. Korean figures

# %%
def configure_korean_font():
    preferred_system_fonts = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    font_files = [path for path in preferred_system_fonts if path.exists()]

    if not font_files:
        drive_root = Path("/content/drive/MyDrive")
        zip_candidates = [
            drive_root / "nanum-gothic (1).zip",
            drive_root / "nanum-gothic.zip",
            BASE_DIR / "nanum-gothic (1).zip",
            BASE_DIR / "nanum-gothic.zip",
        ]
        zip_candidates.extend(sorted(drive_root.glob("nanum-gothic*.zip")))
        font_zip = next((path for path in zip_candidates if path.exists()), None)
        if font_zip is None:
            raise FileNotFoundError(
                "Nanum Gothic ZIP was not found in MyDrive. "
                "Place nanum-gothic (1).zip in /content/drive/MyDrive."
            )
        font_directory = LOCAL_DIR / "nanum_gothic"
        font_directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(font_zip) as archive:
            archive.extractall(font_directory)
        font_files = sorted(font_directory.rglob("*.ttf"))

    if not font_files:
        raise FileNotFoundError("No TTF font was found after extracting Nanum Gothic.")
    for font_path in font_files:
        font_manager.fontManager.addfont(str(font_path))

    regular_font = next(
        (
            path
            for path in font_files
            if path.name.lower() in {"nanumgothic.ttf", "nanumgothicregular.ttf"}
        ),
        font_files[0],
    )
    font_name = font_manager.FontProperties(fname=str(regular_font)).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    print("Korean font:", font_name, regular_font)


configure_korean_font()

model_name_ko = {
    "CatBoost (30 descriptors)": "CatBoost (기술자 30개)",
    "MLP (30 descriptors)": "MLP (기술자 30개)",
    "Structure auxiliary head": "구조 타워 보조 출력",
    "Descriptor Transformer auxiliary head": "기술자 Transformer 보조 출력",
    "Final fusion model": "최종 융합 모델",
}
condition_labels_ko = [
    f"{'CO₂' if gas == 'CO2' else 'CH₄'}\n{pressure:g} bar"
    for gas, pressure in CONDITIONS
]

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for color, model_name in zip(colors, model_order):
    selected = metrics_df[metrics_df["model"] == model_name]
    label = model_name_ko[model_name]
    axes[0].plot(
        condition_labels_ko,
        selected["r2"],
        marker="o",
        label=label,
        color=color,
    )
    axes[1].plot(
        condition_labels_ko,
        selected["normalized_rmse"],
        marker="o",
        label=label,
        color=color,
    )
axes[0].set_ylabel("R²")
axes[0].set_title("고정 독립 시험 세트에서의 기준 모델 성능 비교")
axes[0].grid(alpha=0.25)
axes[0].legend(ncol=2)
axes[1].set_ylabel("압력 조건별 정규화 RMSE")
axes[1].set_xlabel("가스 및 압력 조건")
axes[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "baseline_comparison_by_condition_ko.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5.5))
ordered_ko = summary_df.sort_values(
    "mean_pressure_normalized_rmse", ascending=True
).copy()
ordered_ko["model_ko"] = ordered_ko["model"].map(model_name_ko)
bar_colors = [
    "#16A6B6" if model == "Final fusion model" else "#4C78A8"
    for model in ordered_ko["model"]
]
ax.barh(
    ordered_ko["model_ko"],
    ordered_ko["mean_pressure_normalized_rmse"],
    color=bar_colors,
)
ax.invert_yaxis()
ax.set_xlabel("평균 압력 조건별 정규화 RMSE")
ax.set_title("기준 모델의 전체 성능 비교")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "baseline_comparison_overall_ko.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()

run_metadata = {
    "seed": SEED,
    "split_sizes": {"train": 35_000, "validation": 5_000, "test": 10_000},
    "descriptor_columns": descriptor_columns,
    "target_columns": TARGET_COLUMNS,
    "target_transform": "Per-condition standardized log1p fitted on training rows only",
    "catboost": {
        "version": catboost.__version__,
        "parameters": catboost_parameters,
        "tree_count": int(catboost_model.tree_count_),
        "model_file_size_bytes": catboost_model_path.stat().st_size,
        "training_seconds": catboost_seconds,
    },
    "mlp": {
        "parameter_count": mlp_parameter_count,
        "best_epoch": int(mlp_checkpoint["epoch"]),
        "best_validation_loss": float(mlp_checkpoint["best_val_loss"]),
        "training_seconds": mlp_seconds,
    },
    "software": {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "catboost": catboost.__version__,
    },
    "important_note": (
        "Structure and descriptor Transformer values are auxiliary heads from the jointly "
        "trained dual-tower model; CatBoost and MLP are independently trained baselines."
    ),
}
(OUTPUT_DIR / "baseline_comparison_run_metadata.json").write_text(
    json.dumps(run_metadata, indent=2), encoding="utf-8"
)

print("All outputs saved to:", OUTPUT_DIR)
print("CatBoost official MultiRMSE supports GPU training; GPU reductions can be nondeterministic.")
