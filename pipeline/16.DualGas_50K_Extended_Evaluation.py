# %% [markdown]
# # Extended post-hoc evaluation for the live 50K CO2/CH4 runtime
#
# Run this file in the same Colab runtime after the main training notebook:
# `%run -i "/content/drive/MyDrive/mf_hid/4th/16.DualGas_50K_Extended_Evaluation.py"`

# %%
import copy
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

EXTENDED_EVAL_SEED = 20260721
N_BOOTSTRAPS = 1000
FP32_ROTATION_SAMPLES = 200
ABLATION_TEST_SAMPLES = 2000
OUTPUT_DIR = Path(OUTPUT_DIR)
EXTENDED_OUTPUT_DIR = OUTPUT_DIR / "extended_evaluation"
EXTENDED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

required_runtime_names = [
    "model",
    "test_result",
    "test_loader",
    "test_indices",
    "train_indices",
    "descriptor_all",
    "DESCRIPTOR_COLUMNS",
    "descriptor_pipeline",
    "USED_NODE_FEATURE_NAMES",
    "target_scale",
    "targets",
    "df",
    "h5_ids",
    "CONDITIONS",
    "PRESSURES_BAR",
    "CO2_PRESSURES_BAR",
    "CH4_PRESSURES_BAR",
    "AMP_DTYPE",
    "DEVICE",
    "move_batch_to_device",
    "collate_mofs",
    "EnrichedH5Dataset",
    "make_loader",
]
missing_runtime_names = [name for name in required_runtime_names if name not in globals()]
if missing_runtime_names:
    raise RuntimeError(
        "Run this script in the live training runtime. Missing names: "
        + ", ".join(missing_runtime_names)
    )

rng = np.random.default_rng(EXTENDED_EVAL_SEED)
observations = np.asarray(test_result["observations"], dtype=np.float64)
predictions_by_tower = {
    tower: np.asarray(values, dtype=np.float64)
    for tower, values in test_result["predictions"].items()
}
fusion_predictions = predictions_by_tower["fusion"]
test_mof_ids = np.asarray(test_result["mof_ids"], dtype=str)
test_graph_indices = np.asarray(test_result["graph_indices"], dtype=np.int64)
target_scale_float = np.asarray(target_scale, dtype=np.float64)

if observations.shape != (len(test_indices), len(CONDITIONS)):
    raise RuntimeError(f"Unexpected test-result shape: {observations.shape}")

print("Extended evaluation folder:", EXTENDED_OUTPUT_DIR)
print("Reusing test predictions:", observations.shape)


def condition_label(index):
    condition = CONDITIONS[index]
    return f"{condition['gas']} {condition['pressure_bar']:g} bar"


def safe_json_dump(payload, path):
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )


# %% [markdown]
# ## 1. MOF-level bootstrap confidence intervals

# %%
def bootstrap_condition_metrics(y_true, y_pred, n_bootstraps, random_state):
    local_rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    r2_values = np.empty(n_bootstraps, dtype=np.float64)
    rmse_values = np.empty(n_bootstraps, dtype=np.float64)
    mae_values = np.empty(n_bootstraps, dtype=np.float64)
    for bootstrap_index in range(n_bootstraps):
        sample = local_rng.integers(0, n_samples, size=n_samples)
        observed_sample = y_true[sample]
        predicted_sample = y_pred[sample]
        residual = observed_sample - predicted_sample
        r2_values[bootstrap_index] = r2_score(observed_sample, predicted_sample)
        rmse_values[bootstrap_index] = np.sqrt(np.mean(residual * residual))
        mae_values[bootstrap_index] = np.mean(np.abs(residual))
    return r2_values, rmse_values, mae_values


bootstrap_rows = []
for condition_index, condition in enumerate(tqdm(CONDITIONS, desc="Bootstrap CIs")):
    y_true = observations[:, condition_index]
    y_pred = fusion_predictions[:, condition_index]
    distributions = bootstrap_condition_metrics(
        y_true,
        y_pred,
        N_BOOTSTRAPS,
        EXTENDED_EVAL_SEED + condition_index,
    )
    for metric_name, point, distribution in (
        ("r2", r2_score(y_true, y_pred), distributions[0]),
        ("rmse_mmol_g", np.sqrt(mean_squared_error(y_true, y_pred)), distributions[1]),
        ("mae_mmol_g", mean_absolute_error(y_true, y_pred), distributions[2]),
    ):
        bootstrap_rows.append(
            {
                "gas": condition["gas"],
                "pressure_bar": condition["pressure_bar"],
                "metric": metric_name,
                "point_estimate": point,
                "ci95_lower": np.percentile(distribution, 2.5),
                "ci95_upper": np.percentile(distribution, 97.5),
                "n_bootstraps": N_BOOTSTRAPS,
                "resampling_unit": "MOF",
            }
        )

bootstrap_ci_df = pd.DataFrame(bootstrap_rows)
bootstrap_ci_df.to_csv(
    EXTENDED_OUTPUT_DIR / "fusion_metric_bootstrap_ci.csv", index=False
)
display(bootstrap_ci_df)


# %% [markdown]
# ## 2. Paired tower comparison with confidence intervals

# %%
paired_rows = []
for condition_index, condition in enumerate(tqdm(CONDITIONS, desc="Paired tower CIs")):
    y_true = observations[:, condition_index]
    fusion_absolute_error = np.abs(fusion_predictions[:, condition_index] - y_true)
    for baseline_tower in ("structure", "descriptor"):
        baseline_absolute_error = np.abs(
            predictions_by_tower[baseline_tower][:, condition_index] - y_true
        )
        per_mof_improvement = baseline_absolute_error - fusion_absolute_error
        sample_means = np.empty(N_BOOTSTRAPS, dtype=np.float64)
        local_rng = np.random.default_rng(
            EXTENDED_EVAL_SEED + 100 + 10 * condition_index + (baseline_tower == "descriptor")
        )
        for bootstrap_index in range(N_BOOTSTRAPS):
            sample = local_rng.integers(0, len(y_true), size=len(y_true))
            sample_means[bootstrap_index] = per_mof_improvement[sample].mean()
        paired_rows.append(
            {
                "gas": condition["gas"],
                "pressure_bar": condition["pressure_bar"],
                "comparison": f"fusion_vs_{baseline_tower}",
                "baseline_mae_mmol_g": baseline_absolute_error.mean(),
                "fusion_mae_mmol_g": fusion_absolute_error.mean(),
                "mae_reduction_mmol_g": per_mof_improvement.mean(),
                "mae_reduction_pct": 100.0
                * per_mof_improvement.mean()
                / max(baseline_absolute_error.mean(), 1.0e-12),
                "ci95_lower_mmol_g": np.percentile(sample_means, 2.5),
                "ci95_upper_mmol_g": np.percentile(sample_means, 97.5),
                "probability_fusion_better": np.mean(sample_means > 0.0),
            }
        )

paired_tower_df = pd.DataFrame(paired_rows)
paired_tower_df.to_csv(
    EXTENDED_OUTPUT_DIR / "paired_tower_mae_bootstrap.csv", index=False
)
display(paired_tower_df)


# %% [markdown]
# ## 3. Calibration, bias, rank correlation, and normalized DCG

# %%
def ndcg_at_fraction(y_true, y_score, fraction):
    k = max(1, int(math.ceil(len(y_true) * fraction)))
    predicted_order = np.argsort(y_score)[::-1][:k]
    ideal_order = np.argsort(y_true)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(y_true[predicted_order] * discounts)
    ideal_dcg = np.sum(y_true[ideal_order] * discounts)
    return dcg / max(ideal_dcg, 1.0e-12)


diagnostic_rows = []
calibration_rows = []
for condition_index, condition in enumerate(CONDITIONS):
    y_true = observations[:, condition_index]
    y_pred = fusion_predictions[:, condition_index]
    residual = y_pred - y_true
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    diagnostic_rows.append(
        {
            "gas": condition["gas"],
            "pressure_bar": condition["pressure_bar"],
            "calibration_slope": slope,
            "calibration_intercept_mmol_g": intercept,
            "mean_bias_pred_minus_true_mmol_g": residual.mean(),
            "median_bias_mmol_g": np.median(residual),
            "residual_std_mmol_g": residual.std(ddof=1),
            "spearman_rho": spearmanr(y_true, y_pred).statistic,
            "kendall_tau": kendalltau(y_true, y_pred).statistic,
            "ndcg_top_1pct": ndcg_at_fraction(y_true, y_pred, 0.01),
            "ndcg_top_5pct": ndcg_at_fraction(y_true, y_pred, 0.05),
            "ndcg_top_10pct": ndcg_at_fraction(y_true, y_pred, 0.10),
        }
    )
    bins = pd.qcut(pd.Series(y_pred).rank(method="first"), 10, labels=False).to_numpy()
    for bin_index in range(10):
        mask = bins == bin_index
        calibration_rows.append(
            {
                "gas": condition["gas"],
                "pressure_bar": condition["pressure_bar"],
                "predicted_decile": bin_index + 1,
                "n": int(mask.sum()),
                "mean_predicted_mmol_g": y_pred[mask].mean(),
                "mean_observed_mmol_g": y_true[mask].mean(),
                "mean_bias_mmol_g": residual[mask].mean(),
            }
        )

diagnostics_df = pd.DataFrame(diagnostic_rows)
calibration_df = pd.DataFrame(calibration_rows)
diagnostics_df.to_csv(
    EXTENDED_OUTPUT_DIR / "calibration_bias_and_ranking.csv", index=False
)
calibration_df.to_csv(
    EXTENDED_OUTPUT_DIR / "prediction_decile_calibration.csv", index=False
)
display(diagnostics_df)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for condition_index, axis in enumerate(axes.ravel()):
    condition = CONDITIONS[condition_index]
    rows = calibration_df[
        (calibration_df["gas"] == condition["gas"])
        & np.isclose(calibration_df["pressure_bar"], condition["pressure_bar"])
    ]
    limit = max(rows["mean_observed_mmol_g"].max(), rows["mean_predicted_mmol_g"].max())
    axis.plot(rows["mean_observed_mmol_g"], rows["mean_predicted_mmol_g"], "o-")
    axis.plot([0, limit], [0, limit], "k--", linewidth=1)
    axis.set(title=condition_label(condition_index), xlabel="Observed", ylabel="Predicted")
    axis.grid(alpha=0.25)
fig.suptitle("Prediction-Decile Calibration (mmol g$^{-1}$)", fontsize=15)
fig.tight_layout()
fig.savefig(EXTENDED_OUTPUT_DIR / "prediction_decile_calibration.png", dpi=220)
plt.show()


# %% [markdown]
# ## 4. Working capacity and pure-component selectivity proxies

# %%
derived_tasks = {
    "CO2_working_capacity_2p5_minus_0p01bar": (4, 0),
    "CO2_working_capacity_2p5_minus_0p5bar": (4, 3),
    "CH4_working_capacity_4p5_minus_0p05bar": (9, 5),
    "CH4_working_capacity_4p5_minus_0p5bar": (9, 6),
}
derived_rows = []
derived_ranking_rows = []
for task_name, (high_index, low_index) in derived_tasks.items():
    y_true = observations[:, high_index] - observations[:, low_index]
    y_pred = fusion_predictions[:, high_index] - fusion_predictions[:, low_index]
    derived_rows.append(
        {
            "task": task_name,
            "r2": r2_score(y_true, y_pred),
            "rmse_mmol_g": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae_mmol_g": mean_absolute_error(y_true, y_pred),
            "spearman_rho": spearmanr(y_true, y_pred).statistic,
        }
    )
    for fraction in (0.01, 0.05, 0.10):
        k = max(1, int(math.ceil(len(y_true) * fraction)))
        true_top = set(np.argpartition(y_true, -k)[-k:])
        predicted_top = set(np.argpartition(y_pred, -k)[-k:])
        derived_ranking_rows.append(
            {
                "task": task_name,
                "top_fraction": fraction,
                "k": k,
                "recall_at_k": len(true_top & predicted_top) / k,
                "ndcg_at_k": ndcg_at_fraction(y_true, y_pred, fraction),
            }
        )

common_pressure_indices = {
    0.05: (1, 5),
    0.5: (3, 6),
    2.5: (4, 8),
}
selectivity_rows = []
selectivity_ranking_rows = []
for pressure, (co2_index, ch4_index) in common_pressure_indices.items():
    epsilon = 0.01
    true_log_ratio = np.log10(
        (observations[:, co2_index] + epsilon) / (observations[:, ch4_index] + epsilon)
    )
    predicted_log_ratio = np.log10(
        (fusion_predictions[:, co2_index] + epsilon)
        / (fusion_predictions[:, ch4_index] + epsilon)
    )
    selectivity_rows.append(
        {
            "pressure_bar": pressure,
            "definition": "log10((CO2 uptake + 0.01)/(CH4 uptake + 0.01))",
            "r2_log_ratio": r2_score(true_log_ratio, predicted_log_ratio),
            "rmse_log10_ratio": np.sqrt(
                mean_squared_error(true_log_ratio, predicted_log_ratio)
            ),
            "mae_log10_ratio": mean_absolute_error(true_log_ratio, predicted_log_ratio),
            "spearman_rho": spearmanr(true_log_ratio, predicted_log_ratio).statistic,
        }
    )
    for fraction in (0.01, 0.05, 0.10):
        k = max(1, int(math.ceil(len(true_log_ratio) * fraction)))
        true_top = set(np.argpartition(true_log_ratio, -k)[-k:])
        predicted_top = set(np.argpartition(predicted_log_ratio, -k)[-k:])
        selectivity_ranking_rows.append(
            {
                "pressure_bar": pressure,
                "top_fraction": fraction,
                "k": k,
                "recall_at_k": len(true_top & predicted_top) / k,
                "ndcg_at_k": ndcg_at_fraction(
                    true_log_ratio - true_log_ratio.min(),
                    predicted_log_ratio,
                    fraction,
                ),
            }
        )

derived_metrics_df = pd.DataFrame(derived_rows)
derived_ranking_df = pd.DataFrame(derived_ranking_rows)
selectivity_metrics_df = pd.DataFrame(selectivity_rows)
selectivity_ranking_df = pd.DataFrame(selectivity_ranking_rows)
derived_metrics_df.to_csv(EXTENDED_OUTPUT_DIR / "working_capacity_metrics.csv", index=False)
derived_ranking_df.to_csv(EXTENDED_OUTPUT_DIR / "working_capacity_ranking.csv", index=False)
selectivity_metrics_df.to_csv(
    EXTENDED_OUTPUT_DIR / "pure_component_selectivity_proxy_metrics.csv", index=False
)
selectivity_ranking_df.to_csv(
    EXTENDED_OUTPUT_DIR / "pure_component_selectivity_proxy_ranking.csv", index=False
)
display(derived_metrics_df)
display(selectivity_metrics_df)


# %% [markdown]
# ## 5. Geometric, metal, and internal applicability-domain robustness

# %%
test_frame = df.iloc[test_graph_indices].reset_index(drop=True)
normalized_absolute_error = np.abs(fusion_predictions - observations) / target_scale_float

subgroup_rows = []
for feature in ("surface_area_m2g", "void_fraction", "lcd", "pld"):
    if feature not in test_frame.columns:
        continue
    values = pd.to_numeric(test_frame[feature], errors="coerce")
    valid = values.notna().to_numpy()
    bins = np.full(len(values), -1, dtype=np.int64)
    bins[valid] = pd.qcut(
        values[valid].rank(method="first"), 5, labels=False
    ).to_numpy()
    for quintile in range(5):
        mask = bins == quintile
        subgroup_rows.append(
            {
                "subgroup_type": "geometry_quintile",
                "subgroup_feature": feature,
                "subgroup": f"Q{quintile + 1}",
                "n": int(mask.sum()),
                "feature_mean": values[mask].mean(),
                "mean_normalized_absolute_error": normalized_absolute_error[mask].mean(),
                "co2_mean_normalized_absolute_error": normalized_absolute_error[mask, :5].mean(),
                "ch4_mean_normalized_absolute_error": normalized_absolute_error[mask, 5:].mean(),
            }
        )

metal_columns = [column for column in test_frame.columns if column.startswith("metal_type_")]
for metal_column in metal_columns:
    mask = pd.to_numeric(test_frame[metal_column], errors="coerce").fillna(0).to_numpy() > 0
    if mask.sum() < 100:
        continue
    subgroup_rows.append(
        {
            "subgroup_type": "metal_presence",
            "subgroup_feature": metal_column,
            "subgroup": metal_column.removeprefix("metal_type_"),
            "n": int(mask.sum()),
            "feature_mean": 1.0,
            "mean_normalized_absolute_error": normalized_absolute_error[mask].mean(),
            "co2_mean_normalized_absolute_error": normalized_absolute_error[mask, :5].mean(),
            "ch4_mean_normalized_absolute_error": normalized_absolute_error[mask, 5:].mean(),
        }
    )

subgroup_df = pd.DataFrame(subgroup_rows)
subgroup_df.to_csv(EXTENDED_OUTPUT_DIR / "subgroup_robustness.csv", index=False)

# This is an internal novelty diagnostic, not a true external OOD benchmark.
descriptor_scaler = StandardScaler()
train_descriptor = descriptor_scaler.fit_transform(descriptor_all[train_indices])
test_descriptor = descriptor_scaler.transform(descriptor_all[test_graph_indices])
n_components = min(16, train_descriptor.shape[1], len(train_indices) - 1)
pca = PCA(n_components=n_components, random_state=EXTENDED_EVAL_SEED)
train_pca = pca.fit_transform(train_descriptor)
test_pca = pca.transform(test_descriptor)
nearest_neighbor = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
nearest_neighbor.fit(train_pca)
nearest_distance = nearest_neighbor.kneighbors(test_pca, return_distance=True)[0][:, 0]
distance_quintile = pd.qcut(
    pd.Series(nearest_distance).rank(method="first"), 5, labels=False
).to_numpy()
applicability_rows = []
for quintile in range(5):
    mask = distance_quintile == quintile
    applicability_rows.append(
        {
            "descriptor_distance_quintile": quintile + 1,
            "n": int(mask.sum()),
            "mean_nearest_train_distance": nearest_distance[mask].mean(),
            "mean_normalized_absolute_error": normalized_absolute_error[mask].mean(),
            "co2_mean_normalized_absolute_error": normalized_absolute_error[mask, :5].mean(),
            "ch4_mean_normalized_absolute_error": normalized_absolute_error[mask, 5:].mean(),
        }
    )
applicability_df = pd.DataFrame(applicability_rows)
applicability_df.to_csv(
    EXTENDED_OUTPUT_DIR / "internal_applicability_domain.csv", index=False
)
display(applicability_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
geometry_rows = subgroup_df[subgroup_df["subgroup_type"] == "geometry_quintile"]
for feature in geometry_rows["subgroup_feature"].drop_duplicates():
    rows = geometry_rows[geometry_rows["subgroup_feature"] == feature]
    axes[0].plot(
        np.arange(1, 6), rows["mean_normalized_absolute_error"], "o-", label=feature
    )
axes[0].set(
    title="Error Across Geometric-Descriptor Quintiles",
    xlabel="Descriptor quintile",
    ylabel="Mean normalized absolute error",
    xticks=np.arange(1, 6),
)
axes[0].legend()
axes[1].plot(
    applicability_df["descriptor_distance_quintile"],
    applicability_df["mean_normalized_absolute_error"],
    "o-",
)
axes[1].set(
    title="Internal Applicability-Domain Diagnostic",
    xlabel="Nearest-training-descriptor distance quintile",
    ylabel="Mean normalized absolute error",
    xticks=np.arange(1, 6),
)
for axis in axes:
    axis.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(EXTENDED_OUTPUT_DIR / "subgroup_and_applicability_domain.png", dpi=220)
plt.show()


# %% [markdown]
# ## 6. Grouped input ablation on 2,000 fixed test MOFs

# %%
ablation_rng = np.random.default_rng(EXTENDED_EVAL_SEED)
ablation_global_indices = np.sort(
    ablation_rng.choice(
        test_indices,
        size=min(ABLATION_TEST_SAMPLES, len(test_indices)),
        replace=False,
    )
)
_, _, ablation_loader = make_loader(ablation_global_indices, False)

raw_feature_count = len(DESCRIPTOR_COLUMNS)
indicator_raw_indices = descriptor_pipeline.named_steps["imputer"].indicator_.features_
indicator_position = {
    int(raw_index): raw_feature_count + indicator_index
    for indicator_index, raw_index in enumerate(indicator_raw_indices)
}
descriptor_column_index = {
    column: index for index, column in enumerate(DESCRIPTOR_COLUMNS)
}
node_feature_index = {
    feature: index for index, feature in enumerate(USED_NODE_FEATURE_NAMES)
}


def transformed_descriptor_indices(columns):
    indices = []
    for column in columns:
        raw_index = descriptor_column_index.get(column)
        if raw_index is None:
            continue
        indices.append(raw_index)
        if raw_index in indicator_position:
            indices.append(indicator_position[raw_index])
    return sorted(set(indices))


descriptor_groups = {
    "descriptor_geometry": [
        column
        for column in ("surface_area_m2g", "void_fraction", "lcd", "pld")
        if column in descriptor_column_index
    ],
    "descriptor_alchemi_summary": [
        column
        for column in DESCRIPTOR_COLUMNS
        if column.startswith("local_density_")
        or column.endswith("_bond_fraction")
        or column in {"spatial_only_edge_fraction", "mean_edge_bond_score", "n_atoms_cif"}
    ],
    "descriptor_metal_and_oms": [
        column
        for column in DESCRIPTOR_COLUMNS
        if column.startswith("metal_type_")
        or column.startswith("metal_")
        or column.startswith("open_metal_site_")
    ],
    "descriptor_ligand_and_pore_chemistry": [
        column
        for column in DESCRIPTOR_COLUMNS
        if column.startswith("ligand_") or column.startswith("pore_proxy_")
    ],
}

node_groups = {
    "node_element_properties": [
        feature
        for feature in USED_NODE_FEATURE_NAMES
        if feature
        not in {
            "coordination_number",
            "soft_coordination_number",
            "local_density_r3",
            "local_density_r6",
        }
    ],
    "node_coordination": ["coordination_number", "soft_coordination_number"],
    "node_local_density": ["local_density_r3", "local_density_r6"],
}


@torch.no_grad()
def predict_ablation(loader, ablation_name):
    model.eval()
    predictions = []
    observations_local = []
    graph_indices_local = []
    for batch in tqdm(loader, desc=ablation_name, leave=False):
        graph_indices_local.extend(batch["graph_index"].numpy().tolist())
        batch = move_batch_to_device(batch)

        if ablation_name in descriptor_groups:
            indices = transformed_descriptor_indices(descriptor_groups[ablation_name])
            if indices:
                batch["descriptor"][:, indices] = 0.0
        elif ablation_name == "descriptor_all":
            batch["descriptor"].zero_()
        elif ablation_name in node_groups:
            indices = [
                node_feature_index[name]
                for name in node_groups[ablation_name]
                if name in node_feature_index
            ]
            batch["x_cont"][:, indices] = 0.0
            batch["x_cont_mask"][:, indices] = False
        elif ablation_name == "edge_bond_score":
            batch["edge_bond_score"].zero_()
        elif ablation_name == "edge_relation":
            batch["edge_relation"].zero_()
        elif ablation_name == "alchemi_local_and_edges":
            names = [
                "coordination_number",
                "soft_coordination_number",
                "local_density_r3",
                "local_density_r6",
            ]
            indices = [node_feature_index[name] for name in names]
            batch["x_cont"][:, indices] = 0.0
            batch["x_cont_mask"][:, indices] = False
            batch["edge_bond_score"].zero_()
            batch["edge_relation"].zero_()

        with torch.autocast("cuda", dtype=AMP_DTYPE):
            output = model(batch)["fusion"]
        predictions.append(output.float().cpu().numpy())
        observations_local.append(batch["target"].float().cpu().numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(observations_local),
        np.asarray(graph_indices_local),
    )


ablation_names = [
    "baseline",
    *descriptor_groups.keys(),
    "descriptor_all",
    *node_groups.keys(),
    "edge_bond_score",
    "edge_relation",
    "alchemi_local_and_edges",
]
ablation_rows = []
baseline_normalized_rmse = None
for ablation_name in ablation_names:
    predicted, observed, evaluated_indices = predict_ablation(
        ablation_loader, ablation_name
    )
    if not np.array_equal(evaluated_indices, ablation_global_indices):
        raise RuntimeError("Ablation loader changed graph order.")
    pressure_rmse = np.sqrt(np.mean((observed - predicted) ** 2, axis=0))
    normalized_rmse = pressure_rmse / target_scale_float
    if ablation_name == "baseline":
        baseline_normalized_rmse = normalized_rmse
    row = {
        "ablation": ablation_name,
        "n_mofs": len(observed),
        "mean_normalized_rmse": normalized_rmse.mean(),
        "co2_mean_normalized_rmse": normalized_rmse[:5].mean(),
        "ch4_mean_normalized_rmse": normalized_rmse[5:].mean(),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        label = f"{condition['gas'].lower()}_{condition['pressure_bar']:g}bar".replace(
            ".", "p"
        )
        row[f"normalized_rmse_{label}"] = normalized_rmse[condition_index]
        row[f"delta_normalized_rmse_{label}"] = (
            normalized_rmse[condition_index]
            - baseline_normalized_rmse[condition_index]
        )
    row["importance_delta_mean_normalized_rmse"] = (
        normalized_rmse - baseline_normalized_rmse
    ).mean()
    ablation_rows.append(row)

ablation_df = pd.DataFrame(ablation_rows).sort_values(
    "importance_delta_mean_normalized_rmse", ascending=False
)
ablation_df.to_csv(EXTENDED_OUTPUT_DIR / "grouped_input_ablation.csv", index=False)
display(ablation_df)

plot_rows = ablation_df[ablation_df["ablation"] != "baseline"].sort_values(
    "importance_delta_mean_normalized_rmse"
)
fig, axis = plt.subplots(figsize=(10, 7))
colors = ["#D55E00" if value >= 0 else "#7A7A7A" for value in plot_rows["importance_delta_mean_normalized_rmse"]]
axis.barh(
    plot_rows["ablation"],
    plot_rows["importance_delta_mean_normalized_rmse"],
    color=colors,
)
axis.axvline(0.0, color="black", linewidth=1)
axis.set(
    title="Grouped Input Ablation on the Independent Test Subset",
    xlabel="Increase in mean pressure-normalized RMSE",
    ylabel="Ablated input group",
)
axis.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(EXTENDED_OUTPUT_DIR / "grouped_input_ablation.png", dpi=220)
plt.show()


# %% [markdown]
# ## 7. Full-precision rotation-invariance test

# %%
def random_rotation_matrix_fp32(generator):
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


rotation_count = min(FP32_ROTATION_SAMPLES, len(test_indices))
rotation_dataset = EnrichedH5Dataset(test_indices[:rotation_count])
rotation_generator = torch.Generator().manual_seed(EXTENDED_EVAL_SEED)
fp32_rotation_rows = []
model.eval()
for sample_index in tqdm(range(rotation_count), desc="FP32 rotation test"):
    original = collate_mofs([rotation_dataset[sample_index]])
    rotated = {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in original.items()
    }
    rotation = random_rotation_matrix_fp32(rotation_generator)
    rotated["edge_vec"] = rotated["edge_vec"].float() @ rotation.T
    original = move_batch_to_device(original)
    rotated = move_batch_to_device(rotated)
    with torch.no_grad():
        original_prediction = model(original)["fusion"].float()
        rotated_prediction = model(rotated)["fusion"].float()
    absolute_difference = (
        original_prediction - rotated_prediction
    ).abs().squeeze(0).cpu().numpy()
    row = {
        "mof_id": rotation_dataset[sample_index]["mof_id"],
        "maximum_absolute_difference_mmol_g": absolute_difference.max(),
        "mean_absolute_difference_mmol_g": absolute_difference.mean(),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        label = f"{condition['gas'].lower()}_{condition['pressure_bar']:g}bar".replace(
            ".", "p"
        )
        row[f"absolute_difference_{label}_mmol_g"] = absolute_difference[
            condition_index
        ]
    fp32_rotation_rows.append(row)

fp32_rotation_df = pd.DataFrame(fp32_rotation_rows)
fp32_rotation_df.to_csv(
    EXTENDED_OUTPUT_DIR / "rotation_invariance_fp32.csv", index=False
)
display(fp32_rotation_df.describe())

fig, axis = plt.subplots(figsize=(8, 5))
axis.hist(
    fp32_rotation_df["maximum_absolute_difference_mmol_g"],
    bins=24,
    color="#4C78A8",
    edgecolor="black",
)
axis.set(
    title="Full-Precision Rotation-Invariance Stress Test",
    xlabel="Maximum absolute prediction difference (mmol g$^{-1}$)",
    ylabel="Number of MOFs",
)
axis.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(EXTENDED_OUTPUT_DIR / "rotation_invariance_fp32.png", dpi=220)
plt.show()


# %% [markdown]
# ## 8. Repeated-outlier audit and final manifest

# %%
outlier_rows = []
for sample_index, mof_id in enumerate(test_mof_ids):
    absolute_error = np.abs(fusion_predictions[sample_index] - observations[sample_index])
    normalized_error = absolute_error / target_scale_float
    relative_error = absolute_error / np.maximum(observations[sample_index], 0.05)
    outlier_rows.append(
        {
            "mof_id": mof_id,
            "graph_index": int(test_graph_indices[sample_index]),
            "mean_normalized_absolute_error": normalized_error.mean(),
            "maximum_normalized_absolute_error": normalized_error.max(),
            "mean_floor_relative_error": relative_error.mean(),
            "conditions_in_worst_1pct": 0,
        }
    )

outlier_df = pd.DataFrame(outlier_rows)
for condition_index in range(len(CONDITIONS)):
    condition_error = np.abs(
        fusion_predictions[:, condition_index] - observations[:, condition_index]
    ) / target_scale_float[condition_index]
    threshold = np.quantile(condition_error, 0.99)
    outlier_df["conditions_in_worst_1pct"] += condition_error >= threshold
outlier_df = outlier_df.sort_values(
    ["conditions_in_worst_1pct", "mean_normalized_absolute_error"],
    ascending=False,
)
outlier_df.to_csv(EXTENDED_OUTPUT_DIR / "repeated_outlier_audit.csv", index=False)
display(outlier_df.head(30))

summary = {
    "test_mofs": int(len(test_indices)),
    "bootstrap_resamples": N_BOOTSTRAPS,
    "ablation_test_mofs": int(len(ablation_global_indices)),
    "fp32_rotation_test_mofs": int(rotation_count),
    "best_overall_normalized_rmse_from_ablation_subset": float(
        ablation_df.loc[
            ablation_df["ablation"] == "baseline", "mean_normalized_rmse"
        ].iloc[0]
    ),
    "fp32_rotation_maximum_difference_mmol_g": float(
        fp32_rotation_df["maximum_absolute_difference_mmol_g"].max()
    ),
    "fp32_rotation_median_difference_mmol_g": float(
        fp32_rotation_df["maximum_absolute_difference_mmol_g"].median()
    ),
    "files": sorted(path.name for path in EXTENDED_OUTPUT_DIR.iterdir()),
    "important_interpretation_notes": [
        "Pure-component CO2/CH4 uptake ratios are screening proxies, not IAST selectivities.",
        "Descriptor-distance quintiles are internal novelty diagnostics, not external OOD validation.",
        "Input ablation is measured on a fixed 2,000-MOF subset to control runtime.",
        "Bootstrap intervals resample MOFs and do not include training-seed uncertainty.",
    ],
}
safe_json_dump(summary, EXTENDED_OUTPUT_DIR / "extended_evaluation_manifest.json")
print("Extended evaluation completed. Outputs:", EXTENDED_OUTPUT_DIR)
