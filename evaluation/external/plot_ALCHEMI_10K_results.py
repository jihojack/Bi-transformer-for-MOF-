#!/usr/bin/env python3
"""Create publication-ready plots for the external 10K ALCHEMI evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVAL_DIR = Path("/content/drive/MyDrive/mf_hid/4th/재평가")
ROWS_CSV = EVAL_DIR / "external_10K_alchemi_evaluation_rows.csv"
METRICS_CSV = EVAL_DIR / "external_10K_alchemi_per_target_metrics.csv"
METRICS_JSON = EVAL_DIR / "external_10K_alchemi_metrics.json"

TARGETS = [
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
PREDICTIONS = [f"pred_{target}" for target in TARGETS]
LABELS = [
    r"CO$_2$ 0.01 bar",
    r"CO$_2$ 0.05 bar",
    r"CO$_2$ 0.10 bar",
    r"CO$_2$ 0.50 bar",
    r"CO$_2$ 2.50 bar",
    r"CH$_4$ 0.05 bar",
    r"CH$_4$ 0.50 bar",
    r"CH$_4$ 0.90 bar",
    r"CH$_4$ 2.50 bar",
    r"CH$_4$ 4.50 bar",
]

# Previous ASE-based CPU compatibility run on the exact same 10,000 structures.
CPU_REFERENCE = {
    "co2_q_0p01bar": (0.8667221614126217, 0.06709143955858782, 0.022967568129051374),
    "co2_q_0p05bar": (0.9056618598156846, 0.1261531246736529, 0.0613267723320083),
    "co2_adsorption_0.1bar": (0.9147994817847664, 0.16788741922090109, 0.0906576448477726),
    "co2_adsorption_0.5bar": (0.9266059367296408, 0.32532048126982244, 0.20302535282807022),
    "co2_adsorption_2.5bar": (0.9581867159609998, 0.5238180889815461, 0.36918218827070953),
    "ch4_adsorption_0.05bar": (0.8994790049348637, 0.03200473432120998, 0.010560904163814515),
    "ch4_adsorption_0.5bar": (0.9383820809457597, 0.10323545433524797, 0.0538030538623333),
    "ch4_adsorption_0.9bar": (0.9442073440256684, 0.13497710355136058, 0.0762445309379272),
    "ch4_adsorption_2.5bar": (0.955062168422805, 0.20355157411703, 0.13072016835351516),
    "ch4_adsorption_4.5bar": (0.962308802605747, 0.24839976733771602, 0.16772889877216302),
}

OUTPUTS = {
    "scatter": EVAL_DIR / "external_10K_alchemi_scatter_grid.png",
    "metrics": EVAL_DIR / "external_10K_alchemi_metric_summary.png",
    "residuals": EVAL_DIR / "external_10K_alchemi_standardized_residuals.png",
    "equivalence": EVAL_DIR / "ALCHEMI_vs_CPU_metric_difference.png",
    "comparison_csv": EVAL_DIR / "ALCHEMI_vs_CPU_metric_comparison.csv",
}


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


for path in (ROWS_CSV, METRICS_CSV, METRICS_JSON):
    require(path)

rows = pd.read_csv(ROWS_CSV)
metrics = pd.read_csv(METRICS_CSV).set_index("target").loc[TARGETS].reset_index()
summary = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
if len(rows) != 10_000 or rows["mof_id"].nunique() != 10_000:
    raise RuntimeError("Evaluation rows must contain 10,000 unique MOFs.")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
colors = ["#1F77B4"] * 5 + ["#D95F02"] * 5


# 1. True-versus-predicted density plots.
fig, axes = plt.subplots(2, 5, figsize=(18, 7.4), constrained_layout=True)
for index, (target, prediction, label) in enumerate(
    zip(TARGETS, PREDICTIONS, LABELS)
):
    ax = axes.flat[index]
    y_true = rows[target].to_numpy(dtype=float)
    y_pred = rows[prediction].to_numpy(dtype=float)
    upper = float(max(y_true.max(), y_pred.max()))
    upper = max(upper, 1.0e-6)
    ax.hexbin(
        y_true,
        y_pred,
        gridsize=55,
        extent=(0, upper, 0, upper),
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    ax.plot([0, upper], [0, upper], "--", color="#D62728", linewidth=1.2)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    row = metrics.iloc[index]
    ax.set_title(label, color=colors[index], fontweight="bold")
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {row.r2:.3f}\nRMSE = {row.rmse:.3f}\nMAE = {row.mae:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    if index // 5 == 1:
        ax.set_xlabel("GCMC reference (mmol/g)")
    if index % 5 == 0:
        ax.set_ylabel("Model prediction (mmol/g)")
fig.suptitle(
    "External 10K: NVIDIA ALCHEMI preprocessing and frozen-model predictions\n"
    "Hexagon color represents log point density; dashed line is ideal agreement",
    fontsize=15,
    fontweight="bold",
)
fig.savefig(OUTPUTS["scatter"], bbox_inches="tight")
plt.close(fig)


# 2. Per-target R2 and absolute-error summary with bootstrap intervals.
x = np.arange(len(TARGETS))
fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
ax = axes[0]
r2_low = metrics["r2"] - metrics["r2_ci95_low"]
r2_high = metrics["r2_ci95_high"] - metrics["r2"]
ax.errorbar(
    x,
    metrics["r2"],
    yerr=np.vstack([r2_low, r2_high]),
    fmt="o",
    color="#202020",
    ecolor="#555555",
    elinewidth=1.5,
    capsize=3,
)
ax.plot(x[:5], metrics["r2"].iloc[:5], color="#1F77B4", linewidth=1.4)
ax.plot(x[5:], metrics["r2"].iloc[5:], color="#D95F02", linewidth=1.4)
ax.axhline(0.9, linestyle="--", color="#777777", linewidth=1)
ax.set_ylim(min(0.84, metrics["r2_ci95_low"].min() - 0.01), 0.98)
ax.set_ylabel(r"$R^2$")
ax.set_title(r"Predictive fit ($R^2$, bootstrap 95% CI)")
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
width = 0.38
ax.bar(x - width / 2, metrics["rmse"], width, label="RMSE", color="#4C78A8")
ax.bar(x + width / 2, metrics["mae"], width, label="MAE", color="#F58518")
ax.set_ylabel("Error (mmol/g)")
ax.set_title("Absolute prediction error")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
fig.suptitle(
    "External 10K performance by gas and pressure",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    -0.02,
    (
        f"Macro mean: R²={summary['macro_all_targets']['r2']:.4f}, "
        f"RMSE={summary['macro_all_targets']['rmse']:.4f}, "
        f"MAE={summary['macro_all_targets']['mae']:.4f}"
    ),
    ha="center",
)
fig.savefig(OUTPUTS["metrics"], bbox_inches="tight")
plt.close(fig)


# 3. Standardized residual distributions; whiskers show 5th-95th percentiles.
standardized = []
for target, prediction in zip(TARGETS, PREDICTIONS):
    y_true = rows[target].to_numpy(dtype=float)
    residual = rows[prediction].to_numpy(dtype=float) - y_true
    standardized.append(residual / max(y_true.std(ddof=0), 1.0e-12))

fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
box = ax.boxplot(
    standardized,
    whis=(5, 95),
    showfliers=False,
    patch_artist=True,
    medianprops={"color": "#111111", "linewidth": 1.3},
)
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.72)
ax.axhline(0, linestyle="--", color="#333333", linewidth=1)
ax.set_xticks(np.arange(1, len(LABELS) + 1))
ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_ylabel("Standardized residual: (prediction - reference) / SD(reference)")
ax.set_title(
    "External 10K standardized residual distributions\n"
    "Boxes: IQR; whiskers: 5th-95th percentiles; extreme points hidden",
    fontweight="bold",
)
ax.grid(axis="y", alpha=0.25)
fig.savefig(OUTPUTS["residuals"], bbox_inches="tight")
plt.close(fig)


# 4. Direct numerical equivalence of ALCHEMI and the prior CPU-compatible graph run.
comparison_rows = []
for index, target in enumerate(TARGETS):
    cpu_r2, cpu_rmse, cpu_mae = CPU_REFERENCE[target]
    alchemi = metrics.iloc[index]
    comparison_rows.append(
        {
            "target": target,
            "label": LABELS[index].replace("$", ""),
            "cpu_r2": cpu_r2,
            "alchemi_r2": float(alchemi.r2),
            "delta_r2_alchemi_minus_cpu": float(alchemi.r2 - cpu_r2),
            "cpu_rmse": cpu_rmse,
            "alchemi_rmse": float(alchemi.rmse),
            "delta_rmse_alchemi_minus_cpu": float(alchemi.rmse - cpu_rmse),
            "cpu_mae": cpu_mae,
            "alchemi_mae": float(alchemi.mae),
            "delta_mae_alchemi_minus_cpu": float(alchemi.mae - cpu_mae),
        }
    )
comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUTPUTS["comparison_csv"], index=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
delta_columns = [
    ("delta_r2_alchemi_minus_cpu", r"$\Delta R^2$"),
    ("delta_rmse_alchemi_minus_cpu", r"$\Delta$RMSE (mmol/g)"),
    ("delta_mae_alchemi_minus_cpu", r"$\Delta$MAE (mmol/g)"),
]
for ax, (column, ylabel) in zip(axes, delta_columns):
    values = comparison[column].to_numpy(dtype=float)
    ax.bar(x, values, color=colors, alpha=0.82)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(axis="y", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=55, ha="right")
    ax.set_title("ALCHEMI minus CPU-compatible result")
fig.suptitle(
    "Preprocessing-backend equivalence on the identical external 10K set",
    fontsize=15,
    fontweight="bold",
)
fig.savefig(OUTPUTS["equivalence"], bbox_inches="tight")
plt.close(fig)

print("Saved plots:")
for key, path in OUTPUTS.items():
    print(f"- {key}: {path}")
print("\nMaximum absolute ALCHEMI-vs-CPU metric differences:")
print(
    comparison[
        [
            "delta_r2_alchemi_minus_cpu",
            "delta_rmse_alchemi_minus_cpu",
            "delta_mae_alchemi_minus_cpu",
        ]
    ]
    .abs()
    .max()
    .to_string()
)
