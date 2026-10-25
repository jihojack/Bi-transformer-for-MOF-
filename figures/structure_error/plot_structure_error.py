"""Structure-stratified errors on the frozen model's internal 10,000-MOF test set.

Run: python plot_structure_error.py
Dependencies: numpy, pandas, matplotlib.
No training or inference is performed. Original saved predictions are used.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
RESULTS = WORKSPACE / 'results' / 'internal'
PREDICTIONS = RESULTS / 'test_predictions.csv'
FEATURES = ROOT / 'structure_error_per_mof.csv'
REFERENCE = RESULTS / 'extended' / 'subgroup_robustness.csv'
CHECKPOINT = WORKSPACE / 'app' / 'assets' / 'best_dual_gas_alchemi_50k.pt'
SEED = 20261025
BOOTSTRAPS = 1000
FEATURE_NAMES = ['surface_area_m2g', 'void_fraction', 'lcd', 'pld']
STYLE = {
    'surface_area_m2g': ('표면적 (m²/g)', 'Surface area (m²/g)'),
    'void_fraction': ('공극률', 'Void fraction'),
    'lcd': ('최대 공동 직경, LCD (Å)', 'Largest cavity diameter, LCD (Å)'),
    'pld': ('제한 공극 직경, PLD (Å)', 'Pore limiting diameter, PLD (Å)'),
}
fonts = {f.name for f in font_manager.fontManager.ttflist}
korean = next((f for f in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic'] if f in fonts), 'DejaVu Sans')
plt.rcParams.update({
    'font.family': ['DejaVu Sans', korean], 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#65717b', 'axes.linewidth': .8,
    'text.color': '#23313f', 'axes.labelcolor': '#23313f',
    'xtick.color': '#43515e', 'ytick.color': '#43515e',
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def bootstrap_means(values, rng):
    """Resample MOFs jointly for both gases; five pressure conditions stay together."""
    out = np.empty((BOOTSTRAPS, values.shape[1]))
    for start in range(0, BOOTSTRAPS, 100):
        size = min(100, BOOTSTRAPS-start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        out[start:start+size] = values[indices].mean(axis=1)
    return np.quantile(out, [.025, .975], axis=0)


def build_statistics():
    original_metadata = json.loads((ROOT/'analysis_metadata.json').read_text(encoding='utf-8'))
    scale = np.asarray(original_metadata['scale_per_condition_mmol_g'], dtype=np.float64)
    predictions = pd.read_csv(PREDICTIONS)
    test = pd.read_csv(FEATURES)
    indices = predictions.graph_index.to_numpy(dtype=int)
    split = json.loads((WORKSPACE/'data'/'splits'/'fixed_split_70_10_20.json').read_text(encoding='utf-8'))
    assert len(predictions) == 10000
    assert predictions.mof_id.is_unique and test.mof_id.is_unique
    assert np.array_equal(test.mof_id.to_numpy(), predictions.mof_id.to_numpy())
    assert set(indices) == set(split['test_indices'])
    assert not set(indices) & set(split['train_indices'])
    obs_cols = [c for c in predictions if c.startswith('observed_')]
    pred_cols = [c.replace('observed_', 'fusion_', 1) for c in obs_cols]
    assert len(obs_cols) == 10 and all(c in predictions for c in pred_cols)
    assert all('co2_' in c for c in obs_cols[:5]) and all('ch4_' in c for c in obs_cols[5:])
    observed = predictions[obs_cols].to_numpy(dtype=np.float64)
    predicted = predictions[pred_cols].to_numpy(dtype=np.float64)
    absolute = np.abs(predicted-observed)
    normalized = absolute / scale
    # One paired score per MOF and gas, averaged across the five pressures.
    scores = np.column_stack([normalized[:, :5].mean(1), normalized[:, 5:].mean(1)])
    raw_scores = np.column_stack([absolute[:, :5].mean(1), absolute[:, 5:].mean(1)])
    mean_observed = np.column_stack([observed[:, :5].mean(1), observed[:, 5:].mean(1)])
    per_mof = test[['mof_id']+FEATURE_NAMES].copy()
    per_mof['co2_normalized_absolute_error'] = scores[:, 0]
    per_mof['ch4_normalized_absolute_error'] = scores[:, 1]
    per_mof['co2_mae_mmol_g'] = raw_scores[:, 0]
    per_mof['ch4_mae_mmol_g'] = raw_scores[:, 1]
    rows = []
    rng = np.random.default_rng(SEED)
    for feature in FEATURE_NAMES:
        values = test[feature]
        assert np.isfinite(values).all()
        # Reproduce saved evaluation: stable ranks split ties in test-row order.
        bins = pd.qcut(values.rank(method='first'), 5, labels=False).to_numpy()
        per_mof[f'{feature}_quintile'] = bins+1
        for q in range(5):
            mask = bins == q
            ci = bootstrap_means(scores[mask], rng)
            for gas_index, gas in enumerate(['CO2','CH4']):
                rows.append({
                    'feature':feature, 'quintile':q+1, 'gas':gas, 'n_mofs':int(mask.sum()),
                    'feature_min':float(values[mask].min()), 'feature_max':float(values[mask].max()),
                    'feature_mean':float(values[mask].mean()),
                    'mean_normalized_absolute_error':float(scores[mask,gas_index].mean()),
                    'ci95_lower':float(ci[0,gas_index]), 'ci95_upper':float(ci[1,gas_index]),
                    'mae_mmol_g':float(raw_scores[mask,gas_index].mean()),
                    'mean_observed_uptake_mmol_g':float(mean_observed[mask,gas_index].mean()),
                })
    summary = pd.DataFrame(rows)
    reference = pd.read_csv(REFERENCE)
    max_difference = 0.
    for row in summary.itertuples():
        old = reference[(reference.subgroup_type == 'geometry_quintile') &
                        (reference.subgroup_feature == row.feature) &
                        (reference.subgroup == f'Q{row.quintile}')].iloc[0]
        old_score = old[f'{row.gas.lower()}_mean_normalized_absolute_error']
        max_difference = max(max_difference, abs(row.mean_normalized_absolute_error-old_score))
    assert max_difference < 1e-6, max_difference
    summary.to_csv(ROOT/'structure_error_by_quintile.csv',index=False,encoding='utf-8-sig')
    per_mof.to_csv(ROOT/'structure_error_per_mof.csv',index=False,encoding='utf-8-sig')
    metadata = {
        'evaluation_set':'internal held-out test set', 'n_mofs':10000,
        'mofs_per_quintile':2000, 'bootstrap_resamples':BOOTSTRAPS, 'bootstrap_seed':SEED,
        'predictions_source':PREDICTIONS.relative_to(WORKSPACE).as_posix(),
        'features_source':FEATURES.relative_to(WORKSPACE).as_posix(),
        'checkpoint_source':CHECKPOINT.relative_to(WORKSPACE).as_posix(),
        'scale_per_condition_mmol_g':scale.tolist(),
        'definition':'For each condition: abs(predicted-observed)/max(train_target_std,0.05); mean over 5 pressures per gas, then over MOFs in the group.',
        'ci_definition':'Pointwise 95% percentile bootstrap CI of the group mean, resampling MOFs within fixed groups; training-seed and bin-boundary uncertainty excluded.',
        'tie_handling':'Equal-count quintiles from feature ranks(method=first); equal values can be split across adjacent bins.',
        'max_abs_difference_from_saved_subgroup_results':max_difference,
        'caution':'Descriptive association, not causal or independent feature effects; normalization does not control for subgroup uptake distributions.',
    }
    (ROOT/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Verified against original subgroup results. Max absolute difference:',max_difference)
    return summary


def plot(summary, lang='ko', raw=False):
    fig, axes = plt.subplots(2,2,figsize=(10.8,7.7),sharey=not raw)
    fig.subplots_adjust(left=.10,right=.98,top=.81,bottom=.16,wspace=.16,hspace=.48)
    ymax = max(summary.ci95_upper.max()*1.12,.28)
    for panel,(ax,feature) in enumerate(zip(axes.flat,FEATURE_NAMES)):
        for gas,color,marker in [('CO2','#214e82','o'),('CH4','#bd681b','s')]:
            data = summary[(summary.feature == feature)&(summary.gas == gas)].sort_values('quintile')
            x = data.quintile.to_numpy()
            if raw:
                ax.plot(x,data.mae_mmol_g,color=color,marker=marker,linewidth=1.9,
                        markersize=5.5,markerfacecolor='white',markeredgewidth=1.5,label='CO₂' if gas=='CO2' else 'CH₄')
            else:
                mu = data.mean_normalized_absolute_error.to_numpy()
                lo,hi = data.ci95_lower.to_numpy(),data.ci95_upper.to_numpy()
                ax.errorbar(x,mu,yerr=np.array([mu-lo,hi-mu]),color=color,
                            marker=marker,linewidth=1.9,markersize=5.5,markerfacecolor='white',
                            markeredgewidth=1.5,capsize=3,elinewidth=1.2,
                            label='CO₂' if gas=='CO2' else 'CH₄')
        means = summary[(summary.feature == feature)&(summary.gas=='CO2')].sort_values('quintile').feature_mean
        digits = 0 if feature=='surface_area_m2g' else (3 if feature=='void_fraction' else 2)
        ax.set_xticks(np.arange(1,6),[f'Q{i}\n{v:.{digits}f}' for i,v in enumerate(means,1)])
        title = STYLE[feature][0 if lang=='ko' else 1]
        ax.set_title(f'({chr(97+panel)}) {title}',loc='left',weight='bold',pad=11,fontsize=12)
        ax.set_xlim(.72,5.28)
        ax.set_ylim(0,None if raw else ymax)
        if not raw:
            ax.yaxis.set_major_locator(MultipleLocator(.05))
        if panel%2==0 or raw:
            ax.set_ylabel(('평균 절대오차 (mmol g⁻¹)' if raw else '정규화 평균 절대오차') if lang=='ko'
                          else ('MAE (mmol g⁻¹)' if raw else 'Mean normalized absolute error'),labelpad=9,fontsize=10.5)
        ax.tick_params(axis='x',labelsize=10)
        ax.grid(axis='y',linewidth=.75,color='#e1e6eb')
        ax.set_axisbelow(True)
    title = ('구조 특성별 예측 오차' if lang=='ko' else 'Prediction error by structural properties')
    if raw:
        title += ' — MAE'
    fig.suptitle(title,y=.975,fontsize=17,weight='bold')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.925),
               frameon=False,ncol=2,handlelength=2.6,columnspacing=3,fontsize=12)
    if lang=='ko':
        note = '내부 시험 MOF 10,000개 · 각 구간 2,000개 · 기체별 5개 압력 조건의 오차 평균\n'
        note += 'Q1 → Q5: 특성값이 낮은 구간 → 높은 구간. 구간명 아래 숫자는 해당 특성의 구간 평균.'
        note += '\n오차막대: MOF 단위 부트스트랩 1,000회의 구간별 평균 95% 신뢰구간.' if not raw else ''
    else:
        note = 'Internal test set: 10,000 MOFs; 2,000 per bin. Errors averaged across 5 pressures per gas.\n'
        note += 'Q1 → Q5: low → high feature values. Numbers below bin labels denote mean feature values.'
        note += '\nError bars: pointwise 95% CIs from 1,000 within-bin MOF bootstrap resamples.' if not raw else ''
    fig.text(.54,.062,note,ha='center',va='center',fontsize=9,color='#566370',linespacing=1.5)
    stem = f'structure_error_{"mae" if raw else "normalized"}_{lang}'
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute bins and bootstrap CIs from bundled per-MOF records and saved predictions.')
    args = parser.parse_args()
    stats = build_statistics() if args.recompute else pd.read_csv(ROOT/'structure_error_by_quintile.csv')
    plot(stats,'ko')
    plot(stats,'en')
    plot(stats,'ko',raw=True)
    print('Saved graphs and data:',ROOT)
