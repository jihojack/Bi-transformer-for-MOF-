"""Reproduce the paired-head comparison using the internal 10,000-MOF test set.

Run: python plot_tower_comparison.py
Dependencies: pandas, numpy, matplotlib.
CSV inputs next to this script are exact copies of saved evaluation outputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'test_metrics_all_towers.csv'
BOOT = ROOT / 'paired_tower_mae_bootstrap.csv'
fonts = {f.name for f in font_manager.fontManager.ttflist}
korean = next((f for f in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic'] if f in fonts), 'DejaVu Sans')
plt.rcParams.update({
    'font.family': ['DejaVu Sans', korean],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#67717c', 'axes.linewidth': .8,
    'xtick.color': '#38434f', 'ytick.color': '#38434f',
    'text.color': '#1d2936', 'axes.labelcolor': '#1d2936',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.unicode_minus': False,
})
PALETTE = {'structure': '#d17c22', 'descriptor': '#24847e', 'fusion': '#214e82'}
MARKERS = {'structure': 's', 'descriptor': '^', 'fusion': 'o'}
LABELS = {
    'ko': {'fusion': '융합 예측', 'structure': '구조 타워 보조 예측', 'descriptor': '전역 기술자 타워 보조 예측'},
    'en': {'fusion': 'Fusion', 'structure': 'Structure auxiliary', 'descriptor': 'Descriptor auxiliary'},
}


def export(fig, name):
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / f'{name}.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)


def mae_plot(frame, lang):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.45))
    fig.subplots_adjust(left=.08, right=.98, top=.73, bottom=.23, wspace=.28)
    for panel, (ax, gas) in enumerate(zip(axes, ['CO2', 'CH4'])):
        p = frame[(frame.tower == 'fusion') & (frame.gas == gas)].sort_values('pressure_bar')
        x = np.arange(len(p))
        for tower in ['structure', 'descriptor', 'fusion']:
            rows = frame[(frame.tower == tower) & (frame.gas == gas)].sort_values('pressure_bar')
            ax.plot(x, rows.mae_mmol_g, label=LABELS[lang][tower],
                    color=PALETTE[tower], marker=MARKERS[tower],
                    linewidth=2.5 if tower == 'fusion' else 1.7,
                    markersize=6, markerfacecolor='white', markeredgewidth=1.6,
                    zorder=5 if tower == 'fusion' else 3)
        ax.set_xticks(x, [f'{v:.2g}' for v in p.pressure_bar])
        ax.set_xlabel('압력 (bar)' if lang == 'ko' else 'Pressure (bar)', labelpad=8)
        ax.set_ylabel('MAE (mmol g⁻¹)', labelpad=8)
        ax.set_title(('(a) CO₂' if panel == 0 else '(b) CH₄') + ' · 298 K', loc='left', pad=11, weight='bold')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis='y', color='#e3e7eb', linewidth=.75)
        ax.set_axisbelow(True)
        ax.margins(x=.09, y=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [2, 0, 1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='upper center', bbox_to_anchor=(.52, .88), frameon=False,
               ncol=3, columnspacing=1.7, handlelength=2.5, fontsize=10)
    fig.suptitle('기체·압력 조건별 융합 예측과 보조 예측의 오차 비교' if lang == 'ko'
                 else 'Fusion and auxiliary-head errors by gas and pressure', y=.98, fontsize=15, weight='bold')
    note = ('동일한 내부 시험 MOF 10,000개 · MAE가 낮을수록 우수\n'
            '각 타워의 보조 헤드와 융합 헤드는 함께 학습됨. 압력 조건은 동일 간격으로 배치함.'
            if lang == 'ko' else
            'Same internal test set: 10,000 MOFs · Lower MAE is better\n'
            'Auxiliary heads and the fusion head were jointly trained. Pressure conditions are equally spaced.')
    fig.text(.52, .075, note, ha='center', va='center', fontsize=9, color='#566370', linespacing=1.55)
    export(fig, f'tower_mae_comparison_{lang}')


def improvement_plot(bootstrap):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8))
    fig.subplots_adjust(left=.13, right=.98, top=.76, bottom=.30, wspace=.37)
    for ax, gas in zip(axes, ['CO2', 'CH4']):
        p = bootstrap[bootstrap.gas == gas]
        pressures = sorted(p.pressure_bar.unique())
        y = np.arange(len(pressures))[::-1]
        for comparison, tower, offset in [('fusion_vs_structure','structure',.12), ('fusion_vs_descriptor','descriptor',-.12)]:
            rows = p[p.comparison == comparison].sort_values('pressure_bar')
            diff = rows.mae_reduction_mmol_g.to_numpy()
            lo, hi = rows.ci95_lower_mmol_g.to_numpy(), rows.ci95_upper_mmol_g.to_numpy()
            ax.errorbar(diff, y+offset, xerr=np.array([diff-lo, hi-diff]),
                        marker=MARKERS[tower], markersize=5, linestyle='none',
                        capsize=3, color=PALETTE[tower], elinewidth=1.6,
                        label='구조 보조 예측 대비' if tower == 'structure' else '기술자 보조 예측 대비')
        ax.axvline(0, color='#596570', linestyle='--', linewidth=1)
        ax.set_yticks(y, [f'{v:.2g} bar' for v in pressures])
        ax.set_ylim(-.55, 4.55)
        ax.set_title('CO₂' if gas == 'CO2' else 'CH₄', loc='left', weight='bold', pad=12)
        ax.set_xlabel('MAE 감소량 (mmol g⁻¹)', labelpad=9)
        ax.grid(axis='x', color='#e3e7eb', linewidth=.75)
        ax.set_axisbelow(True)
        right = p.ci95_upper_mmol_g.max() * 1.10
        ax.set_xlim(-right*.05, right)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.suptitle('융합 예측의 MAE 감소량과 95% 신뢰구간', y=.98, fontsize=15, weight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.54,.90), frameon=False, ncol=2, fontsize=10)
    fig.text(.54,.075,
             '감소량 = 보조 예측 MAE − 융합 예측 MAE. 0보다 클수록 융합 예측이 우수함.\n'
             '동일 시험 MOF 10,000개를 기준으로 MOF 단위 부트스트랩 1,000회. 학습 시드 변동은 포함하지 않음.',
             fontsize=9, color='#566370', ha='center', linespacing=1.55)
    export(fig, 'tower_mae_improvement_ci_ko')


def main():
    raw = pd.read_csv(DATA)
    df = raw[raw.scope == 'condition'].copy()
    boot = pd.read_csv(BOOT)
    assert len(df) == 30 and (df.n_test == 10000).all()
    assert set(df.tower) == {'fusion','structure','descriptor'}
    assert len(boot) == 20
    summary = {}
    for gas in ['CO2','CH4']:
        for comparison, tower in [('fusion_vs_structure','structure'),('fusion_vs_descriptor','descriptor')]:
            sub = boot[(boot.gas == gas) & (boot.comparison == comparison)]
            summary[f'{gas}_{comparison}'] = {
                'min_mae_reduction_pct': float(sub.mae_reduction_pct.min()),
                'max_mae_reduction_pct': float(sub.mae_reduction_pct.max()),
                'all_ci95_lower_above_zero': bool((sub.ci95_lower_mmol_g > 0).all()),
            }
            for row in sub.itertuples():
                base = df[(df.gas == gas) & (df.tower == tower) & np.isclose(df.pressure_bar,row.pressure_bar)].iloc[0]
                fused = df[(df.gas == gas) & (df.tower == 'fusion') & np.isclose(df.pressure_bar,row.pressure_bar)].iloc[0]
                assert abs(base.mae_mmol_g-fused.mae_mmol_g-row.mae_reduction_mmol_g) < 1e-6
    mae_plot(df, 'ko')
    mae_plot(df, 'en')
    improvement_plot(boot)
    (ROOT/'comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Saved PNG, SVG, PDF and summary to', ROOT)


if __name__ == '__main__':
    main()
