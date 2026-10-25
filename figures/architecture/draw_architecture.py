"""Draw the verified MOF model architecture with Matplotlib only.

Run: python draw_architecture.py
Outputs: English/Korean overview and detailed module diagram (PNG/SVG/PDF).
Architecture source: 15.DualGas_ALCHEMI_50K_L4_Training.py;
dimensions cross-checked against best_dual_gas_alchemi_50k.pt.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
INK = '#24303a'
MUTED = '#586776'
COLORS = dict(input='#fae4e5', embed='#e4def0', attention='#fce5c3',
              norm='#f0f1ca', ff='#d5ebf4', output='#d5ead8',
              pool='#d8ece7', group='#f7f8fa', white='#ffffff')
KOREAN = next((f for f in ['Malgun Gothic', 'Noto Sans CJK KR', 'NanumGothic']
               if f in {x.name for x in font_manager.fontManager.ttflist}), 'DejaVu Sans')
plt.rcParams.update({'font.family': ['DejaVu Sans', KOREAN], 'font.size': 10, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})


def label(ax, x, y, text, size=11, weight='normal', color=INK, ha='center'):
    return ax.text(x, y, text, ha=ha, va='center', fontsize=size,
                   fontweight=weight, color=color, linespacing=1.4, zorder=5)


def box(ax, x, y, w, h, text='', color='white', size=11, lw=1.7, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.025,rounding_size=0.10',
                       facecolor=COLORS.get(color, color), edgecolor=INK,
                       linewidth=lw, linestyle=(0, (4, 3)) if dashed else '-', zorder=2)
    ax.add_patch(p)
    if text:
        label(ax, x+w/2, y+h/2, text, size)
    return p


def arrow(ax, points, dashed=False, color=INK, lw=1.7):
    for i in range(len(points)-1):
        p = FancyArrowPatch(points[i], points[i+1],
                            arrowstyle='-|>' if i == len(points)-2 else '-',
                            mutation_scale=14, linewidth=lw, color=color,
                            linestyle=(0, (4, 3)) if dashed else '-',
                            shrinkA=0, shrinkB=0, zorder=3)
        ax.add_patch(p)


def save(fig, stem):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview(ko=False):
    fig, ax = plt.subplots(figsize=(16, 16.5))
    ax.set(xlim=(0, 16), ylim=(0, 17)); ax.axis('off')
    t = lambda en, kr: kr if ko else en
    label(ax, 8, 16.65, t('Dual-tower MOF adsorption predictor', '이중 타워 기반 MOF 흡착량 예측 모델'), 21, 'bold')
    label(ax, 8, 16.23, t('Actual checkpoint architecture  |  14,536,691 trainable parameters  |  T = 298 K',
                          '실제 체크포인트 구조  |  학습 파라미터 14,536,691개  |  T = 298 K'), 10, color=MUTED)

    box(ax, 4.1, 14.98, 7.8, .83,
        'CO₂: 0.01 / 0.05 / 0.1 / 0.5 / 2.5 bar\nCH₄: 0.05 / 0.5 / 0.9 / 2.5 / 4.5 bar', 'output', 11)
    label(ax, 12.25, 15.38, t('10 uptake values\n(mmol g⁻¹)', '흡착량 10개\n(mmol g⁻¹)'), 10, ha='left')
    box(ax, 4.5, 13.65, 7.0, .95,
        t('Positive, monotonic transform — separately for each gas', '기체별 양수·단조 증가 변환') +
        '\nq = expm1(min(cumsum(softplus(r)), 6))', 'norm', 11)
    arrow(ax, [(8,14.60), (8,14.98)])
    box(ax, 5.2, 12.57, 5.6, .70,
        t('Shared MLP head: 256 → 256 → 1', '공유 MLP 출력 헤드: 256 → 256 → 1') + '\nSiLU', 'embed', 11)
    arrow(ax, [(8,13.27), (8,13.65)])
    label(ax, 8.18, 12.31, t('First 10 query-token outputs', '앞의 질의 토큰 10개 선택'), 9, color=MUTED,ha='left')

    box(ax, 4.5, 10.77, 7.0, 1.18,
        t('Fusion Transformer encoder × 4', '융합 Transformer 인코더 × 4') +
        '\n8 heads · d = 256 · FFN = 1024\nPre-LN blocks + final LayerNorm', 'attention', 12)
    arrow(ax, [(8,11.95), (8,12.57)])
    box(ax, 4.5, 9.64, 7.0, .72,
        t('Concatenate [10 queries | 10 structure | 10 descriptor]',
          '토큰 결합 [질의 10개 | 구조 10개 | 기술자 10개]') +
        '\n30 × 256  +  token-type embeddings', 'pool', 10)
    arrow(ax, [(8,10.36), (8,10.77)])

    box(ax, .15, 10.72, 3.45, 1.12,
        t('10 learned fusion queries', '학습형 융합 질의 10개') +
        '\n+ condition tokens\n10 × 256', 'embed', 10)
    arrow(ax, [(3.6,11.28),(4.0,11.28),(4.0,10.0),(4.5,10.0)])

    # Separate auxiliary outputs: branches are supervised but not fused back.
    box(ax, .15, 8.85, 2.30, .95,
        t('Structure auxiliary\nMLP → monotonic q', '구조 보조 예측\nMLP → 단조 흡착량'), 'white', 9, dashed=True)
    box(ax, 13.55, 8.85, 2.30, .95,
        t('Descriptor auxiliary\nMLP → monotonic q', '기술자 보조 예측\nMLP → 단조 흡착량'), 'white', 9, dashed=True)

    for cx in [5.1,10.9]:
        box(ax, cx-2.0, 8.10, 4.0, .88,
            t('Condition adapter × 2', '기체·압력 조건 적응 × 2') +
            '\nTransformer + LN → 10 × 256', 'attention', 10.5)
    arrow(ax, [(5.1,8.98),(5.1,9.64)])
    arrow(ax, [(10.9,8.98),(10.9,9.64)])
    arrow(ax, [(3.1,8.55),(2.75,8.55),(2.75,9.33),(2.45,9.33)], True)
    arrow(ax, [(12.9,8.55),(13.25,8.55),(13.25,9.33),(13.55,9.33)], True)

    # Main tower groups.
    box(ax, 2.75, 2.48, 4.7, 4.93, color='group', lw=1.4)
    box(ax, 8.55, 2.48, 4.7, 4.93, color='group', lw=1.4)
    label(ax, 5.1, 7.08, t('Structure tower', '구조 타워'), 14, 'bold')
    label(ax, 10.9, 7.08, t('Descriptor tower', '전역 기술자 타워'), 14, 'bold')

    box(ax, 3.1, 5.91, 4.0, .70,
        t('Invariant attention pooling', '회전 불변 어텐션 풀링') + '\n8 learned queries → 8 × 256', 'pool', 10)
    box(ax, 3.1, 4.25, 4.0, 1.22,
        t('Equivariant graph attention × 6', '등변성 그래프 어텐션 × 6') +
        '\n8 heads · scalar: 256\nvector: 48 × 3', 'attention', 11)
    box(ax, 3.1, 2.91, 4.0, .90,
        t('Atom embedding + node MLP', '원자 임베딩 + 노드 MLP') +
        '\nZ embedding + [13 features; 13 masks]\nscalar h ∈ ℝ²⁵⁶; vector v = 0', 'embed', 10)
    arrow(ax,[(5.1,3.81),(5.1,4.25)])
    arrow(ax,[(5.1,5.47),(5.1,5.91)])
    arrow(ax,[(5.1,6.61),(5.1,6.82),(7.7,6.82),(7.7,7.72),(5.1,7.72),(5.1,8.10)])

    box(ax, 8.90, 5.91, 4.0, .70,
        t('Final LayerNorm', '최종 LayerNorm') + '\n31 × 256 tokens', 'norm', 11)
    box(ax, 8.90, 4.55, 4.0, .92,
        t('Transformer encoder × 4', 'Transformer 인코더 × 4') +
        '\n8 heads · FFN = 1024 · Pre-LN', 'attention', 11)
    box(ax, 8.90, 2.91, 4.0, 1.20,
        t('Feature-wise tokenization', '특성별 토큰화') +
        '\ntⱼ = xⱼwⱼ + bⱼ + eⱼ\n30 feature tokens + 1 [CLS]', 'embed', 11)
    arrow(ax,[(10.9,4.11),(10.9,4.55)])
    arrow(ax,[(10.9,5.47),(10.9,5.91)])
    arrow(ax,[(10.9,6.61),(10.9,6.82),(13.5,6.82),(13.5,7.72),(10.9,7.72),(10.9,8.10)])

    # Shared conditioning feeds both adapters from a horizontal bus.
    box(ax, .15, 5.50, 2.1, 1.50,
        t('Gas + pressure', '기체 + 압력') +
        '\n5 pressure features\n6 gas properties\nGas ID + condition ID\n10 × 256', 'input', 9)
    arrow(ax,[(1.2,7.0),(1.2,7.55),(4.05,7.55),(4.05,8.1)])
    arrow(ax,[(4.05,7.55),(9.85,7.55),(9.85,8.1)])
    arrow(ax,[(1.2,7.55),(.07,7.55),(.07,12.13),(1.85,12.13),(1.85,11.84)], lw=1.2)

    box(ax, .15, 2.65, 2.10, 2.45,
        t('Edge context', '간선 문맥') +
        '\n64 RBF\n+ 24 relation dims\n+ soft bond score\n↓\nMLP → 256\n\nUnit directions', 'ff', 9)
    arrow(ax,[(2.25,4.86),(3.1,4.86)],lw=1.2)

    box(ax, 3.1, .70, 4.0, 1.04,
        t('CIF → periodic atomic graph', 'CIF → 주기적 원자 그래프') +
        '\n6 Å cutoff · max. 32 neighbors/atom\nZ, node features, edge vectors', 'input', 10)
    box(ax, 8.9, .70, 4.0, 1.04,
        t('30 geometry + chemistry descriptors', '기하·물리화학 기술자 30개') +
        t('\nMedian imputation + RobustScaler', '\n중앙값 대체 + RobustScaler'), 'input', 10)
    arrow(ax,[(5.1,1.74),(5.1,2.91)])
    arrow(ax,[(10.9,1.74),(10.9,2.91)])
    arrow(ax,[(5.1,2.02),(1.2,2.02),(1.2,2.65)],lw=1.2)
    label(ax, 8, .22,
        t('Dashed branches: auxiliary supervision. Modality dropout (p = 0.05) is applied before fusion during training.',
          '점선: 보조 예측 분기. 학습 중 융합 직전에 modality dropout(p = 0.05)을 적용합니다.'),9,color=MUTED)
    save(fig, 'mof_architecture_ko' if ko else 'mof_architecture_en')


def details():
    fig, ax = plt.subplots(figsize=(18, 11.2))
    ax.set(xlim=(0,18),ylim=(0,11.2)); ax.axis('off')
    label(ax,9,10.82,'Module details of the implemented model',20,'bold')
    label(ax,9,10.40,'Residual paths, normalization order and output constraint',11,color=MUTED)
    for x,w,title in [(0.3,5.3,'A  Equivariant graph block × 6'),(6.2,5.3,'B  Pre-LN Transformer block'),(12.1,5.6,'C  Gas-wise output transform')]:
        box(ax,x,.58,w,9.35,color='group',lw=1.4)
        label(ax,x+w/2,9.52,title,13,'bold')

    # A: faithful summary of the implemented scalar/vector update.
    box(ax,.80,1.02,4.30,.68,'Scalar h: N × 256; vector v: N × 48 × 3','input',11)
    box(ax,1.20,2.02,3.50,.62,'LayerNorm(h) → Q, K, V','norm',11)
    arrow(ax,[(2.95,1.70),(2.95,2.02)])
    box(ax,1.20,3.02,3.50,.96,'Edge-conditioned attention\nsoftmax over incoming neighbors\nScalar + directional vector messages','attention',10.5)
    arrow(ax,[(2.95,2.64),(2.95,3.02)])
    box(ax,1.20,4.42,3.50,.83,'Residual message updates\nh′ = h + Dropout(Wₛ Σmₛ)\nv′ = v + Wᵥ Σmᵥ','ff',10.5)
    arrow(ax,[(2.95,3.98),(2.95,4.42)])
    arrow(ax,[(.95,1.70),(.64,1.70),(.64,4.83),(1.2,4.83)],lw=1.4)
    box(ax,1.20,5.72,3.50,.89,'FFN([h′; ‖v′‖])\n304 → 512 → 256\nSiLU + Dropout','ff',11)
    arrow(ax,[(2.95,5.25),(2.95,5.72)])
    box(ax,1.20,7.03,3.50,.68,'hₒᵤₜ = LayerNorm(h′ + FFN)','norm',11)
    arrow(ax,[(2.95,6.61),(2.95,7.03)])
    arrow(ax,[(4.7,4.83),(5.10,4.83),(5.10,7.37),(4.7,7.37)],lw=1.4)
    box(ax,1.20,8.13,3.50,.68,'vₒᵤₜ = v′ ⊙ sigmoid(Wg hₒᵤₜ)','pool',10.5)
    arrow(ax,[(2.95,7.71),(2.95,8.13)])

    # B: actual torch norm_first=True ordering, not the post-LN reference image.
    box(ax,7.15,1.05,3.4,.58,'Input tokens X','input',12)
    box(ax,7.15,2.10,3.4,.58,'LayerNorm','norm',12)
    box(ax,7.15,3.05,3.4,.77,'Multi-head self-attention\n8 heads · d = 256','attention',12)
    box(ax,7.15,4.27,3.4,.58,'Residual add (+ dropout)','norm',11)
    box(ax,7.15,5.32,3.4,.58,'LayerNorm','norm',12)
    box(ax,7.15,6.30,3.4,.82,'Feed-forward network\n256 → 1024 → 256 · GELU','ff',11)
    box(ax,7.15,7.59,3.4,.58,'Residual add (+ dropout)','norm',11)
    for y1,y2 in [(1.63,2.10),(2.68,3.05),(3.82,4.27),(4.85,5.32),(5.90,6.30),(7.12,7.59)]:
        arrow(ax,[(8.85,y1),(8.85,y2)])
    arrow(ax,[(8.85,1.84),(6.67,1.84),(6.67,4.56),(7.15,4.56)])
    arrow(ax,[(8.85,5.05),(11.02,5.05),(11.02,7.88),(10.55,7.88)])
    label(ax,8.85,8.84,'Descriptor × 4  |  Adapter × 2\nFusion × 4; final LayerNorm after each stack',9.7,color=MUTED)

    # C.
    box(ax,12.90,1.05,4.0,.75,'10 raw MLP outputs\nSplit into CO₂ (5) and CH₄ (5)','embed',11)
    box(ax,12.90,2.35,4.0,.73,'Softplus\nPositive log-space increments','attention',12)
    box(ax,12.90,3.65,4.0,.77,'Cumulative sum\nIncreasing log(1 + q)','ff',12)
    box(ax,12.90,5.02,4.0,.70,'Upper clamp at 6','norm',12)
    box(ax,12.90,6.32,4.0,.70,'expm1: q = exp(log(1 + q)) − 1','embed',11)
    box(ax,12.90,7.63,4.0,.85,'5 nonnegative uptake values / gas\nq(p₁) ≤ q(p₂) ≤ ··· ≤ q(p₅)','output',11)
    for y1,y2 in [(1.80,2.35),(3.08,3.65),(4.42,5.02),(5.72,6.32),(7.02,7.63)]:
        arrow(ax,[(14.9,y1),(14.9,y2)])
    label(ax,14.9,8.93,'No Softmax on the regression output',10,color=MUTED)
    label(ax,9,.17,'Attention uses Softmax internally. The final adsorption output uses a cumulative Softplus transform, not a probability normalization.',10,color=MUTED)
    save(fig,'mof_architecture_module_details')


if __name__ == '__main__':
    overview(False)
    overview(True)
    details()
    print('Saved 9 figure files to', OUT)
