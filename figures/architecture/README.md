# MOF 모델 아키텍처 그림

## 파일

- `mof_architecture_ko`: 한글 전체 구조도
- `mof_architecture_en`: 영문 전체 구조도
- `mof_architecture_module_details`: 내부 연산 상세도(영문)
- 각 그림은 300 dpi PNG, SVG, PDF로 저장됩니다.
- `draw_architecture.py`: Matplotlib으로 그림을 재생성하는 코드

실행: `python draw_architecture.py`

## 확인한 구현

원본 학습 코드: `pipeline/15.DualGas_ALCHEMI_50K_L4_Training.py`

- Config: 52행 이후
- EnrichedEquivariantAttentionBlock: 667행 이후
- InvariantTokenPool: 781행 이후
- AlchemiStructureTower: 805행 이후
- DescriptorTower: 881행 이후
- PressureAdapter: 914행 이후
- positive_monotonic_curve: 937행 이후
- DualGasAlchemiModel: 953행 이후

체크포인트: `app/assets/best_dual_gas_alchemi_50k.pt`

체크포인트에서 확인한 크기: descriptor dimension 30, feature token weight
30 × 256, node projector input 26 = 연속 노드 특성 13 + 결측 마스크 13,
fusion queries 10 × 256. 학습 파라미터 수는 14,536,691개입니다.

## 그림 읽는 법

화살표는 아래에서 위로 읽습니다. 구조 타워는 주기적 원자 그래프를 처리하고,
전역 기술자 타워는 기하 및 물리화학 기술자 30개를 처리합니다. 기술자 타워를
화학적 특성만 처리하는 타워로 해석하면 실제 구현과 다릅니다.

두 조건 적응 모듈은 각각 [조건 토큰 10개; 해당 타워 토큰]을 결합해
self-attention을 수행하고 앞의 조건 토큰 10개를 반환합니다.
융합 모듈은 [학습형 질의 + 조건 토큰; 구조 조건 토큰; 기술자 조건 토큰]을
결합하며 각 유형별 임베딩을 더합니다. 총 토큰 수는 30개입니다.

점선 보조 예측 분기는 두 타워를 융합과 함께 학습시키기 위한 출력입니다.
두 독립 모델을 따로 학습한 뒤 앙상블한 구조가 아닙니다.
Modality dropout은 학습 시에만 융합 직전의 타워 조건 토큰에 적용됩니다.
그림은 계산 흐름의 모듈 수준 요약이며 모든 선형층과 dropout을 펼친 것은 아닙니다.

Pre-LN Transformer 블록은 LayerNorm → self-attention → residual add →
LayerNorm → FFN → residual add 순서이며 각 스택 끝에 별도 LayerNorm이 있습니다.

출력은 CO2 5개 및 CH4 5개 조건별 raw 값을 기체별로 나눈 후
Softplus → 누적합 → 상한 6 → expm1 변환을 적용합니다.
이에 따라 기체 내 압력 증가 순서에서 흡착량은 감소하지 않습니다.
상한 때문에 값이 같아질 수 있으므로 엄밀한 제약은 단조 비감소입니다.
Softmax는 어텐션 내부에서만 사용되며 최종 회귀값의 변환은 아닙니다.

## 논문용 캡션 예시

이중 타워 기반 MOF 흡착량 예측 모델의 구조. 구조 타워는 화학적 특성이
보강된 주기적 원자 그래프를 6개의 등변성 그래프 어텐션 블록으로 처리하고,
전역 기술자 타워는 30개 기하·물리화학 기술자를 4층 Transformer로 처리한다.
각 표현은 2층 기체·압력 조건 적응 모듈을 거쳐 4층 융합 Transformer에서
통합된다. 공유 MLP 헤드와 기체별 단조 비감소 출력 변환을 통해 298 K에서
CO2 및 CH4의 각 5개 압력 조건 흡착량을 예측한다. 점선은 공동 학습에
사용하는 보조 예측 분기를 나타낸다.
