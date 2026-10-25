# 실행 및 재현

## 재현 가능 범위

1. 공개 자료만으로 분할 중복 검사, 저장된 예측의 분석·그림 재생성, API 계약·전처리 검사를 수행할 수 있습니다.
2. 원본 데이터 및 기존 30K CSV를 별도로 준비하면 아래 구축·학습 절차를 실행할 수 있습니다. 가중치를 비공개로 유지하므로 공개 자료만으로 원래 동결 모델을 즉시 실행하는 완전한 재현 패키지는 아닙니다.
3. 원래 비공개 체크포인트를 가진 운영자는 해시를 검증하고 ONNX를 변환해 동결 추론을 대조할 수 있습니다. 모델을 다른 파일로 바꾸면 해시도 명시적으로 바꿔야 합니다.

## 서버 CPU 앱 및 동결 추론

README의 Python 3.11 환경과 `requirements-server.txt`를 설치하고 `MOF_MODEL_PATH`를 비공개 ONNX 파일로 지정합니다. `python -m server.app`은 127.0.0.1:7866에서 실행합니다. 원격 서비스는 [배포 문서](deployment.md)의 Gunicorn 설정을 사용합니다. 공개 저장소나 브라우저에는 가중치를 넣지 않습니다.

기본 예제는 hmof-5067709입니다. 기준 출력은 아래와 같으며 CPU 환경에 따라
작은 부동소수점 차이가 생길 수 있습니다.

| 기체 | 압력(bar) | 예측 mmol/g |
|---|---:|---:|
| CO2 | 0.01 | 0.0183305051 |
| CO2 | 0.05 | 0.0922820643 |
| CO2 | 0.1 | 0.1875720620 |
| CO2 | 0.5 | 0.9397379756 |
| CO2 | 2.5 | 4.3403620720 |
| CH4 | 0.05 | 0.0384717733 |
| CH4 | 0.5 | 0.3975046277 |
| CH4 | 0.9 | 0.7140058875 |
| CH4 | 2.5 | 1.9331362247 |
| CH4 | 4.5 | 3.3641068935 |

`tools/verify_private_inference.py`는 실제 35개 기술자 변경 사례와 CIF 업로드를 서버 API에 보내 원래 PyTorch 출력과 비교합니다(허용 최대 절대차 1e-4 mmol/g). `tests/test_server.py`는 공개 API 계약과 20개 CIF 전처리를 검사하지만 가짜 예측기로 수행하는 API 계약 검사를 실제 모델 정확도 검증으로 해석하지 않습니다. 추가 10K 전체 재추론이나 재학습은 이 검사에 포함되지 않습니다.

앱의 예제 목록은 현재 18개입니다. 제외한 원래 12·13번 구조도 검증 자료에는 남겨 두었으며, 35개 사례 검사는 두 구조를 일반 CIF 업로드 경로로 보내므로 데모 변경 때문에 수치 검증 사례를 줄이지 않습니다. `node --test tests/frontend.test.cjs`는 자동 연결 재시도, 예제 수 표시, 18개 순차 예측 및 결과 탐색을 DOM 모형으로 검사합니다. 실제 브라우저 렌더링 검사와는 구분합니다.

변환 도구는 신뢰할 수 있는 본인 체크포인트만 입력합니다. pickle 체크포인트는 실행 코드를 포함할 수 있습니다.

```text
python tools/export_server_model.py --checkpoint /private/best_dual_gas_alchemi_50k.pt --output /private/new-model.onnx
```

출력은 공개 체크아웃 밖의 새 파일이어야 합니다. 변환 시 중복 인덱스 합산을 유지하는 ScatterElements, exp(x)-1 및 동적 shape 연산을 사용하며 원래 PyTorch 기준값과 20개 구조·200개 출력을 비교합니다. 검증 JSON은 비공개 출력 파일 옆에 생성합니다.

## 그림 재생성

```bash
pip install -r requirements-plots.txt
python figures/architecture/draw_architecture.py
python figures/tower_comparison/plot_tower_comparison.py
python figures/structure_error/plot_structure_error.py
```

구조별 그림은 기본적으로 함께 제공한 검증된 구간 통계 CSV에서 생성합니다.
`--recompute` 옵션을 사용하면 함께 제공한 MOF별 오차 자료에서 구간 및
신뢰구간을 다시 계산합니다. 두 경로 모두 모델 재추론은 하지 않습니다.
한글 글꼴이 없는 환경에서는 Malgun Gothic 또는 Noto Sans CJK KR 등의
글꼴을 설치해야 한글판이 정상 표시됩니다. 영문판도 제공합니다.

## 전체 GPU 학습

고정 seed는 42이지만 원본 설정의 `cudnn.deterministic=False`와 GPU 병렬 연산 때문에 재학습 결과의 비트 단위 일치는 보장하지 않습니다. 새 학습은 빈 출력 폴더에서 `resume_training=False`로 시작하여 이전 상태를 이어받지 않도록 합니다. 원래 고정 분할 35K/5K/10K를 유지하고, 추가 평가 10K는 이 50K 전체에서 제외합니다. 평가 결과를 보고 추가 표본을 다시 뽑거나 학습·조율에 사용하지 않습니다.

다음 Python/Colab 파일은 실제 연구 파이프라인입니다. 새 실행에는
`data/README.md`에 정리한 원본 파일이 별도로 필요합니다.

1. `pipeline/13.Build_MOF_CO2_CH4_50K.py`: 원본 archive와 기존 30K CSV(`--existing-csv`)에서 50K와 분할·CIF ZIP 구성
2. `pipeline/14.ALCHEMI_CO2_CH4_50K_Feature_Extraction.py` 또는 ipynb: Colab GPU에서 이웃·강화 특성 구성
3. `pipeline/15.DualGas_ALCHEMI_50K_L4_Training.py` 또는 ipynb: 학습·내부 시험
4. `pipeline/16.DualGas_50K_Extended_Evaluation.py`: 학습 직후 **같은 런타임**에서 `%run -i`로 실행
5. `pipeline/17.Descriptor_Baseline_Comparison_50K_Colab.py`: 별도 descriptor baseline 비교

14–17의 원래 Colab 기본 경로는 `/content/drive/MyDrive/mf_hid/4th`입니다.
자신의 Drive 위치에 맞춰 Config 또는 셀 상단 경로를 바꾸고 서로 같은
데이터·분할·전처리 자산을 사용해야 합니다. 특히 16은 학습 런타임 변수에
의존하므로 단독 실행 파일처럼 실행하지 않습니다.

GPU 단계는 ALCHEMI Toolkit-Ops 0.3.1, Warp, ASE, pymatgen, mendeleev 등을
설치하는 셀을 포함합니다. CPU 앱용 가상환경과 Colab GPU 환경을 분리하세요.
전처리 joblib는 scikit-learn 버전에 영향을 받을 수 있으며 제공된 CPU 설치
버전과 새 학습 환경에서의 버전이 다르면 재현 수치가 변할 수 있습니다.

## 추가 10K: CPU

데이터 archive와 원래 50K CSV를 준비한 뒤:

```bash
python evaluation/external_10k_sample.py --source-zip data/raw/bulk-dl-mofdb-version-dc8a0295db.zip --existing-csv data/processed/MOF_CO2_CH4_50K.csv --output-dir data/external_inputs
python inference/mof_cpu_inference.py --external-dir data/external_inputs --run-external -1
python inference/mof_cpu_inference.py --external-dir data/external_inputs --evaluate
```

원본 내부 HDF5와의 그래프 대조는 선택 기능입니다. 원본 자산을
`--original-dir`로 지정하고 `--validate-original N`을 사용할 수 있습니다.
파일 이름과 지원 옵션은 `--help`에서 확인합니다.

## 추가 10K: Colab ALCHEMI

`evaluation/external/`의 notebook과 본체 Python 파일을 별도 Colab 폴더에 두고
입력 CSV, CIF ZIP, 분리된 정답 CSV 및 동결 모델·전처리 자산을 준비합니다.
기본 `MOF_BASE_DIR`는 원래 연구 Drive 경로이고 환경변수로 변경할 수 있습니다.
출력 예측이 저장·해시된 뒤 정답을 읽는 순서를 유지합니다.
이 저장소에 업로드된 요약값은 원래 실행 결과 기록이며 GPU 전체 재평가를
새로 수행한 결과는 아닙니다.
