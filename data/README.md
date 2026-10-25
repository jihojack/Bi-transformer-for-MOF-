# 데이터 및 모델 자산

## 포함한 자료

- `splits/fixed_split_70_10_20.json`: 원래 50K의 train/val/test 인덱스·MOF ID
- `splits/mof_50k_split_sampling.csv`: 행과 분할·샘플링 단계의 대응
- `splits/external_10k_ids.csv`: 실제 추가 평가의 표본 순서 및 MOF ID
- `build_50k_manifest.json`: 원본 구성 및 해시 기록
- `external_sampling_manifest.json`: 추가 10K 추출 seed·모집단·해시 기록
- `external_colab_manifest.json`: Colab용 입력·라벨·CIF 일치 및 중복 검사 기록
- `../app/assets/`: 공개 전처리 파라미터와 체크포인트 해시 메타데이터 (가중치 없음)
- `../app/examples/`: 예제 CIF 20개, 기술자와 GCMC 기준값
- `../results/internal/test_predictions.csv`: 내부 시험 10K의 기준값·예측값

구축 및 샘플링 manifest의 개인 PC 절대경로는 파일명으로 정리했습니다.
파일 SHA-256과 샘플 구성 정보는 유지했습니다. CPU와 Colab용 외부 입력은
CSV 형식·해시가 다를 수 있으므로 해당 manifest와 파일을 함께 사용합니다.

## 별도로 준비할 자료

대량 원본 archive·CIF ZIP·HDF5·학습 캐시·optimizer 체크포인트는 포함하지
않았습니다. 원래 데이터 출처는 MOFX-DB의 hMOF 계열 자료이며 원본 archive
파일명은 `bulk-dl-mofdb-version-dc8a0295db.zip`입니다. 같은 버전 archive의
필터링·단위·압력 보간 조건을 유지해야 원래 ID 구성과 지표를 재현할 수 있습니다.

13번 구축 스크립트는 원래 30K를 유지해 확장하도록 설계되어 있어,
정확한 재구축에는 기존 `MOF_CO2_30K_light.csv`도 필요합니다.
`--existing-csv`로 지정하며 이 파일은 이 저장소에 포함하지 않았습니다.
이미 최종 50K CSV·CIF를 준비했다면 14번 특성 추출 단계에서 시작할 수 있습니다.

일반적인 로컬 배치 위치:

```text
data/raw/bulk-dl-mofdb-version-dc8a0295db.zip
data/processed/MOF_CO2_CH4_50K.csv
data/processed/MOF_CO2_CH4_50K_CIF.zip
data/processed/MOF_CO2_CH4_50K_alchemi_enriched.csv
data/processed/periodic_graphs_50k_alchemi_compact.h5
data/external_inputs/external_candidate_pool_inputs.csv
data/external_inputs/external_candidate_pool_cifs.zip
data/external_inputs/external_candidate_pool_labels_private.csv
```

이 로컬 데이터 폴더들은 Git 관리에서 제외합니다.

## 모델

학습된 가중치는 공개 저장소와 Git 이력에 포함하지 않습니다. 서버 운영자가
체크포인트 또는 변환된 ONNX를 별도 비공개 보관소에서 준비해야 합니다.
`server/model_manifest.json`에는 파일 크기와 SHA-256만 공개합니다.
개별 예제의 입력/출력 수치는 모델 가중치와 다릅니다.

## 출처와 이용 조건

MOFX-DB 원본 데이터 및 외부 소프트웨어의 이용 조건과 인용 요건은
각 제공자의 자료를 따릅니다. 이 폴더가 원본 데이터의 권리를 변경하거나
새 이용 권한을 부여하는 것은 아닙니다.
