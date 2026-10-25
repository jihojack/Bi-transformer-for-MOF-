MOF Adsorption Predictor
========================

새 앱 실행: 저장소 루트에서 requirements-server.txt를 설치하고
MOF_MODEL_PATH를 비공개 ONNX 파일로 지정한 뒤 python -m server.app 실행.
앱 주소: http://127.0.0.1:7866
이 폴더는 원래 PyTorch 코드·공개 전처리·예제 보존용입니다.
학습된 가중치는 포함하지 않습니다. 새 화면은 frontend/, API는 server/입니다.
종료 방법: 검은색 실행 창을 닫거나 Ctrl+C를 누릅니다.

주의
- CIF와 기술자는 지정한 서버로 전송됩니다. 가중치는 방문자에게 전송하지 않습니다.
- 학습 당시와 같은 정의의 구조 기술자를 입력해야 합니다.
- 출력은 298 K 순수 성분 GCMC 기반 학습 분포의 모델 예측값입니다.
- 실험값 또는 결함/유연 MOF에 대한 보증값이 아닙니다.
