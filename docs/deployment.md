# 비공개 모델 서버 배포

구성: 브라우저/Netlify → Flask API → 서버 내부 ONNX 모델 → 예측 수치 반환. 클라이언트에는 모델, 텐서 그래프, 다운로드 토큰을 보내지 않는다. 공개 저장소에는 가중치가 없으며 자동 배포만 눌러서는 예측 서버가 완성되지 않는다.

## 1. 비공개 모델 준비

원래 소유한 체크포인트로 `tools/export_server_model.py`를 실행하거나, 이미 검증한 서버용 ONNX 파일을 비공개 위치에 보관한다. SHA-256과 크기는 `server/model_manifest.json`에 있다. 모델의 정밀도는 FP32이며 재학습·양자화를 하지 않는다.

모델은 다음 중 하나로 서버에만 공급한다.

- VM 또는 컨테이너의 비공개 파일 마운트. `MOF_MODEL_PATH`로 위치를 지정한다.
- 서버만 접근 가능한 HTTPS artifact 저장소. Render 비밀 환경변수에 `MOF_MODEL_URL`, 필요 시 `MOF_MODEL_BEARER_TOKEN`을 넣는다. `tools/download_private_model.py`가 시작 시 다운로드하고 해시를 검증한다.

공개 GitHub, 공개 Docker 이미지, Netlify 파일, frontend/config.js에 모델이나 비밀 URL/토큰을 넣지 않는다. 비밀 URL을 브라우저에 숨겨 넣는 방식도 사용하지 않는다. 클라우드 제공자와 서버 관리자에게까지 가중치를 숨기는 구조는 아니며, 방문자에게 배포하지 않는 구조다.

## 2. 로컬 서버 검증

Python 3.11에서 `pip install -r requirements-server.txt` 후, PowerShell 예:

```powershell
$env:MOF_MODEL_PATH='D:\private-models\mof_fusion.onnx'
python -m server.app
```

위 경로는 사용자 본인의 비공개 파일 위치로 바꾼다. http://127.0.0.1:7866 에서 확인한다. Windows의 이 명령은 로컬 확인용 Flask 서버이며, Linux 배포에는 아래 Gunicorn을 사용한다.

```text
python tools/verify_private_inference.py
gunicorn --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT server.app:app
```

## 3. Render 무료 시험 배포

`render.yaml`로 Python Web Service를 만든다. `requirements-server.txt`에는 PyTorch·pymatgen·pandas를 넣지 않아 메모리를 줄였다. 워커 1개, 동시 예측 1개, ONNX 스레드 1개로 제한한다. 기본 원자 수 상한은 200개이다. 더 큰 구조는 서버 메모리를 확인한 뒤 `MOF_MAX_ATOMS`를 조정한다.

Render에서 실제 설정할 값:

- `MOF_MODEL_URL`: 본인 비공개 모델 저장소의 HTTPS 다운로드 endpoint
- `MOF_MODEL_BEARER_TOKEN`: 필요한 경우 읽기 전용 토큰
- `MOF_ALLOWED_ORIGINS`: Netlify를 쓸 경우 정확한 `https://...netlify.app` 주소
- 재학습한 다른 모델이면 `MOF_MODEL_SHA256`도 해당 파일의 값으로 변경

`/healthz`가 준비 상태에만 200을 반환한다. 모델 공급이 안 됐으면 503으로 실패하며 가짜 예측으로 대체하지 않는다. 무료 인스턴스 절전·콜드 스타트·속도·메모리는 실제 클라우드에서 추가 확인해야 한다. 이 저장소 준비만으로 클라우드 배포를 했다고 주장하지 않는다.

## 4. Netlify 화면만 배포

서버 주소가 정해진 뒤:

```text
python tools/package_frontend.py --output outputs/netlify-client --api-origin https://YOUR-API-HOST
```

생성 폴더 전체를 기존 Netlify 프로젝트에 배포한다. 이 폴더는 HTML/CSS/JS 네 파일뿐이며 가중치가 없다. `MOF_ALLOWED_ORIGINS`에도 같은 Netlify 출처를 추가한다. 같은 서버에서 화면도 제공하면 이 단계는 필요 없다.

기존의 모델 포함 정적 배포는 별개다. 새 화면을 배포해도 이전 Netlify deploy URL이 사라진다고 가정하지 말고, 이전 배포의 접근과 모델 파일도 서비스 관리 화면에서 정리해야 한다.

## 안전·운영 한계

- 업로드 CIF는 요청 단위 임시 폴더에서 처리 후 삭제한다. 앱은 파일 내용·인증 토큰을 로그에 남기지 않는다. 호스팅 제공자의 접속 로그 정책은 별도다.
- 완전 점유 단일 주기 구조와 동결 원소 표의 11개 원소만 지원한다. 부분 점유, 다른 원소, 지나치게 조밀한 격자는 거부한다.
- 출처 허용 목록과 메모리 보호 제한은 인증을 대신하지 않는다. 민감한 CIF를 받는 공개 서비스라면 HTTPS, 로그인 및 추가적인 API 제한이 필요하다.
- 가중치 다운로드 경로는 제공하지 않지만 무제한 질의를 통한 모델 모방까지 방지한다고 주장하지 않는다.

공식 문서: [Render Flask](https://render.com/docs/deploy-flask), [Render 무료 제한](https://render.com/docs/free), [Flask Gunicorn](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/).
