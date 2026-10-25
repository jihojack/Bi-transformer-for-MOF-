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
gunicorn --config gunicorn.conf.py --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT server.app:app
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

## 시작 실패 진단

`[mof_startup]`으로 시작하는 JSON 로그는 즉시 출력되며, 비밀 URL·경로·토큰·원본 예외 메시지를 기록하지 않는다. 정상 순서는 `private_model_preparation_started` → `private_model_verified` → `application_created` → `model_loader_thread_started` → `model_hash_verified` → `creating_onnx_session` 단계 → `ready` 단계이다. ONNX 로딩 중에는 15초 간격 진행 로그가 나오며 Linux에서는 프로세스 RSS와 컨테이너 메모리 사용량/상한도 포함한다.

- `private_model_preparation_failed`: 다운로드 또는 파일 검증 실패. HTTP 오류면 상태 번호만 기록한다.
- `model_load_failed`: `error_code` 및 실패 단계를 확인한다. 파일 없음, 해시 불일치, ONNX 세션 초기화 오류를 구분한다.
- `creating_onnx_session`에서 진행 로그만 반복: 네이티브 모델 초기화가 끝나지 않은 상태다. 이 로그만으로 CPU·메모리·교착 중 원인을 확정하지 않는다.
- `model_warm_start_disabled`: `MOF_SKIP_WARM=1`이 설정되었는지 확인한다.
- `/api/status`는 초기화 단계, 경과 시간, 로더 스레드 생존 여부, 초기화/현재 프로세스 ID를 반환한다. `/healthz`는 여전히 실제 모델 준비 전에는 503이다. 진단을 위해 가짜 성공 응답으로 바꾸지 않는다.

이 변경은 초기화 진단용이다. 가중치, 연산 옵션, 최적화 수준, 예측 계산은 바꾸지 않는다. 2026-10-25의 첫 Render 배포는 준비 상태 검사 503 반복 후 시간 초과했으며, 정확한 원인은 당시 로그만으로 확정하지 못했다. 새 커밋의 클라우드 성공 여부는 다음 배포 로그로 별도 검증해야 한다.

### 확인된 pre-fork 초기화 오류와 수정

후속 Render 로그에서 다운로드·해시 검증은 통과했고, 부모 PID 57의 모델은 5.204초에 `ready`가 되었다. 그러나 요청을 처리한 워커 PID 62는 부모의 로딩 중 상태를 복제한 채 로더 스레드가 없는 상태였다. 부모가 모델을 다 올려도 워커의 Python 객체에는 반영되지 않으므로 준비 상태 검사가 계속 503이었다. 이 실행에서는 다운로드 인증이나 모델 로딩 속도가 실패 원인이 아니었다.

WSGI 모듈 import 시에는 이제 로더를 시작하지 않는다. `gunicorn.conf.py`의 `post_worker_init`가 fork 이후 워커에서만 모델을 초기화하며, 부모에서 복제한 Predictor/락도 새로 만든다. 설정 파일을 호스트가 덮어쓰는 경우 첫 요청에서도 같은 시작 함수를 호출한다. 동일 워커에서 중복 스레드는 만들지 않으며 준비 상태는 여전히 실제 모델 로딩 완료 후에만 200이다. 로그의 `model_loader_thread_started`/`ready` PID와 `/api/status`의 `process_id`, `loader_process_id`가 일치해야 한다.

토큰·모델 URL·가중치·ONNX 연산 옵션은 바꾸지 않는다. Linux CI는 공개 가중치 대신 임시 합성 Identity 그래프로 실제 ONNX Runtime/Gunicorn을 구동하여 일반 시작, `--preload` 시작, 훅이 없는 경우의 첫 요청 시작을 확인한다. 실제 Render 재배포 성공은 별도 확인한다.

## 안전·운영 한계

- 업로드 CIF는 요청 단위 임시 폴더에서 처리 후 삭제한다. 앱은 파일 내용·인증 토큰을 로그에 남기지 않는다. 호스팅 제공자의 접속 로그 정책은 별도다.
- 완전 점유 단일 주기 구조와 동결 원소 표의 11개 원소만 지원한다. 부분 점유, 다른 원소, 지나치게 조밀한 격자는 거부한다.
- 출처 허용 목록과 메모리 보호 제한은 인증을 대신하지 않는다. 민감한 CIF를 받는 공개 서비스라면 HTTPS, 로그인 및 추가적인 API 제한이 필요하다.
- 가중치 다운로드 경로는 제공하지 않지만 무제한 질의를 통한 모델 모방까지 방지한다고 주장하지 않는다.

공식 문서: [Render Flask](https://render.com/docs/deploy-flask), [Render 무료 제한](https://render.com/docs/free), [Flask Gunicorn](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/).
