# 01. Runpod 글로벌 볼륨과 네트워크 볼륨의 모델 로딩 속도 비교하기

- **상태:** 2026-10-22 실측으로 검증됨 (글로벌 볼륨 베타)
- **GPU:** Serverless NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 1 · Pod 측정은 RTX 4090(EU-RO-1), RTX PRO 6000(US-NC-2)
- **모델:** [PassingByPixels/Qwen3.8-27B-NVFP4](https://huggingface.co/PassingByPixels/Qwen3.8-27B-NVFP4) (20.61GB, 16개 파일) · MiniMax H3 Ref2VA Turbo4 (46.56GB, 6개 파일)

같은 모델 파일을 네트워크 볼륨(EU-RO-1)과 글로벌 볼륨에 넣고, ① 볼륨 사이 복사 시간, ② 새 Pod에서 처음 읽는 속도, ③ Serverless 워커가 0개일 때 첫 요청 대기와 첫 영상 생성 시간을 비교합니다.

## 확인한 환경

| 항목 | 네트워크 볼륨 | 글로벌 볼륨 |
| --- | --- | --- |
| 위치 | EU-RO-1, 80GB, STANDARD | 데이터센터 선택 없음 (베타) |
| Pod 마운트 | `/workspace` | `/workspace-global` (네트워크 볼륨과 함께 붙였을 때) |
| Serverless 마운트 | `/runpod-volume` | `/runpod-volume` |
| 파일 시스템 | `fuse` | `fuse.geesefs` (오브젝트 스토리지 기반) |
| LLM 워커 | Hub `worker-vllm` v2.28.0 (`runpod/worker-v1-vllm:v2.28.0@sha256:6d8e4ff4…`) | 같음 |
| 영상 워커 | 직접 만든 H3 ComfyUI 이미지 (이 레시피에 포함하지 않음) | 같음 |
| Serverless 설정 | min 0 / max 1, idle 30초, FlashBoot 끔 | 같음 |

## 라이선스와 주의

- 모델: Qwen3.8-27B-NVFP4와 MiniMax H3([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3))의 사용 조건을 먼저 확인하세요. 가중치는 저장소에 넣지 않습니다.
- H3 측정은 직접 만든 이미지와, 모델을 옮겨 둔 비공개 Hugging Face 저장소를 썼습니다. 저장소 이름·리비전·이미지는 `H3_REPO`, `H3_REV`, `H3_IMAGE` 환경변수로 넣습니다. 이 워커는 기동할 때 모델 파일 SHA-256을 전부 계산하는 단계를 일부러 넣었습니다. ComfyUI 기본 동작이 아닙니다.
- 입력: 얼굴을 바꾸는 워크플로라 사용 권한이 있는 영상과 동의를 받은 사진만 쓰세요([inputs/README.md](inputs/README.md)).
- 비용: 2026-10-22 측정 전체에 Pod 약 $1.80(청구액), Serverless는 실행 시간 기준 약 $15–17(추정)이 들었습니다. 글로벌 볼륨은 저장량과 요청 수로 과금되고, 계정 잔액이 $0이 되면 15일 뒤 볼륨이 삭제됩니다. 측정이 끝나면 엔드포인트·볼륨을 모두 지우세요.

## 글로벌 볼륨에 모델 넣기

1. 콘솔 Storage → + New volume → 종류 Global volume. 데이터센터와 용량은 고르지 않습니다.
2. 네트워크 볼륨에 모델이 있다면, Pod 하나에 두 볼륨을 같이 붙이고 복사합니다. 공식 문서는 `rsync --archive`를 안내하지만, 글로벌 볼륨은 파일 시각을 저장하지 못해서 시각 복사 옵션(`-t`)을 넣으면 `rsync`가 오류 코드 23으로 끝났습니다(2026-10-22). 이 레시피는 다음처럼 복사합니다.

   ```bash
   rsync -rl --inplace --exclude=.cache /workspace/models/ /workspace-global/models/
   ```

   `-rl --inplace`로도 디렉터리 권한·시각 경고가 남지만([results/step1-rsync-warnings.txt](results/step1-rsync-warnings.txt)), 복사 후 SHA-256 검증은 4건 모두 통과했습니다.
3. 글로벌 볼륨은 파일 잠금, 원자적 rename, 하드 링크, 권한 설정이 안 됩니다. 볼륨에는 모델 파일만 두고, ComfyUI·파이썬 환경은 이미지에 넣습니다.
4. Serverless 엔드포인트 Advanced에서 Global Volume을 고릅니다. 엔드포인트 하나에는 네트워크 볼륨과 글로벌 볼륨 중 하나만 붙일 수 있습니다.

## 실행

```bash
read -s RUNPOD_API_KEY; export RUNPOD_API_KEY
export NV_ID=<network volume id> GV_ID=<global volume id>
export H3_REPO=<org/name> H3_REV=<revision> H3_IMAGE=<your H3 worker image>   # 영상 쪽만

# 1단계 (Pod 안): 받기 → 검증 → 복사. pod_step1.sh가 read_bench.sh를 차례로 부릅니다
./read_bench.sh copy a1-qwen /workspace/models /workspace-global/models

# 2단계 (Mac): 회차마다 새 Pod를 만들고, 첫 읽기를 잰 뒤 지웁니다
python3 step2_round.py eu-r1 --dc EU-RO-1 --gpu "NVIDIA GeForce RTX 4090" --targets "nv=/workspace gv=/workspace-global"
python3 step2_round.py us-r1 --dc US-NC-2 --gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition" --targets "gv=/workspace-global" --no-nv

# 3단계 (Mac): 엔드포인트 5개를 만들고 cold/warm 요청
python3 step3_create.py
python3 bench.py order            # 실행 순서 확인
python3 step3_run.py
```

| 파일 | 하는 일 |
| --- | --- |
| `read_bench.sh` | Pod 안: 복사 시간, 큰 파일 첫 읽기, 작은 파일 1,000개, SHA-256 검증 |
| `pod_step1.sh` · `pod_step2.sh` | Pod 시작 스크립트 (1단계 준비, 2단계 읽기) |
| `step2_round.py` | 새 Pod 생성 → 결과 수집 → 삭제(404 확인) |
| `step3_create.py` · `step3_run.py` · `bench.py` | 엔드포인트 생성, cold/warm 요청, 워커 로그 요약 |
| `manifests/*.sha256` | 모델 파일 검증 목록 |

## 측정 설정

- 큰 파일 첫 읽기: 회차마다 새 Pod에서 `dd bs=16M of=/dev/null`, 같은 Pod에서 같은 파일을 두 번 읽지 않음(메모리 캐시). 파일은 Qwen 9.97GB, H3 확산 모델 21.0GB. 3회.
- cold: 직전에 `/health`의 idle·ready·running·initializing 워커가 0인지 확인하고 보냄. 6분을 기다려도 대기 슬롯이 남아 그대로 보낸 회차는 `forced_cold: true`로 표시. 엔드포인트마다 첫 요청(이미지 받기 포함)은 버림.
- 대기 시간: Runpod `delayTime`. 첫 영상 생성: `executionTime`. warm: 워커가 떠 있을 때 `executionTime` 중앙값.
- 기동 내부 시간: LLM은 vLLM 로그의 가중치 로딩, H3는 `H3_MODEL_READY` 줄의 46.56GB 해시 확인 시간.

## 결과 (2026-10-22)

복사 (EU-RO-1 RTX 4090 Pod, 1회)

| 모델 | 네트워크 → 글로벌 볼륨 |
| --- | ---: |
| Qwen 20.61GB | 124.3초 (165.8MB/s) |
| H3 46.56GB | 215.4초 (216.1MB/s) |

큰 파일 첫 읽기 (3회, 가운데 값 굵게)

| 위치 | Qwen 9.97GB | H3 21.0GB | 작은 파일 1,000개 |
| --- | --- | --- | --- |
| 네트워크 볼륨 · EU-RO-1 | 1,036.5 / **1,066.0** / 1,160.0MB/s | 990.8 / **1,065.2** / 1,073.0MB/s | 7.1 / **8.0** / 8.3초 |
| 글로벌 볼륨 · EU-RO-1 | 159.8 / **178.9** / 216.2MB/s | 161.8 / **173.0** / 271.2MB/s | 26.5 / **29.6** / 100.1초 |
| 글로벌 볼륨 · US-NC-2 | 94.9 / **104.3** / 124.5MB/s | 86.1 / **103.0** / 138.0MB/s | 217.7 / **222.3** / 335.1초 |

LLM 첫 요청 (Qwen3.8-27B NVFP4, RTX PRO 6000)

| 조건 | cold 대기 3회 | 가중치 로딩 | 워커 위치 | warm 실행 중앙값 |
| --- | --- | --- | --- | --- |
| 볼륨 없음 (HF 다운로드) | 179 / 269 / 327초 | 다운로드 포함 12–157초 | US-NE-1, US-NC-2 | 0.4초 |
| 네트워크 볼륨 | 151 / 157 / 206초 | 55–60초 | EU-RO-1 | 0.4초 |
| 글로벌 볼륨 | 342 / 376 / 461초 | 172–285초 | US-PA-1 | 0.4초 |

영상 첫 요청 (H3 Ref2VA Turbo4)

| 조건 | 해시 확인 46.56GB | 첫 생성 | warm 생성 |
| --- | --- | --- | --- |
| 네트워크 볼륨 (cold 3회, warm 6회) | 46–68초 (682–1,023MB/s) | 171–240초 | 중앙값 190초 |
| 글로벌 볼륨 (cold 2회, warm 2회) | 272–333초 (140–171MB/s) | 780초 / 1회 실패 | 206초 (같은 워커) / 853초 (새 워커) |

- 글로벌 볼륨 cold 2회는 두 번 다 EU-RO-1 워커였습니다. 실패한 회차는 모델을 읽느라 첫 스텝까지 15분 35초가 걸렸고, 25분째 핸들러가 생성을 중단했습니다.
- 이 H3 워커는 ComfyUI가 첫 생성에서 모델을 볼륨에서 다시 읽습니다(`dynamic VRAM loading`, `fast_disk=False`). 그래서 느린 읽기가 기동과 첫 생성 양쪽에 붙습니다.
- 개별 기록: [results/](results/) (`serverless-runs.jsonl`, `step1-copy.jsonl`, `step2-read-*.jsonl`, `step3-summary.txt`). Pod·워커·작업 ID는 지웠습니다.

## 넓게 해석하면 안 되는 범위

- 하루(2026-10-22)의 측정이고 글로벌 볼륨은 베타입니다. 성능은 바뀔 수 있습니다.
- LLM은 글로벌 볼륨 워커가 세 번 다 US-PA-1, 네트워크 볼륨 워커는 EU-RO-1에 떴습니다. 위치가 다른 상태의 비교입니다.
- 글로벌 볼륨 데이터가 실제로 어디에 저장되는지는 공개돼 있지 않아, 미국에서 느린 원인은 추정하지 않습니다.
- H3 글로벌 볼륨은 비용 때문에 cold 2회만 쟀습니다. cold 대기 348–1,443초 중 1,443초는 대부분 GPU 배정 대기라 볼륨 속도로 읽으면 안 됩니다.
- 영상 워커는 해시 확인 단계를 넣은 직접 만든 이미지입니다. 다른 ComfyUI 워커는 읽는 양과 횟수가 다를 수 있습니다.
- Runpod 모델 캐시(엔드포인트 Model 칸에 HF 모델 ID)는 이번에 같은 조건으로 재지 않았습니다.

## 정적 검사

```bash
python3 -m unittest discover -s tests -v
```

리소스나 API 키 없이 로그 요약, 엔드포인트 환경변수 이름, 요청 본문(`--dry-run`), `read_bench.sh`의 캐시 읽기 방지를 확인합니다.
