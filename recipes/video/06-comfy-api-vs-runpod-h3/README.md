# 06. 같은 MiniMax H3 워크플로를 Comfy API와 Runpod Serverless에 올려 비교하기

- **상태:** 2026-10-19 실측으로 검증됨
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 1 (양쪽 동일)
- **모델:** MiniMax H3 Ref2VA pruned INT8 ConvRot + Turbo4 Ref2V LoRA(4 step), Comfy-Org/MiniMax-H3
- **출력:** 1344×768, 24 FPS, 124프레임(5.167초), H.264 + AAC 32kHz 스테레오

ComfyUI 워크플로 하나를 Comfy API(Comfy Org 관리형 배포)와 Runpod Serverless(직접 만든 이미지·핸들러)에 같은 그래프로 올리고, 같은 GPU에서 cold 3회·warm 5회씩 요청해 준비 과정, 요청 시간, 요청당 비용을 비교합니다.

## 확인한 환경

| 항목 | Comfy API | Runpod Serverless |
| --- | --- | --- |
| 배포 | Build → Release(linux/nvidia, cu130) → Deployment | Docker 이미지 → Queue Endpoint |
| ComfyUI | 커밋 `5cc4b870` (0.41.0) | 커밋 `5cc4b870` (0.41.0) |
| torch / Python | 2.12.1+cu130 / 3.12.13 | 2.9.1+cu130 / 3.10.19 |
| 워커 | min 0 / max 1, 유휴 30초(고정) | min 0 / max 1, 유휴 5초, FlashBoot |
| 모델 위치 | Comfy network storage | 비공개 Hugging Face 캐시(Runpod 모델 캐시) |
| 요청 방식 | v2 API, 입력은 assets 업로드 | `/run` + `/status`, 입력·출력 base64 |
| 도구 | comfy-cli 1.22.0 | Runpod REST |

Runpod 쪽은 Runpod 기본 템플릿이 아니라 직접 만든 이미지와 핸들러입니다. 핸들러 코드는 이 레시피에 포함하지 않았습니다. `scripts/bench.py`의 Runpod 경로는 아래 형식을 받는 핸들러를 가정합니다.

```text
input:  source_video_base64, reference_image_base64, mode="turbo4", scope="head", prompt, seed, source_audio
output: videos[{role, filename, video_base64, sha256}], metadata{generation_request_seconds, ...}
```

## 라이선스와 주의

- 모델: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)와 원본 MiniMax H3의 사용 조건을 먼저 확인하세요. 가중치는 저장소에 넣지 않습니다.
- 입력: 사람 얼굴을 바꾸는 워크플로입니다. 사용 권한이 있는 영상과, 실존 인물이 아닌 인물 사진 또는 동의를 받은 사진만 쓰세요. 이 레시피에는 입력 파일을 넣지 않았습니다([inputs/README.md](inputs/README.md)).
- 비용: Comfy API는 Standard 이상 유료 플랜이 필요하고, GPU 시간과 저장 공간이 크레딧에서 별도로 빠집니다. 측정이 끝나면 Deployment와 Build를 모두 지워야 저장 공간 과금이 멈춥니다.

## Comfy API 쪽 준비

```bash
pip install "comfy-cli==1.22.0"
export COMFY_CLOUD_API_KEY=...   # build·deploy·status 명령용
export COMFY_API_KEY=...         # bench.py의 v2 API 호출용 (같은 계정 키)

cd comfy-build
comfy build validate . --remote
comfy build push
comfy build release create --target linux/nvidia --watch
comfy deploy up --gpu rtx-pro-6000-server --region anywhere --min 0 --max 1
```

`comfy-build.yaml`은 `comfy build init --from-workflow ../workflows/h3-ref2va-turbo4.api.json`으로 만든 뒤 모델 출처를 채운 스펙입니다. init만 하면 모델이 0개로 시작하므로, `comfy build refs resolve <파일 이름>`으로 찾은 공식 저장소 후보를 넣었습니다. 2026-10-19에는 Release가 6분 35초(39.2GB), Deployment가 약 4분 25초(모델 39.1GB 스테이징 포함) 걸렸습니다.

배포 주소가 나오면 `COMFY_BASE_URL=https://<deployment>.run.comfy.app`로 설정합니다.

## 실행

```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=<your endpoint>
export COMFY_API_KEY=... COMFY_BASE_URL=https://<deployment>.run.comfy.app

python3 scripts/bench.py comfy --label smoke --seed 42100     # 연결 확인 1회
python3 scripts/run_matrix.py --dry-run                       # 실행 순서 확인
python3 scripts/run_matrix.py                                 # 16회 실행 (약 1시간+)
```

`run_matrix.py`는 다음 순서로 실행하고, 결과를 `benchmark-output/`에 남깁니다.

1. Runpod cold-1 → 바로 이어서 warm-1~5
2. Comfy API cold-1 → 바로 이어서 warm-1~5
3. Runpod cold-2, Comfy cold-2, Runpod cold-3, Comfy cold-3

warm을 각 cold 바로 뒤에 붙인 이유는 Runpod의 유휴 5초 때문입니다. 양쪽 cold를 먼저 보내고 warm을 나중에 보내면 Runpod 워커가 그사이에 내려갑니다.

## 측정 설정

- 그래프: `workflows/h3-ref2va-turbo4.api.json` (노드 18개, 모두 ComfyUI 기본 노드). `res_multistep`/`simple`, 4 step, LoRA 강도 1.0
- 시드: 회차마다 다름(cold 42101~42103, warm 42111~42115), 같은 회차는 양쪽이 같은 시드
- 측정 시작·종료: 같은 Mac에서 작업 제출 직전부터 완료 상태를 처음 확인한 순간까지(`client_done_s`). 2초 간격 폴링, 결과 다운로드는 포함하지 않음
- cold 판정: 직전에 실행 중·기동 중인 워커 0, 대기 작업 0을 확인. Comfy API는 상태 스냅샷이 약 5분마다 갱신되므로, 마지막 작업이 끝나고 30초 뒤에 찍힌 스냅샷만 인정
- 플랫폼 기록은 따로 저장: Runpod `delayTime`·`executionTime`, Comfy `created_at`·`started_at`·`completed_at`. 두 정의가 달라 서로 빼거나 더해 비교하지 않음

## 결과 (2026-10-19, RTX PRO 6000 × 1)

| 클라이언트 기준 | Runpod Serverless | Comfy API |
| --- | ---: | ---: |
| warm 5회 중앙값 | 196.3초 | 165.1초 |
| warm 5회 범위 | 195.7–250.5초 | 164.6–167.4초 |
| cold 3회 | 221.9 / 221.8 / 371.2초 | 193.9 / 418.8 / 191.3초 |

- 샘플링 4스텝은 양쪽 모두 약 133초(스텝당 약 33초)로 같았습니다. warm 차이는 샘플링 밖 구간과 배정된 GPU 서버에서 났습니다. 자세한 분해는 [results/summary.md](results/summary.md)에 있습니다.
- cold는 워커 0이어도 빠르게 다시 뜨는 대기 슬롯 여부에 따라 대기가 1~229초로 흔들렸습니다.
- 같은 GPU의 시간당 단가는 Comfy API가 1.3배($4.54 vs $3.49)였고, warm 1건 계산값은 $0.203 vs $0.183, 다음 날 조회한 실제 청구 건당 평균은 $0.254(9건) vs $0.214(10건)였습니다.
- 개별 실행값: [results/runs.csv](results/runs.csv)

## 넓게 해석하면 안 되는 범위

- 워크플로 하나, GPU 한 종류, 하루 저녁의 측정입니다. 다른 워크플로·시간대·리전에서 같은 차이가 난다고 볼 근거는 없습니다.
- Runpod 쪽 warm 5회는 모두 한 워커(US-MO-2)가 처리했습니다. 같은 이미지를 다른 워커(US-NC-2)에서 돌리자 실행 시간이 약 19초 짧았습니다.
- 약 9초의 ComfyUI 실행 차이는 torch 버전(2.9.1의 conv3d 우회)이 가장 유력한 원인이지만, torch를 올려 다시 재지는 않았습니다.
- cold 3회는 범위로만 읽어야 합니다. 어느 쪽 cold가 더 빠르다고 결론 내릴 수 없습니다.
- Comfy API v2 응답에는 문서에 있는 `metrics.queue_ms`/`execution_ms`가 없었고, 상태 값은 실제 시작 전부터 `running`이었습니다(2026-10-19 기준). 그래서 타임스탬프로 계산합니다.

## 정적 검사

```bash
python3 -m unittest discover -s tests -v
```

GPU나 API 키 없이 그래프 설정, 측정 순서, 타임스탬프 계산, 결과 CSV를 확인합니다.
