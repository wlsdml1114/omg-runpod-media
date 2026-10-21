# 07. MiniMax H3 Turbo LoRA를 켜고 끄며 학습 조건(step·shift·샘플러) 확인하기

- **상태:** 2026-10-17 실측(1차 18회 + 재실험 8회, 각 조건 1회)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 1
- **모델:** MiniMax H3 FL2VA pruned INT8 ConvRot (Comfy-Org/MiniMax-H3) + lightx2v/Minimax-h3-Turbo LoRA
- **출력:** 1344×768, 24 FPS, 124프레임(5.167초), 영상 + 소리

H3 Turbo LoRA를 켜고 끄며 생성 시간, step별 예측 화면(x0), 그리고 학습 조건(step 수·shift·샘플러·강도)을 바꿨을 때의 결과를 비교합니다. 바로 쓸 수 있는 권장 워크플로도 함께 넣었습니다.

## 바로 쓸 설정

768p로 학습한 Turbo LoRA는 **video shift 6 / audio shift 3**으로 학습됐습니다([ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo)). ComfyUI H3 템플릿에 LoRA만 넣으면 원본 모델 기본값인 shift 12로 실행됩니다.

| 용도 | 워크플로 | 설정 | 생성 시간(1회) |
| --- | --- | --- | --- |
| 초안 | `workflows/h3-fl2va-turbo4-shift6-er_sde-075.api.json` | Turbo 4-step v1.0 768p, `ModelSamplingMiniMaxH3`(6 / 3), er_sde, simple, 4 step, LoRA 강도 0.75 | 47.1초 |
| 최종 | `workflows/h3-fl2va-base-20step.api.json` | LoRA 없음, res_multistep, simple, 20 step | 192.2초 |

두 워크플로 모두 ComfyUI 기본 노드만 씁니다. `LoadImage`의 파일 이름(`real-hd-01-face.png`)과 `MiniMaxH3ImageToVideo`의 프롬프트를 바꿔서 쓰세요. 544p로 학습한 LoRA(4-step v0.1, 8-step v1.0)는 README 기준 shift 12 / 3이 학습 조건입니다.

## 확인한 환경

| 항목 | 값 |
| --- | --- |
| ComfyUI | 커밋 `5cc4b870` (`inputs/source-revisions.json`) |
| 이미지 | `runpod/pytorch:1.4.1-cu1300-torch291-ubuntu2404`, Python 3.12, torch 2.9.1+cu130 |
| 모델 | `inputs/model-download-manifest.json` (Comfy-Org/MiniMax-H3 rev `e5eb578`, lightx2v/Minimax-h3-Turbo rev `a29e57d`, SHA-256 포함) |
| 기본 그래프 | 2026-09-10 FL2VA 템플릿(`workflows/template-turbo4-i2v.api.json`)에서 해상도·LoRA·step·seed만 바꿈. CFG 없음(BasicGuider) |

## 라이선스와 주의

- 모델: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 사용 조건과 지역 제한(대한민국 배포는 별도 허가 필요)을 먼저 확인하세요. LoRA는 [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)(Apache-2.0)입니다. 가중치는 저장소에 넣지 않습니다.
- 입력: 실험에 쓴 첫 프레임은 AI로 만든 가상 성인 인물 사진이라 저장소에 넣지 않았습니다. 사용 권한이 있는 이미지를 `inputs/real-hd-01-face.png`로 넣으세요.
- 비용: RTX PRO 6000 Pod 1개, 1차 실험 약 38분 + 재실험 약 23분(Secure $2.49/h 기준 약 $2.6 추정). 설치·다운로드(약 50GB) 시간이 포함됩니다.

## 실행 순서

```bash
# 로컬: 실행 목록과 워크플로 만들기
python3 scripts/prepare.py        # 1차 18회 (shift 12)
python3 scripts/prepare_r2.py     # 재실험 8회 (shift 6, 샘플러·강도)

# Pod: 이 폴더를 /workspace/h3-turbo-ablation 으로 복사한 뒤
cd /workspace/h3-turbo-ablation
nohup python3 scripts/setup_remote.py > logs/setup.log 2>&1 &
nohup python3 scripts/run_remote.py > logs/run.log 2>&1 &          # 전체
# python3 scripts/run_remote.py <run_id> ...                       # 일부만

# 로컬: 결과를 받아온 뒤 수치 계산
python3 scripts/analyze.py
```

`setup_remote.py`는 ComfyUI를 고정 커밋으로 받고, 모델을 SHA-256으로 검증하며 내려받고, `custom_nodes/x0capture`를 설치합니다. `run_remote.py`는 ComfyUI를 `127.0.0.1:8188`에서만 띄우고, 실행 목록을 순서대로 처리한 뒤 x0를 저장한 실행은 step별 예측을 영상으로 디코딩합니다. 결과는 `runs/<run_id>/native.mp4`, `x0-decoded/`, `runs.jsonl`에 남습니다.

### x0capture 노드

`SamplerCustomAdvancedX0Steps`는 ComfyUI `SamplerCustomAdvanced`와 같은 방식으로 샘플링하면서, 매 step의 예측 결과(x0)를 `process_latent_out`까지 적용해 `.pt`로 저장합니다. `LoadX0Step`은 저장한 x0를 LATENT로 다시 읽어 `VAEDecode`·`VAEDecodeAudio`로 디코딩할 수 있게 합니다. H3 latent는 영상과 소리가 묶인 NestedTensor라 두 텐서를 함께 저장합니다.

## 측정 설정

- 시간: 모델이 이미 올라간 상태에서 `/prompt` 제출부터 MP4 저장 확인까지(`runs.csv`의 `seconds_request_to_saved`). 첫 실행에는 로딩이 포함될 수 있습니다.
- seed: 42001 고정. 다양성 비교만 42001–42004
- 판정: 결과를 나란히 놓고 작성자가 직접 판정. 소리는 이름을 가리고 순서를 섞어 들음. 정량 지표(`results/metrics.json`)는 336×192 그레이스케일 기준의 보조 지표입니다.

## 결과 (2026-10-17, 각 조건 1회)

### 시간

| 조건 | 시간 |
| --- | --- |
| 기본 20 step | 192–202초 |
| 기본 4 step(LoRA 없음) | 59초 |
| Turbo 4 step | 46–56초 |
| Turbo 2 step / 8 step | 29초 / 83초 |

### step 수 (shift 12)

| LoRA | 실행 step | 판정 |
| --- | --- | --- |
| 4-step | 2 | 흐려짐 |
| 4-step | 4 | 입 모양 어색, 얼굴 변형 |
| 4-step | 8 | 얼굴 변형 |
| 8-step | 4 | 흐려짐 |
| 8-step | 8 | 괜찮음 |

### shift·샘플러·강도 (4-step LoRA, 4 step)

| 조건 | 화면 | 소리(잡음 / 자연스러움, 5점) |
| --- | --- | --- |
| shift 12, res_multistep, 1.0 | 입 모양 어색, 얼굴 변형 | 2 / 3 |
| shift 6, res_multistep, 1.0 | 입 모양 어색 | – |
| shift 6, euler, 1.0 | 소리 잡음, 목소리 어색 | – |
| shift 6, er_sde, 1.0 | 입 모양 어색 | – |
| shift 6, lcm, 1.0 | 목소리 어색 | 4 / 4 |
| **shift 6, er_sde, 0.75** | **괜찮음** | – |
| shift 12, lcm, 1.0 | 목소리 어색 | – |
| v1.2, shift 6, lcm / er_sde | – | 4 / 4, 4 / 4 |
| (참고) 기본 20 step | 괜찮음 | 5 / 5 |

### 그 밖의 관찰

- step별 x0: 기본 모델은 첫 step이 흐린 평균 얼굴이고 6–10 step에서 형태가 잡힘. Turbo는 첫 step부터 눈·코·입 위치가 잡혀 있음
- seed 4개 비교(shift 12): seed 간 평균 픽셀 차이 기본 10.3 / Turbo 9.3, 프레임 간 움직임 기본 0.92–1.19 / Turbo 1.31–1.61. shift 6에서도 Turbo 움직임 1.46–1.67
- LoRA 헤더(`results/lora-headers.json`): rank 128, 블록 50개의 어텐션 qkv·출력과 MLP, 약 0.98B. v1.0 alpha 128, v1.2 alpha 8

## 해석 범위

- 장면 하나(대사 한 줄을 말하는 가상 인물), 조건마다 1회입니다. 다른 장면·seed에서 순위가 바뀔 수 있습니다.
- 판정은 작성자 한 명이 했습니다. 재실험의 영상 판정은 이름이 보이는 상태였고, 소리 판정만 이름을 가렸습니다.
- 소리 점수가 shift 12 → 6에서 좋아진 원인은 확인하지 않았습니다(오디오 shift는 두 조건 모두 3).
- H3 Turbo LoRA의 학습 방법은 공개되지 않았습니다. LightX2V가 Wan용으로 공개한 DMD 방식은 해석에만 썼습니다.
- 모델 가중치는 pruned INT8, 텍스트 인코더는 NVFP4입니다(bf16 원본이 아님).

## 검증

```bash
python3 -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('scripts/*.py')+['custom_nodes/x0capture/__init__.py']]"
python3 -c "import json,glob;[json.load(open(f)) for f in glob.glob('workflows/*.json')+glob.glob('inputs/*.json')+glob.glob('results/*.json')]"
```

관련 글: [H3 Turbo LoRA는 어떻게 20 step을 4 step으로 줄일까?](https://runpod-korea-ug-community.static.hf.space/articles/h3-turbo-lora/index.html)
