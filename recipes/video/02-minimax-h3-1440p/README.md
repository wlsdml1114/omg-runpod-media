# 02. MiniMax H3를 RTX PRO 6000에서 1440p까지 생성하기

- **상태:** 기존 실측 결과로 검증됨
- **검증일:** 2026-08-19
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition × 1
- **모델:** MiniMax H3 T2V + 공식 Turbo 8-step LoRA
- **출력:** 24 FPS, 124프레임, H.264 + AAC

동일한 5초 T2V 설정을 768×448, 1920×1088, 2560×1440으로 실행하여 고해상도 생성 가능 여부와 시간·VRAM을 확인합니다. 1440p는 성공 여부를 보는 1회 탐색 실행이지, 반복 벤치마크가 아닙니다.

## 라이선스 먼저 확인

MiniMax H3 공식 라이선스는 대한민국을 제외 지역으로 명시합니다. 한국에서 사용하거나 배포할 경우 공식 신청서를 제출하고 MiniMax의 별도 허가를 받은 뒤 진행하세요. 신청서 제출만으로 허가가 완료된 것은 아닙니다.

설치와 모델 배치는 [01 준실시간 레시피](../01-minimax-h3-realtime/)를 먼저 따릅니다. 이 폴더의 워크플로는 공식 8-step Turbo LoRA를 사용하므로 `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`가 필요합니다.

## 1회 실행

ComfyUI를 `--use-sage-attention` 옵션으로 실행한 뒤 레시피 폴더에서 다음처럼 호출합니다.

```bash
python -m pip install websocket-client

python scripts/run_once.py \
  --server 127.0.0.1:7860 \
  --base workflows/minimax_h3_turbo8_api.json \
  --prompts inputs/prompts.json \
  --prompt-id P01 \
  --seed 424271 \
  --width 2560 \
  --height 1440 \
  --duration 5 \
  --fps 24 \
  --filename-prefix h3-1440p/run-01 \
  --output benchmark-output/h3-2560x1440.mp4
```

스크립트는 ComfyUI `execution_start`·`execution_success`, API 히스토리와 실제 MP4를 함께 보존합니다. 768×448 warm 측정은 시드를 바꿔 3회 실행했고, 고해상도 두 셀은 각 1회만 실행했습니다.

## 기존 실측 결과

| 해상도 | 표본 | 생성 시간 | 샘플링 VRAM 피크 |
| --- | ---: | ---: | ---: |
| 768×448 | warm 3회 | 중앙값 18.52초 | 41.0 GiB |
| 1920×1088 | 1회 | 176.51초 | 50.3 GiB |
| 2560×1440 | 1회 | 444.53초 | 59.2 GiB |

시간은 출력 파일 확인까지 포함한 래퍼 wall time입니다. 초기 모델 로딩은 warm 집계에서 제외했습니다. 1920×1088과 2560×1440는 반복 측정이 아니므로 성능 순위나 보장값으로 사용하지 마세요.

## 정적 검사

```bash
python -m unittest discover -s tests -v
```

GPU 없이도 Turbo 8-step, LoRA, 주요 노드와 실행 행렬의 해상도를 검사합니다.
