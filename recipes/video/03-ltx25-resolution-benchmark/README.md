# 03. LTX 2.5 해상도별 생성 시간 측정

- **상태:** 기존 실측 결과로 검증됨
- **검증일:** 2026-08-18
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition × 1
- **모델:** LTX 2.5 22B distilled, Comfy INT8 ConvRot
- **출력:** 5초 요청, 24 FPS, 121프레임, H.264 + AAC

같은 프롬프트와 스텝을 유지하고 832×448, 1280×704, 1984×1088을 각 5회씩 실행합니다. 집계용 실행은 매번 시드를 바꿔 ComfyUI 노드 캐시가 샘플링 결과를 재사용하지 않게 합니다.

## 모델 배치

ComfyUI를 `/workspace/ComfyUI`에 설치했다고 가정합니다. 모델의 라이선스와 Hugging Face 접근 조건을 확인한 뒤 실행하세요.

```bash
COMFY_ROOT=/workspace/ComfyUI ./scripts/download_models.sh
```

스크립트는 LTX-2.5 모델·VAE·업스케일러와 Gemma 4 텍스트 인코더를 ComfyUI 표준 모델 디렉터리에 받습니다. 가중치는 저장소에 포함하지 않습니다.

## 실행

```bash
python -m pip install -r requirements.txt

python scripts/run_once.py \
  --server 127.0.0.1:7860 \
  --base workflows/ltx25_t2v_api.json \
  --prompts inputs/prompts.json \
  --prompt-id P01 \
  --seed 424243 \
  --width 832 \
  --height 448 \
  --duration 5 \
  --fps 24 \
  --prompt-enhance \
  --filename-prefix ltx25-resolution/run-01 \
  --output benchmark-output/832x448-run-01.mp4
```

`inputs/run-matrix.csv`의 `warm`행을 순서대로 돌리면 기존 측정 행렬과 같은 해상도·시드를 쓸 수 있습니다. 속도를 비교할 때는 최초 로딩과 해상도별 warm-up을 집계에서 제외하세요.

## 기존 실측 결과

| 해상도 | warm 반복 | 중앙값 | 범위 | 최대 sampled VRAM |
| --- | ---: | ---: | ---: | ---: |
| 832×448 | 5회 | 16.39초 | 14.91–16.53초 | 50.11 GiB |
| 1280×704 | 5회 | 25.65초 | 24.85–26.45초 | 50.61 GiB |
| 1984×1088 | 5회 | 54.87초 | 54.33–58.80초 | 50.86 GiB |

VRAM은 100ms 간격 `nvidia-smi` 장치 샘플입니다. PyTorch allocator peak가 아니며 짧은 피크를 놓칠 수 있습니다. 이 결과로 48GB GPU에서의 성공·실패를 단정하지 마세요.

## 정적 검사

```bash
python -m unittest discover -s tests -v
```
