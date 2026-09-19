# 04. RTX PRO 6000 한 장에 ComfyUI 3개 띄우기

- **상태:** 기존 실측 결과로 검증됨
- **검증일:** 2026-08-29
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition × 1
- **워크로드:** LTX 2.5 I2V, 832×448, 121프레임, 24 FPS, 8-step
- **워커:** ComfyUI 0.34 API 프로세스 3개, 포트 3000–3002

하나의 96GB GPU에 ComfyUI 프로세스를 세 개 띄워 동시에 세 건을 보내는 실험입니다. 모델과 입력은 공유하고 `output`, `temp`, `user`만 워커별로 격리합니다.

결론은 단순합니다. 이 같은 대형 I2V 워크로드에서 3-worker는 직렬 실행보다 빠르지 않았습니다. 대신 서로 다른 큐나 사용자를 격리하는 구조로는 쓸 수 있습니다.

## 전제 조건

검증 스크립트는 다음 고정 환경을 확인합니다.

- ComfyUI `0.34.0`
- PyTorch `2.13.0+cu130`
- RTX PRO 6000 Blackwell
- ComfyUI 실행 파일: `/opt/ComfyUI/main.py`
- Python: `/venv/bin/python`
- 공유 모델·입력 루트: `/workspace/ComfyUI`

경로가 다르면 `COMFY_MAIN`, `COMFY_PYTHON`, `COMFY_SHARED_ROOT`를 설정합니다. 워크플로가 참조하는 입력 이미지를 먼저 복사합니다.

```bash
cp inputs/i2v-source-832x448.png /workspace/ComfyUI/input/
```

## 1. 워커 실행

안전한 기본값은 PyTorch cross attention입니다.

```bash
python scripts/launch_workers.py \
  --attention pytorch \
  --worker-count 3 \
  --base-port 3000
```

스크립트는 각 워커의 PID를 기록하고, 자신이 띄운 ComfyUI 프로세스인지 확인한 뒤에만 재시작합니다. 동시 cold load는 VRAM 피크를 만들 수 있으므로, 실제 측정 전에는 워커 0→1→2 순서로 warm-up하세요.

## 2. 동시 배치

```bash
python scripts/multiworker_benchmark.py \
  --base-url http://127.0.0.1:3000 \
  --base-url http://127.0.0.1:3001 \
  --base-url http://127.0.0.1:3002 \
  --workflow workflows/ltx25_i2v_api.json \
  --input inputs/i2v-source-832x448.png \
  --artifact-dir benchmark-output/pytorch-b01 \
  --label pytorch-b01 \
  --seed-base 610001
```

`--artifact-dir`은 존재하면 덮어쓰지 않고 실패합니다. 각 워커의 시작 시간, ComfyUI 실행 시간, 출력 해시, ffprobe, 전체 VRAM 샘플을 보존합니다.

## 기존 실측 결과

| 구성 | 결과 |
| --- | ---: |
| PyTorch 단일 warm | 12.020초/영상, 4.992개/분 |
| PyTorch 3-worker | 3개 배치 평균 36.774초, 합산 4.895개/분 |
| SageAttention 단일 warm | 12.634초/영상 |
| SageAttention 3-worker | 합산 4.988개/분 |

PyTorch 3-worker는 단일 직렬 대비 처리량이 약 1.94% 낮았습니다. VRAM 최저 여유가 9 MiB까지 내려간 배치가 있었으므로 워크로드를 키우기 전에 OOM 회복 정책을 별도로 검증하세요. SageAttention 빌드는 CUDA·PyTorch·SM120 조합에 묶이므로 이 저장소에 wheel을 포함하지 않습니다.

## 정적 검사

```bash
python -m unittest discover -s tests -v
```
