# MiniMax H3 실시간 생성 재현 템플릿

[← 전체 이미지·영상 레시피](../../../README.md) · [LLM 배포 레시피](https://github.com/wlsdml1114/omg-runpod-llm)

RTX PRO 6000 한 장에서 MiniMax H3의 5초 영상을 **저장 완료까지 얼마나 빨리 끝낼 수 있는지** 확인할 수 있는 ComfyUI 템플릿입니다.

기본 설정은 608×352, 124프레임, 24 FPS, 4-step FL2VA입니다. ComfyUI 기본 `SaveVideo` 대신 함께 제공하는 `H3NVENCSaveVideo` 노드를 사용해 NVIDIA의 H.264 하드웨어 인코더로 결과를 저장합니다.

이 설정으로 진행한 RTX PRO 6000 Blackwell Server Edition 테스트에서는 5.17초 분량의 영상 한 편이 백엔드 기준 중앙값 5.492초에 완료됐습니다. 영상 길이보다 0.325초 더 걸린 결과라 엄밀한 실시간(RTF 1.0 이하)은 아니며, **준실시간에 가까운 재현 출발점**으로 보는 편이 맞습니다.

## 먼저 확인할 라이선스

대한민국에서 MiniMax H3을 실제로 배포하려면 [공식 신청서](https://platform.minimax.io/h3-license)를 제출하고 **MiniMax의 별도 허가를 받아야 합니다**. 신청서 제출만으로 허가된 것은 아닙니다.

저장소의 MIT 라이선스는 템플릿 코드에만 적용됩니다. MiniMax H3 모델, 가중치, 출력물 또는 배포 권리를 대신 허가하지 않습니다. 사용 전 [MiniMax H3 공식 라이선스](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)와 자신의 사용 조건을 직접 확인하세요.

## 들어 있는 파일

```text
.
├── custom_nodes/oh_my_gpu_h3_nvenc/   # FFmpeg h264_nvenc 출력 노드
├── workflows/
│   ├── minimax_h3_fl2va_4step_nvenc_ui.json
│   └── minimax_h3_fl2va_4step_nvenc_api.json
├── scripts/
│   ├── install.sh
│   ├── download_models.sh
│   ├── check_setup.py
│   └── run_benchmark.py
└── tests/test_template.py
```

모델 가중치와 입력 이미지는 포함하지 않습니다.

## 가장 짧은 실행 순서

준비물은 NVIDIA GPU와 드라이버, `h264_nvenc`를 지원하는 FFmpeg, 최신 MiniMax H3 노드가 포함된 ComfyUI, Python 3입니다. 아래 예시에서는 ComfyUI가 `/workspace/ComfyUI`에 있다고 가정합니다.

### 1. NVENC 출력 노드 설치

```bash
git clone https://github.com/wlsdml1114/omg-runpod-media.git
cd omg-runpod-media/recipes/video/01-minimax-h3-realtime
./scripts/install.sh /workspace/ComfyUI
```

설치 후 ComfyUI를 다시 시작합니다.

### 2. MiniMax 허가 후 모델 다운로드

MiniMax로부터 별도 허가를 받은 경우에만 실행합니다. 아래 환경 변수는 승인을 대신 검증하는 장치가 아니라, 사용자가 승인 여부를 다시 확인하도록 둔 안전장치입니다.

```bash
pip install -U huggingface_hub
MINIMAX_H3_LICENSE_APPROVED=1 ./scripts/download_models.sh /workspace/ComfyUI
```

다운로드되는 파일은 다음과 같습니다.

- `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`
- `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`

### 3. 설치 상태 확인

```bash
python scripts/check_setup.py --comfyui /workspace/ComfyUI
```

모델 없이 저장소 구조만 검사하려면 다음 명령을 사용합니다.

```bash
python scripts/check_setup.py --static
```

### 4. ComfyUI에서 한 번 실행

ComfyUI 화면에 `workflows/minimax_h3_fl2va_4step_nvenc_ui.json`을 불러옵니다. 입력 이미지를 올리고 프롬프트를 바꾼 뒤 Queue 버튼을 누르면 됩니다.

기본값은 다음과 같습니다.

- 608×352
- 5초, 24 FPS
- MiniMax H3 FL2VA
- Turbo LoRA 4-step
- H.264 NVENC, preset `p1`, constant QP 18

출력 MP4 옆에는 변환과 인코딩 시간을 나눠 적은 `.timing.json` 파일도 생성됩니다.

## 여러 번 돌려 중앙값 구하기

ComfyUI를 `--listen 0.0.0.0` 등으로 실행한 뒤 입력 이미지를 준비합니다.

```bash
python -m pip install -r requirements.txt
python scripts/run_benchmark.py \
  --server http://127.0.0.1:8188 \
  --image /workspace/input.png \
  --runs 5 \
  --prompt "A cinematic close-up with gentle camera movement and natural ambient sound."
```

각 실행의 MP4와 해시, FFprobe 메타데이터, 백엔드 완료 시간은 `benchmark-output/benchmark-summary.json`에 저장됩니다. 첫 실행에는 모델 로딩과 컴파일이 섞일 수 있으므로, 성능을 비교할 때는 워밍업 후 같은 조건으로 여러 번 측정하세요.

## 수치가 다르게 나올 때 볼 곳

GPU가 같아도 ComfyUI·PyTorch·CUDA 버전, 드라이버, 모델 상주 여부, 입력 크기, 프레임 수, 다른 프로세스의 GPU 점유율에 따라 결과가 달라집니다. 특히 이 템플릿의 5.492초는 저해상도 4-step 설정에서 얻은 값입니다. 해상도나 스텝 수를 올리면 화질과 지연시간이 함께 달라집니다.

NVENC는 생성 모델 자체를 빠르게 만드는 장치가 아닙니다. 샘플링과 VAE 디코딩이 끝난 뒤, 프레임을 MP4로 저장하는 구간을 줄이는 최적화입니다. 같은 실험에서 기본 저장 경로의 전체 완료 시간은 중앙값 7.799초였고, NVENC 경로는 5.492초였습니다. 다만 QP 18 결과 파일은 기본 H.264 출력보다 중앙값 약 3.46배 컸으므로, 실제 서비스에서는 속도뿐 아니라 파일 크기와 전송 시간도 함께 조정해야 합니다.

## 알려진 범위

- 제공 워크플로는 FL2VA 4-step 재현용입니다. T2V, Ref2V, V2V를 모두 검증했다는 뜻은 아닙니다.
- RTX PRO 6000 Blackwell Server Edition 한 장에서 확인한 결과이며, 다른 GPU의 동일 성능을 보장하지 않습니다.
- QP 18과 ComfyUI 기본 저장은 같은 비트레이트·파일 크기로 맞춘 화질 비교가 아닙니다.
- 이 레시피는 MiniMax H3 또는 ComfyUI의 공식 배포물이 아닙니다.

## 코드 라이선스

템플릿 코드와 스크립트는 [MIT License](LICENSE)로 제공합니다. 모델 및 제3자 구성요소에는 각각의 라이선스가 적용됩니다.
