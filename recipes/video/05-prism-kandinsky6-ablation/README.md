# 05. Prism·Kandinsky 6 속도 기법을 하나씩 켜고 끄기

- **상태:** 기존 실측 결과로 검증됨. 레시피 스크립트는 실험 스크립트에서 옮겨 정리한 것이며, **이 형태 그대로 새 Pod에서 처음부터 다시 실행해 보지는 않았습니다.**
- **검증일:** 2026-10-13 (KST)
- **GPU:** NVIDIA H100 80GB HBM3 (H100 SXM) × 1, Runpod Secure Cloud
- **모델:** Prism Alpha (`preview_alpha`), Kandinsky 6.0 Pro-distill 5s + 공식 VSR ×2
- **출력:** I2V, 5초, 24 FPS, 121프레임 생성 → 1280×720·120프레임으로 정규화, 소리 포함

고해상도 영상 생성 비용을 줄이는 세 가지 기법을 한 번에 하나씩만 바꿔 생성 시간과 화면 변화를 봅니다.

1. **Prism 블록 희소 어텐션**: 동적 블록 모양(IVPQ)을 켜고 끄기, sparsity 0.75 → 0.93
2. **Kandinsky 6 저해상도 생성 + SR**: 768×432로 만들고 공식 VSR로 ×2
3. **Kandinsky 6 π-Flow 증류**: Pro(50스텝 × CFG 2패스 = forward 100회) 대신 Pro-distill(10스텝, CFG 없음 = forward 10회)

같은 입력 사진, 같은 프롬프트, 같은 시드(42001)를 쓰고 설정마다 **1회만** 실행했습니다. 품질 우열을 판정하는 벤치마크가 아닙니다.

## 측정 결과

H100 80GB, 720p 5초 I2V, seed 42001, 설정별 1회. 모델 로딩 시간은 뺐습니다.

| ID | 바꾼 것 | 디노이즈 | 스텝당 | 클립당 합계 | 최대 VRAM (torch 할당) |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | Prism 고정 블록 4×4×4, sparsity 0.75 (기준) | 1940초 | 38.8초 | 2112초 | 65.5GB |
| A2 | A0 + 동적 블록 켬 | 1984초 | 39.7초 | 2122초 | 65.5GB |
| A2s | A2와 같은 설정, 다른 장면(scarf) | 1904초 | 38.1초 | 1978초 | 65.5GB |
| A4 | A2 + sparsity 0.93 | **1260초** | **25.2초** | 1324초 | 65.5GB |
| C1 | K6 Pro-distill 768×432, 10스텝, VSR ×2 (cup) | 71초 | 7.1초 | 193초 | 78.1GB |
| C1s | C1과 같은 설정, scarf | 70초 | 7.0초 | 141초 | 78.1GB |
| C2 | C1 + 짧은 프롬프트 | 70초 | 7.0초 | 148초 | 78.1GB |
| C3 | C2 + 프롬프트 보강기 qwen25 | 70초 | 7.0초 | 176초 | 78.1GB |

- 디노이즈 구간: 첫 트랜스포머 forward 시작 → 마지막 forward 반환.
- Prism 클립당 합계: 텍스트·VAE 인코드, 트랜스포머 GPU 이동, 디노이즈, 디코드, 저장. A2 시간에는 블록 모양 기록 훅 비용이 조금 들어 있습니다.
- K6 클립당 합계: `pipe()` 호출 시간 + VSR 시간. C3에는 보강기의 문장 생성 약 26초가 들어 있습니다.
- VSR에는 클립과 별도로 고정 비용이 들었습니다. VSR 로드 194초, 커널 워밍업·VAE 컴파일 1065초입니다.
- Prism은 H100에서 MoE 영상 전문가를 하나씩 교체하며 돌렸습니다(아래 참고). 교체 1회에 15~25초, 클립당 2회입니다. 같은 설정의 이전 H200 실행(교체 없음)은 약 34.9초/스텝이었습니다.
- **K6 Pro 50스텝(K0)과 Pro 720p 직접 생성(B1)은 이번 H100 세션에서 예산 때문에 실행하지 않았습니다.** 증류 전후를 화면으로 비교할 때는 2026-10-11 H200에서 같은 설정으로 만든 Pro 출력을 썼습니다. Pro의 H100 생성 시간은 이 레시피에 없습니다.

원본 숫자는 [`evidence/results-summary.json`](evidence/results-summary.json)에 있습니다.

### 동적 블록 모양 기록 (A2, cup)

[`scripts/prism_shape_hook.py`](scripts/prism_shape_hook.py)로 동적 블록 어텐션 호출 4000번(50스텝 × CFG 2 × 40층)을 모두 기록했습니다. 요약은 [`evidence/A2-block-shape-summary.json`](evidence/A2-block-shape-summary.json)입니다.

| 항목 | 값 |
| --- | --- |
| 토큰 격자 (패딩 포함, T×H×W) | 32×48×80 ≈ 12.3만 토큰 |
| 영역(macro-zone) | 4×6×10 = 240개, 영역당 8×8×8 토큰 |
| 전체 호출에서 기본 4×4×4 비율 | 62% |
| 기본 4×4×4 비율, 첫 스텝 → 마지막 스텝 (조건부 분기) | 75.3% → 43.9% |
| 128토큰 모양 | 거의 선택되지 않음 (0.01%) |
| 쿼리 블록당 선택된 키 블록 | 평균 480 / 1920 (sparsity 0.75의 top-k 상한과 같음) |

## 기본값 함정 (고정 커밋 `8f0c263` 기준)

| 항목 | 문서 | 실제로 실행되는 값 |
| --- | --- | --- |
| 동적 블록 `ENABLE_IVPQ_DYNAMIC_BLOCK` | README 표 기본값 `true`, 인자 설명에 "Prism core" | `prism_infer.sh` `"false"`, CLI `--enable_ivpq_dynamic_block` 기본 `false` |
| `bsa_sparsity` | README 설명 "each query block attends to only 25% of key blocks" (0.75 기준) | CLI 기본 `0.9375`, `prism_infer.sh`는 `0.75` |
| `negative_prompt` | 빈 문자열을 넘기면 꺼질 것 같지만 | CLI가 `''`를 기본 부정 프롬프트로 바꿈. 이 레시피는 래퍼에서 빈 값을 강제 |

동적 블록을 쓰려면 `--enable_ivpq_dynamic_block true`를 직접 넘겨야 합니다. 스크립트를 쓰지 않고 CLI만 호출하면 sparsity가 0.9375로 바뀌는 점도 주의하세요.

Kandinsky 6 Pro-distill 체크포인트 설정(`pro-distill.yaml`)의 기본 해상도는 864×480입니다. 이 레시피는 Pro와 같은 768×432를 명시합니다.

## 환경

| 항목 | 값 |
| --- | --- |
| Pod | H100 80GB SXM × 1, Secure Cloud, 1개 |
| 컨테이너 이미지 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| 디스크 | Volume `/workspace` 450GB, Container 30GB |
| 드라이버 | 580.126.09 |
| Prism 런타임 | Python 3.11.13, PyTorch 2.7.1+cu128, flash-attn 2.8.3 |
| K6 런타임 | Python 3.13.8, PyTorch 2.14.1+cu130 (`uv.lock` 고정), FlashAttention 3 (`hopper` extra) |

실험은 Pro 가중치까지 받아 450GB를 썼습니다. 이 레시피 범위(Prism + Pro-distill + VSR)의 가중치는 약 241GB입니다. 가상환경과 캐시를 더하면 최소 350GB 정도를 권합니다(실측 아님, 계산값).

이번 Pod의 `/workspace`는 네트워크 파일시스템이었습니다. import와 가중치 로드가 느려서 Prism 프로세스 하나를 띄우는 데 약 16~20분, K6 배치 로드에 약 15분, 패키지 설치에 약 40분이 걸렸습니다. 다른 Pod에서는 크게 달라질 수 있습니다.

### 고정한 소스와 가중치

| 대상 | 커밋 / revision |
| --- | --- |
| [Tencent-Hunyuan/Prism](https://github.com/Tencent-Hunyuan/Prism) | `8f0c2633f40ccf89e26f928c6d1771616caeb510` |
| [kandinskylab/kandinsky-6](https://github.com/kandinskylab/kandinsky-6) | `f19c9b8ec2fcea5d391f5391983a540fc5e004da` |
| kandinsky-6-sr (K6 `uv.lock`으로 설치) | `607f775d9859c026b9138966a5ecd303f60d205d` |
| HF `FrancisRing/Prism` | `0ff9e33dc323a9a97ca3defcc019d745e93f3a26` |
| HF `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` | `80d5e6c7f646c5f512b684e2da0765618d805894` |
| HF `kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers` | `f58a2158fdec454c5265088543e2b0eae5c1be3c` |
| HF `kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` (선택, 비교 기준용) | `295ca976810e318fb45c870ba4fc72c9a37fd548` |

받을 파일 목록과 SHA-256은 [`configs/model-manifest.json`](configs/model-manifest.json)에 있습니다.

## 라이선스와 입력

- Prism 코드·가중치: MIT. 포함된 서드파티 구성 요소(MOVA 등)는 각 원 라이선스(Apache-2.0 등)를 따릅니다. 저장소의 `LICENSE`를 확인하세요.
- Kandinsky 6 코드·가중치: MIT. Hugging Face 저장소는 연락처 공유에 동의해야 받을 수 있는 gated 저장소입니다.
- `inputs/`의 두 PNG는 AI로 생성한 가상의 성인 인물 사진입니다. 실존 인물 사진이나 저작물 캐릭터로 바꿔 쓸 때는 권리를 직접 확인하세요.
- 생성 결과물의 이용 조건은 각 모델 라이선스를 따릅니다.

## 1. Pod 만들기

Runpod 콘솔에서 H100 80GB SXM × 1, 위 컨테이너 이미지, Volume 450GB(최소 350GB 권장), Container 30GB로 Pod를 만듭니다. 이 레시피는 Pod를 자동으로 만들거나 지우지 않습니다. 가격은 만들기 직전에 콘솔에서 다시 확인하세요(실험 당시 $3.99/h).

Pod에 접속한 뒤 이 저장소를 받습니다.

```bash
cd /workspace
git clone https://github.com/wlsdml1114/omg-runpod-media.git
cd omg-runpod-media/recipes/video/05-prism-kandinsky6-ablation
source scripts/setup_pod.sh      # ffmpeg, git, uv 0.9.0, 캐시 경로 설정
```

모든 작업 파일은 `WORK_ROOT`(기본 `/workspace/ablation`) 아래에 생깁니다.

## 2. 런타임 설치

두 모델은 Python 버전이 달라서 가상환경을 따로 만듭니다. 두 명령은 병렬로 돌려도 됩니다.

```bash
python3 scripts/install.py --model prism
python3 scripts/install.py --model k6
```

Prism 설치에서 스크립트가 처리하는 우회 두 가지:

- `requirements.txt`의 `descript-audiotools>=0.7.4`는 PyPI에 없습니다(PyPI는 0.7.2까지). 0.7.4를 선언한 공식 소스 커밋 `348ebf2`로 바꿔 설치합니다.
- `flash-attn`은 소스 빌드 대신 v2.8.3 공식 릴리스 wheel(`cu12torch2.7cxx11abi{TRUE|FALSE}-cp311`)을 PyTorch C++ ABI에 맞춰 설치합니다.
- 나머지 의존성은 실험 때 고정한 [`configs/prism-constraints.txt`](configs/prism-constraints.txt)로 제한합니다.

## 3. 가중치 받기

Hugging Face 토큰은 환경 변수로만 넘기고, 셸 기록이나 파일에 남기지 마세요. Kandinsky 저장소는 먼저 웹에서 접근 동의를 해야 합니다.

```bash
read -rs HF_TOKEN && export HF_TOKEN
$WORK_ROOT/source/kandinsky-6/.venv/bin/python scripts/download_weights.py --model prism k6distill k6sr
unset HF_TOKEN
```

`--model k6pro`를 더하면 비교 기준용 K6 Pro(약 90GB)도 받습니다. 스크립트는 manifest의 SHA-256과 다르면 멈춥니다.

## 4. Prism 실행 (A0 / A2 / A2s / A4)

Prism 소스 디렉터리에서 Prism 가상환경으로 실행합니다.

```bash
cd $WORK_ROOT/source/Prism
PY=$WORK_ROOT/envs/prism/bin/python
R=/workspace/omg-runpod-media/recipes/video/05-prism-kandinsky6-ablation

$PY $R/scripts/prism_run.py --runs A0            # 고정 블록
$PY $R/scripts/prism_run.py --runs A2 A2s A4     # 가중치 한 번 로드 후 세 설정 연속
```

설정은 [`configs/runs.json`](configs/runs.json)에 있습니다. 세 설정이 달라지는 인자는 이것뿐입니다.

| ID | `--enable_ivpq_dynamic_block` | `--bsa_sparsity` |
| --- | --- | --- |
| A0 | `false` | `0.75` |
| A2, A2s | `true` | `0.75` |
| A4 | `true` | `0.93` |

공통 인자: `--height 720 --width 1280 --num_frames 121 --fps 24 --num_inference_steps 50 --cfg_scale 5 --visual_shift 7 --audio_shift 7 --seed 42001 --offload cpu --enable_bsa true --bsa_chunk_3d_shape_q 4 4 4 --bsa_chunk_3d_shape_k 4 4 4 --bsa_cdf_threshold 0.20`, 음성 프롬프트는 영상 프롬프트와 같게, negative는 빈 값.

`prism_run.py`는 업스트림 파일을 고치지 않고 메모리 안에서만 다음을 바꿉니다. 수치 계산은 바꾸지 않습니다.

- **MoE 전문가 교체** (`PRISM_EXPERT_SWAP=1`, 기본값): `video_dit`과 `video_dit_2`는 각각 28.6GB입니다. 지금 쓰는 전문가만 GPU에 두고 다른 하나는 CPU로 옮깁니다. 이전 H200 실행에서 둘을 함께 올렸을 때 상주 메모리가 약 88.5GiB였기 때문에, H100 80GB에서는 이 교체 없이 돌리지 않았습니다. H200 이상에서는 `PRISM_EXPERT_SWAP=0`으로 끌 수 있습니다.
- 빈 negative prompt 강제, 구간별 시간 측정, 가중치 재사용.
- `record_shapes`가 켜진 A2는 동적 블록 모양을 `dynamic-block-shapes.npz`로 저장합니다.

결과는 `$WORK_ROOT/runs/<run_id>/`에 `native.mp4`, `prism-observation.json`(시간·메모리), `request.json`으로 남습니다.

블록 모양 요약은 GPU 없이 만들 수 있습니다.

```bash
python3 scripts/summarize_block_shapes.py $WORK_ROOT/runs/abl-A2-prism-dyn-cup/dynamic-block-shapes.npz
```

## 5. Kandinsky 6 Pro-distill 실행 (C1 / C1s / C2 / C3)

```bash
cd $R
$WORK_ROOT/source/kandinsky-6/.venv/bin/python scripts/k6_run.py \
  --checkpoint pro-distill --runs C1 C1s C2 C3
```

스크립트가 만드는 설정 파일(`k6-config.yaml`)은 다음과 같습니다.

```yaml
checkpoint: pro-distill
generation: {height: 432, width: 768, sample_frames: 121, latent_frames: 31,
             num_steps: 10, guidance_weight: 1.0, visual_cond_scheme: tail_cond_first_frame}
cache: {mode: none}
compile: {strategy: none}
offload: {strategy: module}
attention: {engine: flash_attention_3}
beautifier: {name: none}            # C3만 실행 시 qwen25로 바꿈
text_embedder: {quantized_qwen: false}
sr: {checkpoint_path: $WORK_ROOT/weights/k6-sr, resolution_scale: 2, num_steps: 5, seed: 42001}
```

- 스케줄러는 체크포인트 기본값인 PiFlow(10개 grid point, 전체 궤적 128 substep)를 그대로 씁니다. `guidance_weight: 1.0`이라 무조건부 패스를 돌리지 않습니다.
- 모델 하나를 로드해 클립을 차례로 만든 뒤, 모델을 내리고 VSR을 로드해 ×2를 적용합니다. 결과는 `native.mp4`(768×432)와 `native_sr.mp4`(1536×864)입니다.
- module offload 상태에서도 최대 할당이 78.1GB였습니다. H100 80GB 한계에 가깝습니다.

K6 Pro 50스텝 기준(K0)을 직접 만들려면 `configs/runs.json`에 `checkpoint: pro`, `steps: 50`, `guidance: 5.0`인 항목을 추가하고 `--checkpoint pro`로 실행하세요. **이 조합은 이번 H100 세션에서 실행하지 않았습니다.**

## 6. 같은 규격으로 맞추기

```bash
mkdir -p $WORK_ROOT/outputs
for d in $WORK_ROOT/runs/abl-*; do
  src=$d/native_sr.mp4; [ -f "$src" ] || src=$d/native.mp4
  scripts/normalize.sh "$src" "$WORK_ROOT/outputs/$(basename $d).mp4"
done
```

1280×720, 24fps, 120프레임, 5초, lanczos, H.264 crf 18, AAC 192k로 맞춥니다.

## 비용

| 항목 | 값 |
| --- | --- |
| Pod 생성 → 삭제 | 4.27시간 |
| GPU | $3.99/h × 4.27h ≈ $17.02 |
| 디스크 (450GB + 30GB) | 약 $0.28 |
| 합계 추정 | **약 $17.3** |

설치·다운로드·네트워크 파일시스템 로드 시간을 모두 포함한 값입니다. 공급자 시간별 청구 스냅숏은 마지막 시간 버킷이 반영되기 전이었고, 최종 청구서는 확인하지 않았습니다. 생성만 보면 Prism 한 편(클립당 합계 약 35분)은 GPU 값 약 $2.3, K6 Pro-distill 한 편(2~3분, VSR 포함)은 약 $0.2입니다(시간 × 시간당 가격으로 계산한 값).

## 결과를 넓게 해석하면 안 되는 범위

- 설정·장면마다 시드 하나, 실행 한 번입니다. 화면 차이는 "이번 결과에서"의 관찰이며 성공률이 아닙니다.
- 720p에서 잰 값입니다. Prism 논문은 2K 학습·추론을 목표로 하며, 논문 스스로 720p에서는 이득이 작다고 보고합니다.
- Prism H100 실행은 MoE 전문가 교체를 포함합니다. 교체 없는 환경의 스텝당 시간과 직접 비교하지 마세요.
- 동적 블록 vs 고정 블록의 scarf 장면 비교에서 고정 블록 쪽은 이전 H200 출력입니다. 같은 설정의 H200 → H100 차이는 cup 장면에서 프레임 평균 절대차 1.85/255였습니다.
- K6 증류 전후 비교의 Pro 쪽은 이전 H200 출력입니다. Pro의 H100 시간은 측정하지 않았습니다.
- 네트워크 파일시스템 때문에 로드·컴파일이 실행 시간의 큰 부분을 차지했습니다. 로드 시간은 이 Pod에 묶인 값입니다.

## 정리

실험이 끝나면 Runpod 콘솔에서 **Pod를 삭제**하고, 목록에서 사라졌는지 확인하세요. Volume을 남기면 저장 요금이 계속 나옵니다. 가중치를 다시 쓸 계획이 없다면 Network Volume도 함께 지웁니다.

## 정적 검사

```bash
python -m unittest discover -s tests -v
```

테스트는 실행 설정이 한 번에 한 요인만 바꾸는지, 입력 해시, 가중치 revision, 블록 모양 요약 수치, 스크립트에 비밀정보가 없는지를 확인합니다. GPU 실행은 하지 않습니다.
