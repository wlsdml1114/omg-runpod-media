# 01. Ming 3.0 TTS로 내 목소리를 한국어로 클론하기

- **상태:** 검증됨. 2026-10-22 실험(RTX 4000 Ada)의 측정값과, 2026-11-03 새 Pod(RTX 3090)에서 이 README 순서와 스크립트 그대로 처음부터 다시 돌린 결과가 함께 있어요([재검증](#재검증-2026-11-03-rtx-3090)).
- **검증일:** 2026-10-22, 2026-11-03 (UTC)
- **GPU:** NVIDIA RTX 4000 Ada 20GB × 1 (실험), RTX 3090 24GB × 1 (재검증), 둘 다 Runpod Secure Cloud
- **모델:** [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) (inclusionAI, MIT)
- **출력:** wav 44.1kHz mono

내 목소리 녹음 13초와 그 받아쓰기를 넣고, Ming 3.0 TTS 0.5B로 다른 한국어 문장을 읽게 만들어요. Ming은 **한국어 기본 목소리가 없어서**(기본 목소리와 텍스트로 목소리 만들기는 중국어·영어 전용) 한국어는 이렇게 참조 녹음을 따라 하는 방식으로만 쓸 수 있어요.

Endplan 영상 [한국어 보이스 클론, 무료 vs 유료 비교 | Ming 3.0 · ElevenLabs · Gemini](https://www.youtube.com/watch?v=sKSwRBqimgg)에서 ElevenLabs v4, Gemini 3.8 Flash TTS와 비교할 때 쓴 설정이에요.

## 결과 요약

같은 사람 목소리로, 평소 영상 말투로 쓴 23문장을 문장마다 3번씩(69개) 만들었어요.

| | Ming 3.0 (이 레시피) | ElevenLabs v4 | Gemini 3.8 Flash TTS |
| --- | --- | --- | --- |
| 참조 음성 | 13초 + 받아쓰기 | 2분 (Instant Voice Clone, 권장 1~2분) | 13초 (Ming과 같은 클립) + 본인 동의 음성 |
| 실행 | Runpod Pod, RTX 4000 Ada | API (Mac에서 호출) | API (Mac에서 호출) |
| 문장당 생성 시간 중앙값 | 5.9초 | 1.7초 | 6.1초 |
| 69개 비용 | Pod 약 $0.11 | 약 $0.29 (크레딧 1,617을 Creator 요금제로 환산) | 약 $0.09 |
| 한글만 있는 11문장 CER (Scribe, 1회차) | 0.41% | 0.00% | 0.41% |
| 목소리 닮음 (다른 사람 0 ~ 실제 본인 100) | **95** | 88 | 83 |
| 블라인드 2차 "가장 나 같은 것" (23문장) | **17** | 0 | 4 |
| 블라인드 1차 "제일 듣기 좋은 것" (23문장) | 6 | 6 | 11 |

- Ming이 목소리 결과 말투(속도, 담백한 끝맺음)를 가장 닮았어요. 듣기 좋고 발음과 톤이 안정적인 쪽은 ElevenLabs였어요.
- Ming이 흔들린 곳은 **숫자**(예: "VRAM은 48GB, 모델 크기는 0.5B예요")와 **영어 단어가 섞인 문장**이에요. 영어 단어 문장은 세 모델 모두 어색했어요.
- 감정 태그 `[Laughter]`, `[Sigh]`는 공식으로는 중국어·영어만 지원해요. 한국어 문장에서 쓰면 받아쓰기에 "하하", "하…"가 잡혀서, 웃음·한숨 소리가 나는 것으로 보여요.
- 같은 문장이라도 회차(seed)마다 결과가 달라요. 마음에 안 들면 seed를 바꿔 다시 만들면 돼요.

목소리 닮음은 클론에 쓰지 않은 **다른 날 녹음** 3개와의 WavLM-base-plus-sv 코사인 유사도예요. 블라인드 청취는 클론한 본인 1명이 1회차만 들은 결과예요. 개별 측정값과 비교 조건은 [`evidence/results-summary.json`](evidence/results-summary.json)에 있어요.

## 측정 결과 (Ming)

RTX 4000 Ada 20GB, Ming 기본 설정(문장을 조각으로 나누고 조각마다 참조 음성을 앞에 붙임), seed 1001~1003, 69개 실패 0.

| 항목 | 값 |
| --- | ---: |
| 모델 로딩 | 3.1초 |
| 문장당 생성 시간 중앙값 | 5.9초 (최소 3.0초, 최대 57.0초) |
| 생성 시간 ÷ 음성 길이 (RTF) 중앙값 | 1.46 |
| 약 38초짜리 긴 글(L01) | 55.3~57.0초 |
| 만든 음성 합계 | 6.3분 |
| Pod 켜 둔 시간 | 1,466초 (24분), 실험 당시 $0.28/h로 약 $0.11 |

생성 시간은 `synthesize()` 호출 앞뒤를 `torch.cuda.synchronize()`로 잰 값이에요. 모델 로딩, Pod 부팅, 설치 시간은 빠져 있어요. 첫 문장(11.3초)에는 워밍업이 섞여 있어요. 실험은 0.5B만 했고, Ming-3.0-tts-tiny(7.9B MoE)는 돌리지 않았어요.

## 재검증 (2026-11-03, RTX 3090)

새 Pod에서 저장소를 받는 단계부터 이 README의 3~6단계를 그대로 실행했어요.

| 항목 | 값 |
| --- | --- |
| Pod | RTX 3090 24GB × 1, Secure Cloud EU-CZ-1, 호스트 CUDA 13.2 · 드라이버 595.91.07, 이미지 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`(= `runpod-torch-v280` 템플릿 이미지), Container 디스크 40GB |
| `setup_pod.sh` | 성공. torch 2.6.0+cu124가 GPU 인식, `model.safetensors`·`campplus.onnx` SHA-256 일치, preflight 전 항목 ok |
| 설치된 주요 버전 | torch 2.6.0, torchaudio 2.6.0, transformers 4.46.3, diffusers 0.33.0, onnxruntime 1.31.1 |
| 한 문장(5단계) | 10.5초 → 음성 6.00초 (실험 때 같은 문장 1회차도 6.0초) |
| 69개(6단계) | 실패 0. 모델 로딩 2.6초, 문장당 생성 중앙값 6.1초(최소 3.0, 최대 55.6), RTF 중앙값 1.44, 음성 합계 6.3분 |
| 받아쓰기 확인 | 한글만 있는 11문장 1회차, Whisper large-v3(mlx) CER 1.22% — 실험 때 Whisper 1회차 값과 같음. 남은 오류는 "세 개"→"3개" 표기와 웃음 태그 "하하" |
| Pod 켜 둔 시간 | 약 15분(지우기 직전 uptime 906초), $0.50/h로 약 $0.13 |

GPU가 다르면 같은 seed라도 결과가 똑같이 나오지 않아요. 69개 중 38개만 음성 길이가 실험 때와 같았고, 나머지는 최대 1.1초 달랐어요. 생성 속도와 읽기 정확도는 거의 같았어요.

## 환경

| 항목 | 값 |
| --- | --- |
| Pod | RTX 4000 Ada 20GB × 1, Secure Cloud, 호스트 CUDA 13.2 |
| 템플릿 | `runpod-torch-v280` (Runpod Pytorch 2.8.0) |
| 디스크 | `/workspace` 30GB (모델 약 3GB + 가상환경) |
| 파이썬 환경 | 템플릿의 torch를 쓰지 않고 가상환경을 따로 만들어 Ming이 고정한 torch 2.6.0 + cu124 설치 |
| Ming 코드 | [`inclusionAI/Ming-3.0-tts`](https://github.com/inclusionAI/Ming-3.0-tts) `028c1eace193e7acdb256453da89b6e9207e052b` (2026-10-22 main) |
| 모델 | `inclusionAI/Ming-3.0-tts-0.5B` revision `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0` (2026-10-22 main) |

- 0.5B라 VRAM은 넉넉해요. 24GB 카드면 충분하고, 실험에서는 20GB로 돌렸어요.
- **Blackwell GPU(RTX PRO 4500/6000, RTX 5090 등)는 피하세요.** torch 2.6.0 휠이 Blackwell을 지원하지 않아요.
- Ming 코드는 실험 뒤에도 계속 바뀌고 있어요(2026-11-02 커밋에서 텍스트 처리 방식이 바뀜). 결과를 그대로 재현하려면 위 커밋과 리비전으로 고정하세요. `setup_pod.sh`가 고정해 줘요.

## 라이선스와 입력

- **코드·모델:** Ming-3.0-tts 저장소와 Hugging Face 모델 모두 MIT 라이선스예요. 사용 전 원문을 확인하세요.
- **참조 음성:** 이 레시피는 참조 녹음을 저장소에 넣지 않아요. **본인 목소리이거나 녹음 주인에게 허락을 받은 목소리만** 쓰세요. 다른 사람 목소리를 동의 없이 복제하면 퍼블리시티권·초상권 문제가 될 수 있어요.
- **문장 세트:** [`inputs/sentences.tsv`](inputs/sentences.tsv)는 영상에서 쓴 23문장(기본 3, 숫자 5, 영어 4, 약어 2, 발음 4, 구어 2, 감정 2, 긴 글 1)이에요. `[웃음]`, `[한숨]`은 스크립트가 Ming 문법 `[Laughter]`, `[Sigh]`로 바꿔 보내요.

## 1. 참조 녹음 준비 (내 컴퓨터)

[`inputs/README.md`](inputs/README.md)를 보고 10~15초짜리 내 목소리 wav와 그 받아쓰기 txt를 만들어요. 실험에서는 평소 영상 내레이션에서 13초를 잘라 썼어요.

## 2. Pod 만들기

Runpod 콘솔에서 아래처럼 Pod를 만들어요. 이 레시피는 Pod를 자동으로 만들거나 지우지 않아요. 가격과 재고는 만들기 직전에 콘솔에서 다시 확인하세요.

- GPU: RTX 4000 Ada, RTX 3090(둘 다 확인), RTX 4090, L4, A40처럼 24GB 안팎의 Ada·Ampere 카드 하나
- 템플릿: Runpod Pytorch 2.8.0 (`runpod-torch-v280`)
- `/workspace` 30GB 이상

```bash
cd /workspace
git clone https://github.com/wlsdml1114/omg-runpod-media.git
cd omg-runpod-media/recipes/audio/01-ming3-tts-korean-voice-clone
```

## 3. Ming 설치와 모델 받기

```bash
bash scripts/setup_pod.sh      # WORK_ROOT 기본값 /workspace
```

스크립트가 하는 일:

1. Ming 코드를 받아 위 커밋으로 고정
2. `/workspace/venv-ming` 가상환경에 Ming `requirements.txt` 설치. torch가 2.6.0 + CUDA로 잡히지 않으면 cu124 휠로 다시 설치
3. 모델을 고정 리비전으로 `/workspace/models/Ming-3.0-tts-0.5b`에 받고 `model.safetensors`, `campplus.onnx`의 SHA-256 확인
4. Ming의 `tools/preflight.py`로 체크포인트 확인

Hugging Face 토큰은 필요 없어요(공개 모델). 텍스트 정규화용 `pynini`는 설치하지 않아도 돼요. Ming은 중국어·영어가 아닌 언어에서는 정규화를 건너뛰어서, 한국어 숫자는 모델이 그대로 읽어요.

## 4. 참조 녹음 올리기

내 컴퓨터에서 만든 두 파일을 Pod의 레시피 폴더 `ref/`에 올려요. 예: `runpodctl send my_voice.wav my_voice.txt` → Pod에서 `runpodctl receive <코드>` 후 `ref/`로 옮기기. `ref/`, `out/`, `runs.jsonl`은 `.gitignore`에 들어 있어서 커밋되지 않아요.

## 5. 한 문장 만들어 보기

```bash
source /workspace/venv-ming/bin/activate
python scripts/clone_ko.py \
  --repo /workspace/Ming-3.0-tts --model-path /workspace/models/Ming-3.0-tts-0.5b \
  --ref-wav ref/my_voice.wav --ref-text-file ref/my_voice.txt \
  --text "근데 이번엔 목소리를 바꿔 볼 거예요. 제가 직접 녹음하는 대신, 모델 세 개한테 똑같은 대본을 읽혀 봤거든요."
```

`out/text_t1.wav`가 생겨요. 로그 첫 줄에 `sample_rate=44100`이 나오는지 확인하고 들어 보세요.

## 6. 영상과 같은 69개 만들기

```bash
python scripts/clone_ko.py \
  --repo /workspace/Ming-3.0-tts --model-path /workspace/models/Ming-3.0-tts-0.5b \
  --ref-wav ref/my_voice.wav --ref-text-file ref/my_voice.txt \
  --sentences inputs/sentences.tsv --takes 3
```

- 결과: `out/<문장id>_t<1~3>.wav`, 요청마다 시간·실패가 `runs.jsonl`에 한 줄씩 남아요.
- 이미 만든 wav는 건너뛰어요. 끊겼으면 같은 명령을 다시 실행하면 돼요. 다시 만들려면 `--force`를 붙이세요.
- 일부만: `--only S01 N03` (문장 id) 또는 `--only 숫자 영어` (그룹)
- seed는 1회차 1001, 2회차 1002, 3회차 1003이에요(영상과 같음). `--seed-base`로 바꿀 수 있어요.
- `--raw-text`를 붙이면 문장을 나누지 않고 한 번에 보내요. 영상에서는 쓰지 않았어요(기본 설정 기준).

결과 파일은 `runpodctl send out` 등으로 내 컴퓨터로 가져오세요. 다 쓰면 **Pod를 terminate하고** 콘솔에서 사라졌는지 확인하세요. 실험에서는 Pod를 24분 켜 두고 약 $0.11이 나왔어요.

## 해석할 때 주의

- 한 사람 목소리, 한 사람의 청취 판단이에요. 다른 목소리나 다른 청취자에서는 순위가 바뀔 수 있어요.
- 블라인드 2차의 ElevenLabs는 `[calm, low-pitched narration]` 지시를 붙인 버전이라, 1차와 비교할 때 기준(듣기 좋음 → 나 같음)과 버전이 함께 바뀌었어요.
- ElevenLabs는 참조 음성을 2분, Ming·Gemini는 13초 넣었어요. 각 서비스의 권장 조건을 따른 것이라 참조 길이가 같지 않아요.
- 속도는 Ming은 Pod 안에서, API 두 개는 Mac에서 재서 조건이 달라요. 음성 길이·GPU·문장 길이에 따라 달라져요.
- CER은 영어·숫자가 섞인 12문장을 빼고 계산했어요. 받아쓰기가 "ElevenLabs"를 "일레븐랩스"로 적는 표기 차이가 오류로 잡히기 때문이에요.

## 검증

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/setup_pod.sh
```

테스트는 GPU 없이 돌아가요. 파일 구성, 고정한 커밋·리비전·해시, 스크립트 인자, 문장 세트, 근거 파일의 숫자를 확인해요. 실제 음성 생성은 확인하지 않아요.
