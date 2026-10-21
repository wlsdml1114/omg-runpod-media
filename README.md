# OMG Runpod Media

OhMyGPU의 Runpod 기반 이미지·영상 생성 실습 자료입니다. 모델을 한 번 실행해 보는 데서 끝내지 않고, 사용한 GPU와 워크플로, 측정 방법, 결과의 한계를 함께 남겨 같은 조건에서 다시 확인할 수 있는 레시피로 관리합니다.

> 이 저장소는 OhMyGPU가 운영하는 커뮤니티 튜토리얼입니다. Runpod, 모델 개발사 또는 ComfyUI의 공식 제품 저장소가 아닙니다.

## 레시피

| 번호 | 유형 | 주제 | 모델·도구 | GPU | 검증일 | 확인된 결과 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 영상 | [5초 영상을 재생 시간에 가깝게 생성하기](recipes/video/01-minimax-h3-realtime/) | MiniMax H3 FL2VA · ComfyUI · NVENC | RTX PRO 6000 Blackwell × 1 | 2026-09-12 | 5.1667초 영상을 생성·저장 중앙값 5.492초 |
| 02 | 영상 | [MiniMax H3를 1440p까지 생성하기](recipes/video/02-minimax-h3-1440p/) | MiniMax H3 T2V · Turbo 8-step | RTX PRO 6000 Blackwell × 1 | 2026-08-19 | 2560×1440 1회 생성 성공, 444.53초 |
| 03 | 영상 | [LTX 2.5 해상도별 생성 시간 측정](recipes/video/03-ltx25-resolution-benchmark/) | LTX 2.5 T2V · ComfyUI | RTX PRO 6000 Blackwell × 1 | 2026-08-18 | 832×448~1984×1088, 각 5회 warm 중앙값 |
| 04 | 영상 | [RTX PRO 6000 한 장에 ComfyUI 3개 띄우기](recipes/video/04-ltx25-pro6000-three-worker/) | LTX 2.5 I2V · ComfyUI 3-worker | RTX PRO 6000 Blackwell × 1 | 2026-08-29 | 직렬 4.992개/분, 3-worker 4.895개/분 |
| 05 | 영상 | [Prism·Kandinsky 6 속도 기법을 하나씩 켜고 끄기](recipes/video/05-prism-kandinsky6-ablation/) | Prism Alpha · Kandinsky 6 Pro-distill + VSR | H100 80GB SXM × 1 | 2026-10-13 | Prism sparsity 0.93 디노이즈 −36%, K6 distill 디노이즈 약 70초 (720p 5초, 각 1회) |
| 06 | 영상 | [같은 H3 워크플로를 Comfy API와 Runpod Serverless에 올려 비교하기](recipes/video/06-comfy-api-vs-runpod-h3/) | MiniMax H3 Ref2VA Turbo4 · Comfy API · Runpod Serverless | RTX PRO 6000 Blackwell × 1 | 2026-10-19 | warm 중앙값 Comfy API 165.1초 vs Runpod 196.3초, cold 191–419초 (각 warm 5회·cold 3회) |
| 07 | 영상 | [H3 Turbo LoRA를 켜고 끄며 학습 조건(step·shift·샘플러) 확인하기](recipes/video/07-h3-turbo-lora-settings/) | MiniMax H3 FL2VA · Turbo LoRA (lightx2v) · ComfyUI | RTX PRO 6000 Blackwell × 1 | 2026-10-17 | 20 step 192초 → Turbo 4 step 47초. 768p LoRA는 shift 6/3 + er_sde + 강도 0.75에서만 판정 통과 (각 1회) |
| 01 | 이미지 | [FLUX 3 Klein으로 상품 사진 합성하고 같은 인물로 여러 장면 만들기](recipes/image/01-flux3-klein-product-character-edit/) | FLUX 3 Image Klein distilled · ComfyUI · 프롬프트 업샘플러 | L40S 48GB × 1 | 2026-10-19 | 참조 2장 → 4MP 상품 편집 50.8–52.5초, 인물 2MP 23.3–26.1초 (업샘플러 포함, 각 1회) |

새 레시피는 [레시피 작성 규칙](docs/recipe-format.md)에 맞춰 추가합니다. 모델 파일과 대용량 생성 결과는 저장소에 포함하지 않습니다.

## 바로 시작하기

```bash
git clone https://github.com/wlsdml1114/omg-runpod-media.git
cd omg-runpod-media/recipes/video/01-minimax-h3-realtime
```

각 레시피의 README에서 필요한 GPU, 모델 라이선스, 설치 순서와 검증 명령을 먼저 확인하세요. 예제는 유료 Runpod 리소스를 자동으로 만들거나 삭제하지 않습니다.

## 저장소 구조

```text
.
├── recipes/
│   ├── image/                 # 이미지 생성·편집 레시피
│   │   └── 01-flux3-klein-product-character-edit/
│   └── video/                 # 영상 생성·편집 레시피
│       ├── 01-minimax-h3-realtime/
│       ├── 02-minimax-h3-1440p/
│       ├── 03-ltx25-resolution-benchmark/
│       ├── 04-ltx25-pro6000-three-worker/
│       ├── 05-prism-kandinsky6-ablation/
│       ├── 06-comfy-api-vs-runpod-h3/
│       └── 07-h3-turbo-lora-settings/
├── docs/
│   └── recipe-format.md       # 새 레시피가 지켜야 할 공통 형식
├── .gitignore
└── LICENSE
```

레시피 하나는 가능하면 다음 자료를 함께 둡니다.

- 처음 실행하는 사람이 따라갈 수 있는 `README.md`
- UI/API 워크플로 또는 실행 설정
- 설치·다운로드·환경 확인 스크립트
- 반복 측정과 결과 검증 방법
- 모델·코드·출력물의 라이선스 주의사항
- 비밀정보와 대용량 파일을 제외하는 `.gitignore` 규칙

## OhMyGPU Runpod 저장소

| 저장소 | 다루는 내용 |
| --- | --- |
| **OMG Runpod Media** | 이미지·영상 생성, ComfyUI 워크플로, 생성 시간·VRAM·출력 검증 |
| [OMG Runpod LLM](https://github.com/wlsdml1114/omg-runpod-llm) | Pod·Serverless 기반 오픈소스 LLM 배포와 서빙 |

둘 다 특정 GPU의 결과를 모든 환경에 일반화하지 않습니다. GPU, 이미지, 드라이버, CUDA, 프레임 수, 모델 버전과 측정 구간을 함께 기록합니다.

## 시작하기 전에

- Runpod 계정과 사용 가능한 크레딧이 필요합니다.
- Runpod API 키, Hugging Face 토큰, SSH 키를 저장소·로그·스크린샷에 남기지 마세요.
- GPU 재고와 가격은 변동되므로 배포 전 Runpod 콘솔에서 다시 확인하세요.
- 모델별 라이선스와 지역 제한은 저장소의 코드 라이선스와 별개입니다.
- 테스트가 끝나면 Pod와 연결된 스토리지 상태를 직접 확인하세요.

## 라이선스

이 저장소의 예제 코드는 [MIT License](LICENSE)로 배포합니다. 모델 가중치, Runpod, ComfyUI, FFmpeg와 기타 의존성에는 각각의 라이선스와 이용 조건이 적용됩니다.
