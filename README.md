# OMG Runpod Media

OhMyGPU의 Runpod 기반 이미지·영상 생성 실습 자료입니다. 모델을 한 번 실행해 보는 데서 끝내지 않고, 사용한 GPU와 워크플로, 측정 방법, 결과의 한계를 함께 남겨 같은 조건에서 다시 확인할 수 있는 레시피로 관리합니다.

> 이 저장소는 OhMyGPU가 운영하는 커뮤니티 튜토리얼입니다. Runpod, 모델 개발사 또는 ComfyUI의 공식 제품 저장소가 아닙니다.

## 레시피

| 번호 | 유형 | 주제 | 모델·도구 | GPU | 검증일 | 확인된 결과 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 영상 | [5초 영상을 재생 시간에 가깝게 생성하기](recipes/video/01-minimax-h3-realtime/) | MiniMax H3 FL2VA · ComfyUI · NVENC | RTX PRO 6000 Blackwell × 1 | 2026-09-12 | 5.1667초 영상을 생성·저장 중앙값 5.492초 |

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
│   └── video/                 # 영상 생성·편집 레시피
│       └── 01-minimax-h3-realtime/
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
