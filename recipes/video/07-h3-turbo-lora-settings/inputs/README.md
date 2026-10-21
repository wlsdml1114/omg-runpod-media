# 입력

- `real-hd-01-face.png`: 첫 프레임 이미지(저장소에 넣지 않음). 실험에는 AI로 만든 가상 성인 인물의 정면 상반신 사진(1672×941)을 썼습니다. 사용 권한이 있는 이미지를 같은 이름으로 넣으세요. `MiniMaxH3ImageToVideo`가 1344×768로 맞춥니다.
- `model-download-manifest.json`: 내려받을 모델 파일, 저장소 revision, SHA-256
- `source-revisions.json`: ComfyUI 고정 커밋
- `run-matrix.json`: `scripts/prepare.py`, `scripts/prepare_r2.py`가 만드는 실행 목록(생성 파일)
