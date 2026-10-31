# 입력 파일

이 폴더에 아래 두 파일을 직접 넣습니다. 레시피에는 입력 파일을 포함하지 않았습니다.

| 파일 | 규격 |
| --- | --- |
| `source.mp4` | 1344×768(가로) 또는 768×1344(세로), 24 FPS, 124프레임(5.167초), 오디오 트랙 포함 |
| `identity.png` | 바꿔 넣을 인물의 얼굴·머리가 잘 보이는 사진 |

원본 영상을 규격에 맞추는 예시입니다.

```bash
ffmpeg -i original.mp4 -t 5.167 \
  -vf "fps=24,scale=1344:768:force_original_aspect_ratio=increase,crop=1344:768,setsar=1" \
  -af "aresample=32000" -ac 2 -c:v libx264 -crf 16 -c:a aac -b:a 160k \
  -frames:v 124 source.mp4
```

- 사용 권한이 있는 영상만 쓰세요.
- 인물 사진은 실존 인물이 아닌 사진이나 동의를 받은 사진만 쓰세요.
- 다른 폴더의 파일을 쓰려면 `BENCH_INPUTS=/path/to/dir`를 지정합니다.
- 2026-10-19 측정에는 공개 인터뷰 영상을 잘라 확대한 클립과 가상 인물 사진을 썼습니다.
