#!/usr/bin/env bash
# 원본 출력을 비교용 규격(1280x720, 24fps, 120프레임, 5초, AAC)으로 맞춥니다.
# 사용: scripts/normalize.sh <native.mp4> <final.mp4>
set -euo pipefail
ffmpeg -hide_banner -loglevel error -y -i "$1" \
  -vf "scale=1280:720:flags=lanczos,setsar=1" -frames:v 120 -r 24 -t 5 \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 192k "$2"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames,r_frame_rate -of csv=p=0 "$2"
