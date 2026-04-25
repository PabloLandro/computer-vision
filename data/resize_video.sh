#!/usr/bin/env bash

inp="in/P01_01.mp4"
out="out/P01_01.mp4"
dim=256

ffmpeg -i "$inp" \
  -vf "crop=min(iw\,ih):min(iw\,ih),scale=${dim}:${dim}" \
  -c:v libx264 \
  -crf 18 \
  -preset fast \
  "$out"