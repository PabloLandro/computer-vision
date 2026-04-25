#!/usr/bin/env bash
set -e

inp="$1"
out="$2"
dim="${3:-256}"

ffmpeg -i "$inp" \
  -vf "crop=min(iw\,ih):min(iw\,ih),scale=${dim}:${dim}" \
  -c:v libx264 \
  -crf 18 \
  -preset fast \
  "$out"
