#!/usr/bin/env bash
set -e

URLS=(
  "https://example.com/videos/P01_01.mp4"
  "https://example.com/videos/P01_02.mp4"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_DIR="$SCRIPT_DIR/full"
CROPPED_DIR="$SCRIPT_DIR/cropped"
EMBEDDINGS_DIR="$SCRIPT_DIR/../embeddings"

mkdir -p "$FULL_DIR" "$CROPPED_DIR" "$EMBEDDINGS_DIR"

for url in "${URLS[@]}"; do
    id="${url##*/}"
    id="${id%.mp4}"

    echo "=== Processing $id ==="

    echo "Downloading..."
    wget -q -O "$FULL_DIR/${id}.mp4" "$url"

    echo "Cropping..."
    bash "$SCRIPT_DIR/resize_video.sh" "$FULL_DIR/${id}.mp4" "$CROPPED_DIR/${id}.mp4"

    echo "Embedding..."
    python "$SCRIPT_DIR/../jepa2/obtain_embeddings.py" \
        "$CROPPED_DIR/${id}.mp4" \
        "$EMBEDDINGS_DIR/${id}.pkl"

    echo "Done: $id"
done
