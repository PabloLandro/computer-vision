#!/usr/bin/env bash
set -e

BASE_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_DIR="$SCRIPT_DIR/full"
CROPPED_DIR="$SCRIPT_DIR/cropped"
EMBEDDINGS_DIR="$SCRIPT_DIR/../embeddings"
VIDEO_IDS_FILE="$SCRIPT_DIR/video_ids.txt"

mkdir -p "$FULL_DIR" "$CROPPED_DIR" "$EMBEDDINGS_DIR"

exec 3< "$VIDEO_IDS_FILE"

while IFS= read -r id <&3 || [[ -n "$id" ]]; do
    [[ -z "$id" ]] && continue

    prefix="${id%%_*}"
    url="${BASE_URL}/${prefix}/${id}.MP4"

    echo "=== Processing $id ==="

    echo "Downloading from $url"
    wget --progress=bar:force:noscroll -O "$FULL_DIR/${id}.mp4" "$url" </dev/null

    echo "Cropping..."
    bash "$SCRIPT_DIR/resize_video.sh" "$FULL_DIR/${id}.mp4" "$CROPPED_DIR/${id}.mp4" </dev/null

    echo "Embedding..."
    python "$SCRIPT_DIR/../jepa2/obtain_embeddings.py" \
        "$CROPPED_DIR/${id}.mp4" \
        "$EMBEDDINGS_DIR/${id}.pkl" </dev/null

    echo "Done: $id"
done

exec 3<&-
