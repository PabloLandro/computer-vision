#!/usr/bin/env bash
set -euo pipefail

TRAIN_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"
TEST_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_DIR="$SCRIPT_DIR/full"
CROPPED_DIR="$SCRIPT_DIR/cropped"

TRAIN_EMBEDDINGS_DIR="$SCRIPT_DIR/../embeddings/train"
TEST_EMBEDDINGS_DIR="$SCRIPT_DIR/../embeddings/test"

TRAIN_IDS_FILE="$SCRIPT_DIR/train_ids.txt"
TEST_IDS_FILE="$SCRIPT_DIR/test_ids.txt"

mkdir -p "$FULL_DIR" "$CROPPED_DIR" "$TRAIN_EMBEDDINGS_DIR" "$TEST_EMBEDDINGS_DIR"

process_split () {
    local ids_file="$1"
    local base_url="$2"
    local embeddings_dir="$3"
    local split_name="$4"

    while IFS= read -r id || [[ -n "$id" ]]; do
        [[ -z "$id" ]] && continue

        prefix="${id%%_*}"
        url="${base_url}/${prefix}/${id}.MP4"

        echo "=== Processing $split_name/$id ==="

        echo "Downloading from $url"
        wget --progress=bar:force:noscroll -O "$FULL_DIR/${id}.mp4" "$url" </dev/null

        echo "Cropping..."
        bash "$SCRIPT_DIR/resize_video.sh" \
            "$FULL_DIR/${id}.mp4" \
            "$CROPPED_DIR/${id}.mp4" </dev/null

        echo "Embedding..."
        python "$SCRIPT_DIR/../jepa2/obtain_embeddings.py" \
            "$CROPPED_DIR/${id}.mp4" \
            "$embeddings_dir/${id}.pkl" </dev/null

        echo "Done: $split_name/$id"
    done < "$ids_file"
}

#process_split "$TRAIN_IDS_FILE" "$TRAIN_URL" "$TRAIN_EMBEDDINGS_DIR" "train"
process_split "$TEST_IDS_FILE" "$TEST_URL" "$TEST_EMBEDDINGS_DIR" "test"