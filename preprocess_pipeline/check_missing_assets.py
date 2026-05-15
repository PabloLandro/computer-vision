from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

TRAIN_IDS_FILE = REPO_ROOT / "data" / "video_IDs" / "train_ids.txt"
TEST_IDS_FILE = REPO_ROOT / "data" / "video_IDs" / "test_ids.txt"
CROPPED_DIR = REPO_ROOT / "data" / "cropped"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"

WRITE_MERGED_IDS = False
MERGED_IDS_FILE = SCRIPT_DIR / "merged_ids.txt"


def read_ids(path):
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def merge_ids(sources):
    merged = []
    seen = set()
    duplicates = []
    id_sources = {}

    for source_name, path in sources:
        for video_id in read_ids(path):
            id_sources.setdefault(video_id, []).append(source_name)
            if video_id in seen:
                duplicates.append(video_id)
                continue
            seen.add(video_id)
            merged.append(video_id)

    return merged, duplicates, id_sources


def filename_stems(directory):
    if not directory.exists():
        return set()
    return {path.stem for path in directory.iterdir() if path.is_file()}


def source_label(sources):
    return ", ".join(sources)


def print_missing(label, missing, id_sources):
    print(f"{label}: {len(missing)} missing")

    grouped = {}
    for video_id in missing:
        grouped.setdefault(source_label(id_sources.get(video_id, ["unknown"])), []).append(video_id)

    for source, video_ids in grouped.items():
        print(f"{source}:")
        for video_id in video_ids:
            print(video_id)


def main():
    ids, duplicates, id_sources = merge_ids(
        [
            ("train", TRAIN_IDS_FILE),
            ("test", TEST_IDS_FILE),
        ]
    )

    cropped_ids = filename_stems(CROPPED_DIR)
    embedding_ids = filename_stems(EMBEDDINGS_DIR)

    missing_cropped = [video_id for video_id in ids if video_id not in cropped_ids]
    cropped_missing_embeddings = [
        video_id
        for video_id in ids
        if video_id in cropped_ids and video_id not in embedding_ids
    ]
    extra_cropped_missing_embeddings = sorted(cropped_ids - set(ids) - embedding_ids)

    print(f"Merged ids: {len(ids)}")
    print(f"Duplicate ids skipped: {len(duplicates)}")
    print()
    print_missing(f"Ids not in {CROPPED_DIR}", missing_cropped, id_sources)
    print()
    print_missing(
        f"Files in {CROPPED_DIR} not in {EMBEDDINGS_DIR}",
        cropped_missing_embeddings + extra_cropped_missing_embeddings,
        id_sources,
    )

    if WRITE_MERGED_IDS:
        MERGED_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MERGED_IDS_FILE.write_text("\n".join(ids) + "\n")
        print()
        print(f"Wrote merged ids to {MERGED_IDS_FILE}")


if __name__ == "__main__":
    main()
