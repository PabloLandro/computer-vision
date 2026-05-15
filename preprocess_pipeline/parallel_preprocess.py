#!/usr/bin/env python3
from pathlib import Path
from queue import Queue
from threading import Thread
import subprocess
import sys


TRAIN_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"
TEST_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

FULL_DIR = REPO_ROOT / "data" / "full"
CROPPED_DIR = REPO_ROOT / "data" / "cropped"

EMBEDDINGS_DIR = REPO_ROOT / "embeddings"

TRAIN_IDS_FILE = REPO_ROOT / "data" / "video_IDs" / "train_ids.txt"
TEST_IDS_FILE = REPO_ROOT / "data" / "video_IDs" / "test_ids.txt"
MISSING_IDS_FILE = REPO_ROOT / "data" / "video_IDs" / "missing_train_IDs.txt"
EMBEDDER_FILE = SCRIPT_DIR / "jepa2" / "obtain_embeddings.py"
RESIZE_SCRIPT = SCRIPT_DIR / "resize_video.sh"

CREATE_EMBEDDINGS = False


def read_ids(ids_file):
    with ids_file.open() as f:
        for line in f:
            video_id = line.strip()
            if video_id:
                yield video_id


def embedding_path(embeddings_dir, video_id):
    return embeddings_dir / f"{video_id}.pkl"


def download_worker(jobs, downloaded):
    for ids_file, base_url, embeddings_dir, split_name in jobs:
        for video_id in read_ids(ids_file):
            cropped_file = CROPPED_DIR / f"{video_id}.mp4"
            embedding_file = embedding_path(embeddings_dir, video_id)

            if CREATE_EMBEDDINGS and embedding_file.exists():
                print(f"=== Skipping {split_name}/{video_id}: embedding already exists ===", flush=True)
                continue

            if not CREATE_EMBEDDINGS and cropped_file.exists():
                print(f"=== Skipping {split_name}/{video_id}: cropped video already exists ===", flush=True)
                continue

            prefix = video_id.split("_", 1)[0]
            url = f"{base_url}/{prefix}/{video_id}.MP4"
            full_video = FULL_DIR / f"{video_id}.mp4"

            print(f"=== Downloading {split_name}/{video_id} ===", flush=True)
            subprocess.run(
                ["wget", "-O", str(full_video), url],
                stdin=subprocess.DEVNULL,
                check=True,
            )
            downloaded.put((video_id, full_video, embeddings_dir, split_name))
            print(f"Downloaded: {split_name}/{video_id}")

    downloaded.put(None)


def processing_worker(downloaded):
    while True:
        item = downloaded.get()
        if item is None:
            break

        video_id, full_video, embeddings_dir, split_name = item
        cropped_video = CROPPED_DIR / f"{video_id}.mp4"
        embedding_file = embedding_path(embeddings_dir, video_id)

        print(f"=== Processing {split_name}/{video_id} ===", flush=True)
        print("Cropping...", flush=True)
        subprocess.run(
            ["bash", str(RESIZE_SCRIPT), str(full_video), str(cropped_video)],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        print(f"Cropped {split_name}/{video_id}", flush=True)

        if CREATE_EMBEDDINGS:
            print("Embedding...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(EMBEDDER_FILE),
                    str(cropped_video),
                    str(embedding_file),
                ],
                stdin=subprocess.DEVNULL,
                check=True,
            )

        full_video.unlink()
        print(f"Done: {split_name}/{video_id}", flush=True)


def main():
    directories = [FULL_DIR, CROPPED_DIR]
    if CREATE_EMBEDDINGS:
        directories.append(EMBEDDINGS_DIR)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    jobs = [
        # (TRAIN_IDS_FILE, TRAIN_URL, EMBEDDINGS_DIR, "train"),
        # (TEST_IDS_FILE, TEST_URL, EMBEDDINGS_DIR, "test"),
        (MISSING_IDS_FILE, TRAIN_URL, EMBEDDINGS_DIR, "train"),
    ]

    downloaded = Queue(maxsize=5)
    downloader = Thread(target=download_worker, args=(jobs, downloaded))
    processor = Thread(target=processing_worker, args=(downloaded,))

    downloader.start()
    processor.start()
    downloader.join()
    processor.join()


if __name__ == "__main__":
    main()
