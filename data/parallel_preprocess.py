#!/usr/bin/env python3
from pathlib import Path
from queue import Queue
from threading import Thread
import subprocess
import sys


TRAIN_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"
TEST_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test"

SCRIPT_DIR = Path(__file__).resolve().parent
FULL_DIR = SCRIPT_DIR / "full"
CROPPED_DIR = SCRIPT_DIR / "cropped"

TRAIN_EMBEDDINGS_DIR = SCRIPT_DIR.parent / "embeddings" / "train"
TEST_EMBEDDINGS_DIR = SCRIPT_DIR.parent / "embeddings" / "test"

TRAIN_IDS_FILE = SCRIPT_DIR / "train_ids.txt"
TEST_IDS_FILE = SCRIPT_DIR / "test_ids.txt"


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
            embedding_file = embedding_path(embeddings_dir, video_id)
            if embedding_file.exists():
                print(f"=== Skipping {split_name}/{video_id}: embedding already exists ===", flush=True)
                continue

            prefix = video_id.split("_", 1)[0]
            url = f"{base_url}/{prefix}/{video_id}.MP4"
            full_video = FULL_DIR / f"{video_id}.mp4"

            print(f"=== Downloading {split_name}/{video_id} ===", flush=True)
            subprocess.run(
                ["wget", "--quiet", "-O", str(full_video), url],
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
            ["bash", str(SCRIPT_DIR / "resize_video.sh"), str(full_video), str(cropped_video)],
            stdin=subprocess.DEVNULL,
            check=True,
        )

        print("Embedding...", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR.parent / "jepa2" / "obtain_embeddings.py"),
                str(cropped_video),
                str(embedding_file),
            ],
            stdin=subprocess.DEVNULL,
            check=True,
        )

        full_video.unlink()
        print(f"Done: {split_name}/{video_id}", flush=True)


def main():
    for directory in (FULL_DIR, CROPPED_DIR, TRAIN_EMBEDDINGS_DIR, TEST_EMBEDDINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    jobs = [
        # (TRAIN_IDS_FILE, TRAIN_URL, TRAIN_EMBEDDINGS_DIR, "train"),
        (TEST_IDS_FILE, TEST_URL, TEST_EMBEDDINGS_DIR, "test"),
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
