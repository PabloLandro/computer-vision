from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

CROPPED_DIR = REPO_ROOT / "data" / "cropped"
FULL_EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
OBTAIN_EMBEDDINGS_SCRIPT = SCRIPT_DIR / "jepa2" / "obtain_embeddings.py"


def iter_videos():
    yield from sorted(CROPPED_DIR.glob("*.mp4"))


def main():
    FULL_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos())

    for index, video_path in enumerate(videos, start=1):
        output_path = FULL_EMBEDDINGS_DIR / f"{video_path.stem}.pkl"

        if output_path.exists():
            print(f"[{index}/{len(videos)}] Skipping {video_path.name}: embedding already exists", flush=True)
            continue

        print(f"[{index}/{len(videos)}] Embedding {video_path.name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(OBTAIN_EMBEDDINGS_SCRIPT),
                str(video_path),
                str(output_path),
            ],
            stdin=subprocess.DEVNULL,
            check=True,
        )

    print(f"Saved embeddings to {FULL_EMBEDDINGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
