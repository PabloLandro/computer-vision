from __future__ import annotations

import ast
import csv
import json
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_CLASSIFIER_DIR = SCRIPT_DIR.parent
REPO_ROOT = VIDEO_CLASSIFIER_DIR.parent

ANNOTATIONS_CSV = REPO_ROOT / "data" / "annotations" / "annotations_train_test.csv"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
VERB_CLASSES_CSV = REPO_ROOT / "data" / "classes" / "EPIC_verb_classes.csv"
NOUN_CLASSES_CSV = REPO_ROOT / "data" / "classes" / "EPIC_noun_classes.csv"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

RANDOM_SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

DEFAULT_BLOCK_SIZE = 64
MIN_ACTION_OVERLAP_FRAMES = DEFAULT_BLOCK_SIZE // 2
BACKGROUND_ID = -1
BACKGROUND_NAME = "<background>"

# Label alignment policy for a block that overlaps multiple annotated actions.
# The selected annotation is the one with the largest frame overlap.
KEEP_ONLY_RELEVANT_ACTIONS = False


def progress(iterable: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return tqdm(iterable, **kwargs)


@dataclass(frozen=True)
class Annotation:
    video_id: str
    start_frame: int
    stop_frame: int
    verb_id: int
    noun_id: int
    verb: str
    noun: str
    relevant: int


@dataclass(frozen=True)
class BlockMeta:
    video_id: str
    block_index: int
    start_frame: int
    stop_frame: int
    has_annotation: int
    true_verb_id: int
    true_noun_id: int
    true_relevant: int


def read_class_map(path: Path, id_column: str) -> dict[int, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {int(row[id_column]): row["class_key"] for row in reader}


def parse_first_noun_class(raw_value: str) -> int:
    values = ast.literal_eval(raw_value)
    if not values:
        return BACKGROUND_ID
    return int(values[0])


def parse_bool(raw_value: str) -> int:
    return int(str(raw_value).strip().lower() == "true")


def read_annotations(path: Path) -> dict[str, list[Annotation]]:
    annotations_by_video: dict[str, list[Annotation]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relevant = parse_bool(row["relevant"])
            if KEEP_ONLY_RELEVANT_ACTIONS and not relevant:
                continue
            ann = Annotation(
                video_id=row["video_id"],
                start_frame=int(row["start_frame"]),
                stop_frame=int(row["stop_frame"]),
                verb_id=int(row["verb_class"]),
                noun_id=parse_first_noun_class(row["all_noun_classes"]),
                verb=row["verb"],
                noun=row["noun"],
                relevant=relevant,
            )
            annotations_by_video[ann.video_id].append(ann)

    for video_annotations in annotations_by_video.values():
        video_annotations.sort(key=lambda item: (item.start_frame, item.stop_frame))
    return annotations_by_video


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            # Some embedding files are pickled by NumPy 2.x as numpy._core.*.
            # Older environments may only expose numpy.core.*.
            if exc.name != "numpy._core":
                raise

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    with path.open("rb") as f:
        return pickle.load(f)


def best_overlapping_annotation(
    annotations: list[Annotation],
    start_frame: int,
    stop_frame: int,
) -> Annotation | None:
    best_annotation = None
    best_overlap = 0

    for ann in annotations:
        if ann.stop_frame < start_frame:
            continue
        if ann.start_frame > stop_frame:
            break
        overlap = min(stop_frame, ann.stop_frame) - max(start_frame, ann.start_frame) + 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_annotation = ann

    if best_overlap < MIN_ACTION_OVERLAP_FRAMES:
        return None
    return best_annotation


def build_dataset(
    annotations_by_video: dict[str, list[Annotation]],
) -> tuple[Any, dict[str, Any]]:
    features: list[Any] = []
    metas: list[BlockMeta] = []
    verb_labels: list[int] = []
    noun_labels: list[int] = []
    relevant_labels: list[int] = []
    action_keys: list[tuple[int, int]] = []
    skipped_files = []

    embedding_paths = sorted(
        path for path in EMBEDDINGS_DIR.glob("*.pkl") if not path.name.startswith(".")
    )
    if not embedding_paths:
        raise FileNotFoundError(f"No .pkl embedding files found in {EMBEDDINGS_DIR}")

    for path in progress(embedding_paths, desc="Loading embeddings"):
        video_id = path.stem
        payload = load_pickle(path)
        if "embeddings" not in payload:
            skipped_files.append({"path": str(path), "reason": "missing embeddings key"})
            continue

        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        block_size = int(payload.get("block_size", DEFAULT_BLOCK_SIZE))
        annotations = annotations_by_video.get(video_id, [])

        for block_index, embedding in enumerate(embeddings):
            start_frame = block_index * block_size
            stop_frame = start_frame + block_size - 1
            ann = best_overlapping_annotation(annotations, start_frame, stop_frame)

            if ann is None:
                verb_id = BACKGROUND_ID
                noun_id = BACKGROUND_ID
                relevant = 0
                has_annotation = 0
            else:
                verb_id = ann.verb_id
                noun_id = ann.noun_id
                relevant = ann.relevant
                has_annotation = 1

            features.append(embedding)
            verb_labels.append(verb_id)
            noun_labels.append(noun_id)
            relevant_labels.append(relevant)
            action_keys.append((verb_id, noun_id))
            metas.append(
                BlockMeta(
                    video_id=video_id,
                    block_index=block_index,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    has_annotation=has_annotation,
                    true_verb_id=verb_id,
                    true_noun_id=noun_id,
                    true_relevant=relevant,
                )
            )

    if not features:
        raise RuntimeError("No usable embeddings were loaded.")

    x = np.vstack(features).astype(np.float32, copy=False)
    y = {
        "verb": np.asarray(verb_labels, dtype=np.int64),
        "noun": np.asarray(noun_labels, dtype=np.int64),
        "relevant": np.asarray(relevant_labels, dtype=np.int64),
        "action": np.asarray([str(key) for key in action_keys]),
        "meta": metas,
        "skipped_files": skipped_files,
    }
    return x, y


def stratified_block_split(labels: Any) -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for indices in groups.values():
        rng.shuffle(indices)
        n_items = len(indices)
        if n_items == 1:
            train_indices.extend(indices)
            continue
        if n_items == 2:
            train_indices.append(indices[0])
            val_indices.append(indices[1])
            continue

        n_test = max(1, int(round(n_items * TEST_FRACTION)))
        n_val = max(1, int(round(n_items * VAL_FRACTION)))
        while n_test + n_val > n_items - 1:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test : n_test + n_val])
        train_indices.extend(indices[n_test + n_val :])

    for split in (train_indices, val_indices, test_indices):
        rng.shuffle(split)

    return {
        "train": np.asarray(train_indices, dtype=np.int64),
        "val": np.asarray(val_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }


def video_split(metas: list[BlockMeta]) -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    video_ids = sorted({meta.video_id for meta in metas})
    rng.shuffle(video_ids)

    n_videos = len(video_ids)
    n_test = max(1, int(round(n_videos * TEST_FRACTION)))
    n_val = max(1, int(round(n_videos * VAL_FRACTION)))
    while n_test + n_val > n_videos - 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    split_videos = {
        "test": set(video_ids[:n_test]),
        "val": set(video_ids[n_test : n_test + n_val]),
        "train": set(video_ids[n_test + n_val :]),
    }

    return {
        split_name: np.asarray(
            [index for index, meta in enumerate(metas) if meta.video_id in videos],
            dtype=np.int64,
        )
        for split_name, videos in split_videos.items()
    }


def make_label_encoding(values: Any) -> tuple[dict[int, int], dict[int, int]]:
    unique_values = sorted(int(value) for value in set(values))
    raw_to_encoded = {raw: encoded for encoded, raw in enumerate(unique_values)}
    encoded_to_raw = {encoded: raw for raw, encoded in raw_to_encoded.items()}
    return raw_to_encoded, encoded_to_raw


def encode_labels(values: Any, raw_to_encoded: dict[int, int]) -> Any:
    return np.asarray([raw_to_encoded[int(value)] for value in values], dtype=np.int64)


def decode_labels(values: Any, encoded_to_raw: dict[int, int]) -> Any:
    return np.asarray([encoded_to_raw[int(value)] for value in values], dtype=np.int64)


def label_name(label_id: int, class_map: dict[int, str]) -> str:
    if int(label_id) == BACKGROUND_ID:
        return BACKGROUND_NAME
    return class_map.get(int(label_id), f"id:{label_id}")


def action_name(
    verb_id: int,
    noun_id: int,
    verb_map: dict[int, str],
    noun_map: dict[int, str],
) -> str:
    if int(verb_id) == BACKGROUND_ID or int(noun_id) == BACKGROUND_ID:
        return BACKGROUND_NAME
    return f"{label_name(verb_id, verb_map)} {label_name(noun_id, noun_map)}"


def compute_metrics(
    y_true_verb: Any,
    y_pred_verb: Any,
    y_true_noun: Any,
    y_pred_noun: Any,
    y_true_relevant: Any,
    y_pred_relevant: Any,
) -> dict[str, Any]:
    true_actions = np.asarray(list(zip(y_true_verb, y_true_noun)), dtype=object)
    pred_actions = np.asarray(list(zip(y_pred_verb, y_pred_noun)), dtype=object)
    action_exact = np.asarray(
        [
            int(tv == pv and tn == pn)
            for (tv, tn), (pv, pn) in zip(true_actions, pred_actions, strict=True)
        ],
        dtype=np.int64,
    )

    relevant_precision, relevant_recall, relevant_f1, _ = precision_recall_fscore_support(
        y_true_relevant,
        y_pred_relevant,
        average="binary",
        zero_division=0,
    )

    return {
        "verb_accuracy": float(accuracy_score(y_true_verb, y_pred_verb)),
        "verb_macro_f1": float(f1_score(y_true_verb, y_pred_verb, average="macro", zero_division=0)),
        "verb_weighted_f1": float(
            f1_score(y_true_verb, y_pred_verb, average="weighted", zero_division=0)
        ),
        "noun_accuracy": float(accuracy_score(y_true_noun, y_pred_noun)),
        "noun_macro_f1": float(f1_score(y_true_noun, y_pred_noun, average="macro", zero_division=0)),
        "noun_weighted_f1": float(
            f1_score(y_true_noun, y_pred_noun, average="weighted", zero_division=0)
        ),
        "action_exact_accuracy": float(action_exact.mean()),
        "relevance_accuracy": float(accuracy_score(y_true_relevant, y_pred_relevant)),
        "relevance_precision": float(relevant_precision),
        "relevance_recall": float(relevant_recall),
        "relevance_f1": float(relevant_f1),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_predictions_csv(
    path: Path,
    metas: list[BlockMeta],
    pred_verb: Any,
    pred_noun: Any,
    pred_relevant: Any,
    verb_map: dict[int, str],
    noun_map: dict[int, str],
    p_relevant: Any | None = None,
    accepted_pred_verb: Any | None = None,
    accepted_pred_noun: Any | None = None,
) -> None:
    if p_relevant is None:
        p_relevant = [float(int(value)) for value in pred_relevant]
    if accepted_pred_verb is None:
        accepted_pred_verb = [
            int(verb_id) if int(relevant) else BACKGROUND_ID
            for verb_id, relevant in zip(pred_verb, pred_relevant, strict=True)
        ]
    if accepted_pred_noun is None:
        accepted_pred_noun = [
            int(noun_id) if int(relevant) else BACKGROUND_ID
            for noun_id, relevant in zip(pred_noun, pred_relevant, strict=True)
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_id",
                "block_index",
                "frame_start",
                "frame_stop",
                "true_action",
                "raw_pred_action",
                "accepted_pred_action",
                "true_relevant",
                "pred_relevant",
                "p_relevant",
            ]
        )
        for meta, verb_id, noun_id, accepted_verb_id, accepted_noun_id, relevant, prob in zip(
            metas,
            pred_verb,
            pred_noun,
            accepted_pred_verb,
            accepted_pred_noun,
            pred_relevant,
            p_relevant,
            strict=True,
        ):
            writer.writerow(
                [
                    meta.video_id,
                    meta.block_index,
                    meta.start_frame,
                    meta.stop_frame,
                    action_name(meta.true_verb_id, meta.true_noun_id, verb_map, noun_map),
                    action_name(int(verb_id), int(noun_id), verb_map, noun_map),
                    action_name(int(accepted_verb_id), int(accepted_noun_id), verb_map, noun_map),
                    bool(meta.true_relevant),
                    bool(int(relevant)),
                    float(prob),
                ]
            )


def get_torch_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset_summary(
    x: Any,
    y: dict[str, Any],
    splits: dict[str, Any],
) -> dict[str, Any]:
    annotated_blocks = int(sum(meta.has_annotation for meta in y["meta"]))
    background_blocks = int(
        np.logical_or(y["verb"] == BACKGROUND_ID, y["noun"] == BACKGROUND_ID).sum()
    )
    summary = {
        "num_blocks": int(len(x)),
        "embedding_dim": int(x.shape[1]),
        "num_videos": int(len({meta.video_id for meta in y["meta"]})),
        "num_verb_labels": int(len(set(y["verb"].tolist()))),
        "num_noun_labels": int(len(set(y["noun"].tolist()))),
        "num_action_pairs": int(len(set(y["action"].tolist()))),
        "annotated_blocks": annotated_blocks,
        "unannotated_blocks": int(len(x) - annotated_blocks),
        "background_blocks": background_blocks,
        "relevant_blocks": int(np.asarray(y["relevant"]).sum()),
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "skipped_files": y["skipped_files"],
        "split_note": "Splits are block-level and stratified by verb+noun action pair where possible.",
    }
    return summary


def print_dataset_summary(summary: dict[str, Any]) -> None:
    split_sizes = summary["split_sizes"]
    print("Dataset summary:")
    print(f"  blocks: {summary['num_blocks']}")
    print(f"  videos: {summary['num_videos']}")
    print(f"  embedding_dim: {summary['embedding_dim']}")
    print(
        "  split: "
        f"train={split_sizes.get('train', 0)}, "
        f"val={split_sizes.get('val', 0)}, "
        f"test={split_sizes.get('test', 0)}"
    )
    print(f"  annotated blocks: {summary['annotated_blocks']}")
    print(f"  unannotated blocks: {summary['unannotated_blocks']}")
    print(f"  {BACKGROUND_NAME} blocks: {summary['background_blocks']}")
    print(f"  relevant blocks: {summary['relevant_blocks']}")
    print(f"  verb labels: {summary['num_verb_labels']}")
    print(f"  noun labels: {summary['num_noun_labels']}")
    print(f"  action pairs: {summary['num_action_pairs']}")
    if summary["skipped_files"]:
        print(f"  skipped embedding files: {len(summary['skipped_files'])}")
