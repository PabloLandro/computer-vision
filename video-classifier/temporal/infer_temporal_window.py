import csv
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from classifier_data import (
    ANNOTATIONS_CSV,
    BACKGROUND_ID,
    DEFAULT_BLOCK_SIZE,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    NOUN_CLASSES_CSV,
    PREDICTIONS_DIR,
    VERB_CLASSES_CSV,
    BlockMeta,
    best_overlapping_annotation,
    get_torch_device,
    load_pickle,
    read_annotations,
    read_class_map,
    write_predictions_csv,
)
from train_temporal_classifier import (
    TEMPORAL_BATCH_SIZE,
    TemporalWindowClassifier,
    build_temporal_windows,
    compute_action_metrics,
)


# Inference settings.
USE_ENSEMBLE = True
ENSEMBLE_SEEDS = [42, 123, 999]
SINGLE_MODEL_SEED = ENSEMBLE_SEEDS[-1]

MODEL_ROOT = MODELS_DIR / "temporal_window_ensemble"

EMBEDDING_PATH = EMBEDDINGS_DIR / "P05_05.pkl"
ANNOTATIONS_PATH = ANNOTATIONS_CSV
PREDICTION_DIR = PREDICTIONS_DIR / "temporal_window_inference"
PREDICTION_FILENAME_PREFIX = "block_predictions"

EVALUATE_RELEVANT_ONLY = True
PREDICTED_RELEVANT_VALUE = 1
DEVICE = get_torch_device()

OLLAMA_BASE_URL = "http://localhost:11434"


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def get_ollama_model() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        return models[0]["name"] if models else None
    except Exception:
        return None


def ask_ollama_for_recipe(actions: list[str], model: str) -> str:
    unique_actions = sorted(set(actions))
    actions_text = "\n".join(f"- {a}" for a in unique_actions)
    prompt = (
        f"I observed the following cooking actions in a video:\n{actions_text}\n\n"
        "Based on these actions, infer what recipe is being prepared and write it out "
        "as a concise recipe with ingredients and steps."
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def generate_recipe(
    pred_verb: np.ndarray,
    pred_noun: np.ndarray,
    true_verb: np.ndarray,
    true_noun: np.ndarray,
    verb_map: dict[int, str],
    noun_map: dict[int, str],
) -> None:
    annotated_mask = (true_verb != BACKGROUND_ID) & (true_noun != BACKGROUND_ID)
    actions = [
        f"{verb_map.get(int(v), str(v))} {noun_map.get(int(n), str(n))}"
        for v, n, keep in zip(pred_verb, pred_noun, annotated_mask)
        if keep
    ]
    model = get_ollama_model()
    if model is None:
        print("\nOllama not available — skipping recipe generation.")
        return
    unique_actions = sorted(set(actions))
    print("\n--- Prompt ---")
    print("\n".join(f"- {a}" for a in unique_actions))
    print(f"\nGenerating recipe with {model}...")
    try:
        recipe = ask_ollama_for_recipe(actions, model)
        print("\n--- Recipe ---")
        print(recipe)
    except Exception as e:
        print(f"Recipe generation failed: {e}")


def load_video_blocks(
    embedding_path: Path,
    annotations_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    video_id = embedding_path.stem
    payload = load_pickle(embedding_path)
    if "embeddings" not in payload:
        raise KeyError(f"{embedding_path} is missing an 'embeddings' key")

    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    block_size = int(payload.get("block_size", DEFAULT_BLOCK_SIZE))
    annotations_by_video = read_annotations(annotations_path)
    annotations = annotations_by_video.get(video_id, [])

    metas = []
    verb_labels = []
    noun_labels = []
    relevant_labels = []
    for block_index in range(len(embeddings)):
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

        verb_labels.append(verb_id)
        noun_labels.append(noun_id)
        relevant_labels.append(relevant)
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

    y = {
        "verb": np.asarray(verb_labels, dtype=np.int64),
        "noun": np.asarray(noun_labels, dtype=np.int64),
        "relevant": np.asarray(relevant_labels, dtype=np.int64),
        "meta": metas,
    }
    return embeddings.astype(np.float32, copy=False), y


def load_encoded_to_action(model_root: Path) -> dict[int, tuple[int, int]]:
    label_maps_path = model_root / "label_maps.json"
    if label_maps_path.exists():
        import json

        with label_maps_path.open() as f:
            label_maps = json.load(f)
        return {
            int(encoded): (int(pair[0]), int(pair[1]))
            for encoded, pair in label_maps["encoded_to_action"].items()
        }

    checkpoint_path = model_root / f"seed_{SINGLE_MODEL_SEED}" / "classifier.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        int(encoded): (int(pair[0]), int(pair[1]))
        for encoded, pair in checkpoint["encoded_to_action"].items()
    }


def load_model(checkpoint_path: Path, input_dim: int, num_actions: int) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    model = TemporalWindowClassifier(
        input_dim=int(config.get("input_dim", input_dim)),
        num_verbs=len(checkpoint["encoded_to_verb"]),
        num_nouns=len(checkpoint["encoded_to_noun"]),
        num_actions=num_actions,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def model_paths(model_root: Path) -> list[Path]:
    seeds = ENSEMBLE_SEEDS if USE_ENSEMBLE else [SINGLE_MODEL_SEED]
    paths = [model_root / f"seed_{seed}" / "classifier.pt" for seed in seeds]
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return paths


def make_loader(x: np.ndarray, metas: list[BlockMeta]) -> Any:
    temporal_x, temporal_mask = build_temporal_windows(x, metas)
    dataset = TensorDataset(
        torch.from_numpy(temporal_x).float(),
        torch.from_numpy(temporal_mask).bool(),
    )
    return DataLoader(dataset, batch_size=TEMPORAL_BATCH_SIZE, shuffle=False)


def predict_actions(
    models: list[Any],
    loader: Any,
    encoded_to_action: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    pred_verbs = []
    pred_nouns = []

    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            action_logits = []
            for model in models:
                _, _, model_action_logits = model(batch_x, batch_mask)
                action_logits.append(model_action_logits)

            avg_action_logits = torch.stack(action_logits, dim=0).mean(dim=0)
            pred_actions = avg_action_logits.argmax(dim=1).cpu().numpy().tolist()
            for action in pred_actions:
                verb_id, noun_id = encoded_to_action[int(action)]
                pred_verbs.append(int(verb_id))
                pred_nouns.append(int(noun_id))

    return np.asarray(pred_verbs, dtype=np.int64), np.asarray(pred_nouns, dtype=np.int64)


def print_action_comparison(
    metas: list[BlockMeta],
    pred_verb: np.ndarray,
    pred_noun: np.ndarray,
    true_verb: np.ndarray,
    true_noun: np.ndarray,
    verb_map: dict[int, str],
    noun_map: dict[int, str],
) -> None:
    print("\n--- Block predictions ---")
    print(f"{'block':>5}  {'frames':>12}  {'predicted':<30}  {'true':<30}  ok")
    print("-" * 90)
    for i, meta in enumerate(metas):
        pv = verb_map.get(int(pred_verb[i]), str(pred_verb[i]))
        pn = noun_map.get(int(pred_noun[i]), str(pred_noun[i]))
        pred_str = f"{pv} {pn}"

        tv, tn = int(true_verb[i]), int(true_noun[i])
        if tv == BACKGROUND_ID or tn == BACKGROUND_ID:
            true_str = "<background>"
            mark = " "
        else:
            tv_name = verb_map.get(tv, str(tv))
            tn_name = noun_map.get(tn, str(tn))
            true_str = f"{tv_name} {tn_name}"
            mark = "v" if (pred_verb[i] == true_verb[i] and pred_noun[i] == true_noun[i]) else "x"

        frames = f"{meta.start_frame}-{meta.stop_frame}"
        print(f"{i:>5}  {frames:>12}  {pred_str:<30}  {true_str:<30}  {mark}")


def write_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ("verb_acc", "noun_acc", "action_exact"):
            writer.writerow([key, metrics[key]])


def main() -> None:
    x, y = load_video_blocks(EMBEDDING_PATH, ANNOTATIONS_PATH)
    encoded_to_action = load_encoded_to_action(MODEL_ROOT)
    paths = model_paths(MODEL_ROOT)
    models = [
        load_model(path, input_dim=x.shape[1], num_actions=len(encoded_to_action))
        for path in paths
    ]

    loader = make_loader(x, y["meta"])
    pred_verb, pred_noun = predict_actions(models, loader, encoded_to_action)

    run_timestamp = make_run_timestamp()
    mode_name = "ensemble" if USE_ENSEMBLE else f"seed_{SINGLE_MODEL_SEED}"
    prediction_path = (
        PREDICTION_DIR
        / f"{PREDICTION_FILENAME_PREFIX}_{EMBEDDING_PATH.stem}_{mode_name}_{run_timestamp}.csv"
    )
    pred_relevant = np.full(len(pred_verb), PREDICTED_RELEVANT_VALUE, dtype=np.int64)
    verb_map = read_class_map(VERB_CLASSES_CSV, "verb_id")
    noun_map = read_class_map(NOUN_CLASSES_CSV, "noun_id")
    write_predictions_csv(
        prediction_path,
        y["meta"],
        pred_verb,
        pred_noun,
        pred_relevant,
        verb_map,
        noun_map,
    )

    eval_mask = np.ones(len(pred_verb), dtype=bool)
    if EVALUATE_RELEVANT_ONLY:
        eval_mask = (
            (y["relevant"] == 1)
            & (y["verb"] != BACKGROUND_ID)
            & (y["noun"] != BACKGROUND_ID)
        )
    if not eval_mask.any():
        raise RuntimeError("No blocks available for evaluation with the current filter settings.")

    metrics = compute_action_metrics(
        y["verb"][eval_mask],
        pred_verb[eval_mask],
        y["noun"][eval_mask],
        pred_noun[eval_mask],
    )

    metrics_path = (
        PREDICTION_DIR
        / f"metrics_{EMBEDDING_PATH.stem}_{mode_name}_{run_timestamp}.csv"
    )
    write_metrics_csv(metrics_path, metrics)

    print(f"video_id: {EMBEDDING_PATH.stem}")
    print(f"mode: {mode_name}")
    print(f"model_root: {MODEL_ROOT}")
    print(f"prediction_csv: {prediction_path}")
    print(f"evaluated_blocks: {int(eval_mask.sum())} / {len(eval_mask)}")
    print(f"verb_acc: {metrics['verb_acc']:.6f}")
    print(f"noun_acc: {metrics['noun_acc']:.6f}")
    print(f"action_exact: {metrics['action_exact']:.6f}")

    generate_recipe(pred_verb, pred_noun, y["verb"], y["noun"], verb_map, noun_map)


if __name__ == "__main__":
    main()
