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
    RELEVANCE_THRESHOLD,
    TEMPORAL_BATCH_SIZE,
    TemporalWindowClassifier,
    build_temporal_windows,
    compute_action_metrics,
    compute_pipeline_metrics,
    compute_relevance_metrics,
)


# Inference settings.
USE_ENSEMBLE = True
ENSEMBLE_SEEDS = [42, 123, 999]
SINGLE_MODEL_SEED = ENSEMBLE_SEEDS[-1]

MODEL_ROOT = Path("outputs/models/temporal_window_ensemble")

EMBEDDING_DIR = Path("data/unseen")
ANNOTATIONS_PATH = ANNOTATIONS_CSV
PREDICTION_DIR = PREDICTIONS_DIR / "temporal_window_inference"
PREDICTION_FILENAME_PREFIX = "block_predictions"

DEVICE = get_torch_device()

OLLAMA_BASE_URL = "http://localhost:11434"
GENERATE_RECIPE = True


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def get_ollama_model() -> str | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    models = [model["name"] for model in data.get("models", []) if "name" in model]
    if not models:
        return None

    print("\n--- Available Ollama Models ---")
    for idx, model in enumerate(models):
        print(f"[{idx}] {model}")

    while True:
        choice = input("Select Ollama model index for recipe generation: ").strip()

        try:
            model_idx = int(choice)
        except ValueError:
            print("Please enter a numeric model index.")
            continue

        if 0 <= model_idx < len(models):
            return models[model_idx]

        print(f"Please enter an index from 0 to {len(models) - 1}.")


def collapse_consecutive_actions(actions: list[str]) -> list[str]:
    collapsed = []
    previous_action = None
    for action in actions:
        if action != previous_action:
            collapsed.append(action)
        previous_action = action
    return collapsed


def ask_ollama_for_recipe(actions: list[str], model: str) -> str:
    prompt_actions = collapse_consecutive_actions(actions)
    actions_text = "\n".join(f"- {a}" for a in prompt_actions)

    prompt = (
        f"I observed the following cooking actions in a video:\n{actions_text}\n\n"
        "Based on these actions, infer what recipe is being prepared and write it out "
        "as a concise recipe with ingredients and steps."
    )

    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()

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
    pred_relevant: np.ndarray,
    verb_map: dict[int, str],
    noun_map: dict[int, str],
) -> None:
    relevant_mask = pred_relevant == 1

    actions = [
        f"{verb_map.get(int(v), str(v))} {noun_map.get(int(n), str(n))}"
        for v, n, keep in zip(pred_verb, pred_noun, relevant_mask)
        if keep
    ]

    if not actions:
        print("\nNo relevant actions detected — skipping recipe generation.")
        return

    model = get_ollama_model()
    if model is None:
        print("\nOllama not available — skipping recipe generation.")
        return

    prompt_actions = collapse_consecutive_actions(actions)

    print("\n--- Ollama Prompt Actions ---")
    print("\n".join(f"- {a}" for a in prompt_actions))
    print(f"\nGenerating recipe with {model}...")

    try:
        recipe = ask_ollama_for_recipe(actions, model)

        print("\n--- Generated Recipe ---")
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
    if not any(key.startswith("relevance_head.") for key in checkpoint["model_state_dict"]):
        raise RuntimeError(
            f"{checkpoint_path} does not contain relevance head weights. "
            "Retrain the temporal classifier with the relevance-head pipeline before inference."
        )
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_verbs = []
    pred_nouns = []
    pred_relevant = []
    p_relevant = []

    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            action_logits = []
            relevance_logits = []
            for model in models:
                _, _, model_action_logits, model_relevance_logits = model(batch_x, batch_mask)
                action_logits.append(model_action_logits)
                relevance_logits.append(model_relevance_logits)

            avg_action_logits = torch.stack(action_logits, dim=0).mean(dim=0)
            avg_relevance_logits = torch.stack(relevance_logits, dim=0).mean(dim=0)
            relevance_probs = torch.softmax(avg_relevance_logits, dim=-1)[:, 1]
            pred_relevant.extend(
                (relevance_probs >= RELEVANCE_THRESHOLD).long().cpu().numpy().tolist()
            )
            p_relevant.extend(relevance_probs.cpu().numpy().tolist())
            pred_actions = avg_action_logits.argmax(dim=1).cpu().numpy().tolist()
            for action in pred_actions:
                verb_id, noun_id = encoded_to_action[int(action)]
                pred_verbs.append(int(verb_id))
                pred_nouns.append(int(noun_id))

    return (
        np.asarray(pred_verbs, dtype=np.int64),
        np.asarray(pred_nouns, dtype=np.int64),
        np.asarray(pred_relevant, dtype=np.int64),
        np.asarray(p_relevant, dtype=np.float32),
    )


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, metric_key))
        else:
            flattened[metric_key] = float(value)
    return flattened


def write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in flatten_metrics(metrics).items():
            writer.writerow([key, value])


def infer_embedding_file(
    embedding_path: Path,
    encoded_to_action: dict[int, tuple[int, int]],
    paths: list[Path],
    mode_name: str,
    run_timestamp: str,
) -> None:
    x, y = load_video_blocks(embedding_path, ANNOTATIONS_PATH)
    models = [
        load_model(path, input_dim=x.shape[1], num_actions=len(encoded_to_action))
        for path in paths
    ]

    loader = make_loader(x, y["meta"])
    pred_verb, pred_noun, pred_relevant, p_relevant = predict_actions(
        models,
        loader,
        encoded_to_action,
    )

    prediction_path = (
        PREDICTION_DIR
        / f"{PREDICTION_FILENAME_PREFIX}_{embedding_path.stem}_{mode_name}_{run_timestamp}.csv"
    )
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
        p_relevant=p_relevant,
    )

    valid_action_mask = (
        (y["relevant"] == 1)
        & (y["verb"] != BACKGROUND_ID)
        & (y["noun"] != BACKGROUND_ID)
    )
    metrics = compute_pipeline_metrics(
        y,
        np.arange(len(pred_verb)),
        pred_verb,
        pred_noun,
        pred_relevant,
    )
    metrics["oracle_action_accuracy_using_ground_truth_relevance"] = compute_action_metrics(
        y["verb"][valid_action_mask],
        pred_verb[valid_action_mask],
        y["noun"][valid_action_mask],
        pred_noun[valid_action_mask],
    )
    metrics["relevance"] = compute_relevance_metrics(y["relevant"], pred_relevant)

    metrics_path = (
        PREDICTION_DIR
        / f"metrics_{embedding_path.stem}_{mode_name}_{run_timestamp}.csv"
    )
    write_metrics_csv(metrics_path, metrics)

    oracle_metrics = metrics["oracle_action_accuracy_using_ground_truth_relevance"]
    deployed_metrics = metrics["action_metrics_on_true_relevant_blocks"]
    print(f"filename: {embedding_path.name}")
    print(f"video_id: {embedding_path.stem}")
    print(f"mode: {mode_name}")
    print(f"model_root: {MODEL_ROOT}")
    print(f"prediction_csv: {prediction_path}")
    print(
        "oracle action accuracy using ground-truth relevance: "
        f"{oracle_metrics['action_exact']:.6f} over {oracle_metrics['count']} blocks"
    )
    print(
        "deployed action accuracy after predicted relevance gate: "
        f"{deployed_metrics['action_exact']:.6f} over {deployed_metrics['count']} true-relevant blocks"
    )
    print(
        "relevance: "
        f"acc={metrics['relevance_accuracy']:.6f} "
        f"precision={metrics['relevance_precision']:.6f} "
        f"recall={metrics['relevance_recall']:.6f} "
        f"f1={metrics['relevance_f1']:.6f} "
        f"pred_count={metrics['predicted_relevant_count']} "
        f"true_count={metrics['true_relevant_count']}"
    )
    print(f"relevance_threshold: {RELEVANCE_THRESHOLD:.3f}")

    if GENERATE_RECIPE:
        generate_recipe(
            pred_verb,
            pred_noun,
            pred_relevant,
            verb_map,
            noun_map,
        )


def main() -> None:
    embedding_paths = sorted(EMBEDDING_DIR.glob("*.pkl"))
    if not embedding_paths:
        raise FileNotFoundError(f"No .pkl files found in {EMBEDDING_DIR}")

    encoded_to_action = load_encoded_to_action(MODEL_ROOT)
    paths = model_paths(MODEL_ROOT)
    run_timestamp = make_run_timestamp()
    mode_name = "ensemble" if USE_ENSEMBLE else f"seed_{SINGLE_MODEL_SEED}"

    for embedding_path in embedding_paths:
        infer_embedding_file(
            embedding_path,
            encoded_to_action,
            paths,
            mode_name,
            run_timestamp,
        )


if __name__ == "__main__":
    main()
