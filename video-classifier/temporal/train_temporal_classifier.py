import math
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from classifier_data import (
    ANNOTATIONS_CSV,
    BACKGROUND_ID,
    MODELS_DIR,
    NOUN_CLASSES_CSV,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    VERB_CLASSES_CSV,
    build_dataset,
    build_dataset_summary,
    decode_labels,
    encode_labels,
    get_torch_device,
    make_label_encoding,
    print_dataset_summary,
    progress,
    read_annotations,
    read_class_map,
    stratified_block_split,
    write_json,
    write_predictions_csv,
)

# Temporal model settings.
TEMPORAL_WINDOW_RADIUS = 3
TEMPORAL_WINDOW_SIZE = 2 * TEMPORAL_WINDOW_RADIUS + 1
TEMPORAL_BATCH_SIZE = 256
TEMPORAL_EPOCHS = 70
TEMPORAL_LEARNING_RATE = 1e-4
TEMPORAL_WEIGHT_DECAY = 1e-4
TEMPORAL_EARLY_STOP_PATIENCE = 8
TEMPORAL_NUM_WORKERS = 0
TEMPORAL_MODEL_DIM = 768
TEMPORAL_NUM_HEADS = 8
TEMPORAL_FF_DIM = 1536
TEMPORAL_ENCODER_DROPOUT = 0.25
TEMPORAL_ENCODER_LAYERS = 2
TEMPORAL_PROJECTION_DROPOUT = 0.25
VERB_HEAD_DROPOUT = 0.2
NOUN_HEAD_DROPOUT = 0.30
ACTION_HEAD_DROPOUT = 0.20

PAIR_PRIOR_WEIGHT = 0.25
PAIR_PRIOR_SMOOTHING = 1.0
USE_PAIR_PRIOR = True
USE_ACTION_HEAD_FOR_PREDICTION = True
USE_ACTION_ENSEMBLE = False
ENSEMBLE_SEEDS = [42, 123, 999]
USE_SEED_ENSEMBLE = True
VERB_HEAD_WEIGHT = 1.0
NOUN_HEAD_WEIGHT = 1.0
ACTION_HEAD_WEIGHT = 1.0
ACTION_LOSS_WEIGHT = 2.0
RELEVANCE_HEAD_DROPOUT = 0.20
RELEVANCE_LOSS_WEIGHT = 1.0
RELEVANCE_THRESHOLD = 0.5
TRAIN_RELEVANCE_ALL_BLOCKS = True
MIN_CLASS_COUNT = 2


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def filter_relevant_non_singleton_blocks(x: Any, y: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    keep_mask = (
        (np.asarray(y["relevant"]) == 1)
        & (np.asarray(y["verb"]) != BACKGROUND_ID)
        & (np.asarray(y["noun"]) != BACKGROUND_ID)
    )

    while True:
        kept_indices = np.flatnonzero(keep_mask)
        if len(kept_indices) == 0:
            break

        verb_values, verb_counts = np.unique(y["verb"][kept_indices], return_counts=True)
        noun_values, noun_counts = np.unique(y["noun"][kept_indices], return_counts=True)
        valid_verbs = set(verb_values[verb_counts >= MIN_CLASS_COUNT].tolist())
        valid_nouns = set(noun_values[noun_counts >= MIN_CLASS_COUNT].tolist())

        next_keep_mask = keep_mask.copy()
        for index in kept_indices:
            if int(y["verb"][index]) not in valid_verbs or int(y["noun"][index]) not in valid_nouns:
                next_keep_mask[index] = False

        if np.array_equal(next_keep_mask, keep_mask):
            break
        keep_mask = next_keep_mask

    indices = np.flatnonzero(keep_mask)
    if len(indices) == 0:
        raise RuntimeError("No relevant blocks remain after dropping singleton verb/noun classes.")

    filtered_y = {
        "verb": y["verb"][indices],
        "noun": y["noun"][indices],
        "relevant": y["relevant"][indices],
        "action": y["action"][indices],
        "meta": [y["meta"][int(index)] for index in indices],
        "skipped_files": y["skipped_files"],
        "filter_summary": {
            "source_blocks": int(len(x)),
            "kept_relevant_blocks": int(len(indices)),
            "dropped_blocks": int(len(x) - len(indices)),
            "min_class_count": MIN_CLASS_COUNT,
        },
    }
    return x[indices], filtered_y


def compute_action_metrics(
    y_true_verb: Any,
    y_pred_verb: Any,
    y_true_noun: Any,
    y_pred_noun: Any,
) -> dict[str, float]:
    y_true_verb = np.asarray(y_true_verb)
    y_pred_verb = np.asarray(y_pred_verb)
    y_true_noun = np.asarray(y_true_noun)
    y_pred_noun = np.asarray(y_pred_noun)
    if len(y_true_verb) == 0:
        return {
            "verb_acc": 0.0,
            "noun_acc": 0.0,
            "action_exact": 0.0,
            "count": 0,
        }
    action_exact = (y_true_verb == y_pred_verb) & (y_true_noun == y_pred_noun)
    return {
        "verb_acc": float((y_true_verb == y_pred_verb).mean()),
        "noun_acc": float((y_true_noun == y_pred_noun).mean()),
        "action_exact": float(action_exact.mean()),
        "count": int(len(y_true_verb)),
    }


def valid_action_mask_from_labels(y: dict[str, Any]) -> np.ndarray:
    return (
        (np.asarray(y["relevant"]) == 1)
        & (np.asarray(y["verb"]) != BACKGROUND_ID)
        & (np.asarray(y["noun"]) != BACKGROUND_ID)
    )


def compute_relevance_metrics(
    y_true_relevant: Any,
    y_pred_relevant: Any,
) -> dict[str, float]:
    true_values = np.asarray(y_true_relevant, dtype=np.int64)
    pred_values = np.asarray(y_pred_relevant, dtype=np.int64)
    true_positive = int(((true_values == 1) & (pred_values == 1)).sum())
    false_positive = int(((true_values == 0) & (pred_values == 1)).sum())
    false_negative = int(((true_values == 1) & (pred_values == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "relevance_accuracy": float((true_values == pred_values).mean()) if len(true_values) else 0.0,
        "relevance_precision": float(precision),
        "relevance_recall": float(recall),
        "relevance_f1": float(f1),
        "predicted_relevant_count": int((pred_values == 1).sum()),
        "true_relevant_count": int((true_values == 1).sum()),
    }


def compute_pipeline_metrics(
    y: dict[str, Any],
    indices: Any,
    pred_verb: Any,
    pred_noun: Any,
    pred_relevant: Any,
) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    valid_action_mask = valid_action_mask_from_labels(y)[indices]
    predicted_relevant_mask = np.asarray(pred_relevant, dtype=np.int64) == 1
    accepted_pred_verb = np.where(predicted_relevant_mask, pred_verb, BACKGROUND_ID)
    accepted_pred_noun = np.where(predicted_relevant_mask, pred_noun, BACKGROUND_ID)

    metrics: dict[str, Any] = compute_relevance_metrics(y["relevant"][indices], pred_relevant)
    metrics["oracle_relevant_action_metrics"] = compute_action_metrics(
        y["verb"][indices][valid_action_mask],
        pred_verb[valid_action_mask],
        y["noun"][indices][valid_action_mask],
        pred_noun[valid_action_mask],
    )
    metrics["action_metrics_on_true_relevant_blocks"] = compute_action_metrics(
        y["verb"][indices][valid_action_mask],
        accepted_pred_verb[valid_action_mask],
        y["noun"][indices][valid_action_mask],
        accepted_pred_noun[valid_action_mask],
    )
    predicted_eval_mask = predicted_relevant_mask & valid_action_mask
    metrics["action_metrics_on_predicted_relevant_blocks"] = compute_action_metrics(
        y["verb"][indices][predicted_eval_mask],
        pred_verb[predicted_eval_mask],
        y["noun"][indices][predicted_eval_mask],
        pred_noun[predicted_eval_mask],
    )
    return metrics


def build_temporal_windows(x: Any, metas: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    index_by_video_block = {
        (meta.video_id, int(meta.block_index)): index
        for index, meta in enumerate(metas)
    }
    windows = np.zeros(
        (len(x), TEMPORAL_WINDOW_SIZE, int(x.shape[1])),
        dtype=np.float32,
    )
    valid_mask = np.zeros((len(x), TEMPORAL_WINDOW_SIZE), dtype=bool)

    for center_index, meta in enumerate(metas):
        for window_position, offset in enumerate(
            range(-TEMPORAL_WINDOW_RADIUS, TEMPORAL_WINDOW_RADIUS + 1)
        ):
            neighbor_key = (meta.video_id, int(meta.block_index) + offset)
            neighbor_index = index_by_video_block.get(neighbor_key)
            if neighbor_index is None:
                continue
            windows[center_index, window_position] = x[neighbor_index]
            valid_mask[center_index, window_position] = True

    return windows, valid_mask


def build_pair_prior(
    y_verb_encoded: Any,
    y_noun_encoded: Any,
    y_relevant: Any,
    train_indices: Any,
    num_verbs: int,
    num_nouns: int,
    device: torch.device,
) -> Any:
    pair_counts = np.full(
        (num_verbs, num_nouns),
        PAIR_PRIOR_SMOOTHING,
        dtype=np.float64,
    )
    for index in train_indices:
        if int(y_relevant[int(index)]) != 1:
            continue
        verb_id = int(y_verb_encoded[int(index)])
        noun_id = int(y_noun_encoded[int(index)])
        if verb_id < 0 or noun_id < 0:
            continue
        pair_counts[verb_id, noun_id] += 1.0

    pair_probs = pair_counts / pair_counts.sum(axis=1, keepdims=True)
    pair_prior = np.log(pair_probs)
    return torch.tensor(pair_prior, dtype=torch.float32, device=device)


class TemporalWindowClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_verbs: int,
        num_nouns: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, TEMPORAL_MODEL_DIM),
            nn.LayerNorm(TEMPORAL_MODEL_DIM),
            nn.GELU(),
            nn.Dropout(TEMPORAL_PROJECTION_DROPOUT),
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, TEMPORAL_WINDOW_SIZE, TEMPORAL_MODEL_DIM)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_MODEL_DIM,
            nhead=TEMPORAL_NUM_HEADS,
            dim_feedforward=TEMPORAL_FF_DIM,
            dropout=TEMPORAL_ENCODER_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TEMPORAL_ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.verb_head = self.make_head(num_verbs, VERB_HEAD_DROPOUT)
        self.noun_head = self.make_head(num_nouns, NOUN_HEAD_DROPOUT)
        self.action_head = self.make_head(num_actions, ACTION_HEAD_DROPOUT)
        self.relevance_head = self.make_head(2, RELEVANCE_HEAD_DROPOUT)

    @staticmethod
    def make_head(num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(TEMPORAL_MODEL_DIM, TEMPORAL_MODEL_DIM),
            nn.LayerNorm(TEMPORAL_MODEL_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(TEMPORAL_MODEL_DIM, num_classes),
        )

    def forward(self, inputs: Any, valid_mask: Any | None = None) -> tuple[Any, Any, Any, Any]:
        hidden = self.projection(inputs) + self.positional_embedding
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        center_hidden = encoded[:, TEMPORAL_WINDOW_RADIUS]
        return (
            self.verb_head(center_hidden),
            self.noun_head(center_hidden),
            self.action_head(center_hidden),
            self.relevance_head(center_hidden),
        )


def temporal_config(input_dim: int, num_actions: int) -> dict[str, Any]:
    return {
        "input_dim": int(input_dim),
        "temporal_window_radius": TEMPORAL_WINDOW_RADIUS,
        "temporal_window_size": TEMPORAL_WINDOW_SIZE,
        "model_dim": TEMPORAL_MODEL_DIM,
        "num_heads": TEMPORAL_NUM_HEADS,
        "feedforward_dim": TEMPORAL_FF_DIM,
        "encoder_dropout": TEMPORAL_ENCODER_DROPOUT,
        "encoder_layers": TEMPORAL_ENCODER_LAYERS,
        "projection_dropout": TEMPORAL_PROJECTION_DROPOUT,
        "verb_head_dropout": VERB_HEAD_DROPOUT,
        "noun_head_dropout": NOUN_HEAD_DROPOUT,
        "action_head_dropout": ACTION_HEAD_DROPOUT,
        "relevance_head_dropout": RELEVANCE_HEAD_DROPOUT,
        "batch_size": TEMPORAL_BATCH_SIZE,
        "epochs": TEMPORAL_EPOCHS,
        "learning_rate": TEMPORAL_LEARNING_RATE,
        "weight_decay": TEMPORAL_WEIGHT_DECAY,
        "early_stop_patience": TEMPORAL_EARLY_STOP_PATIENCE,
        "num_workers": TEMPORAL_NUM_WORKERS,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "train_relevant_only": not TRAIN_RELEVANCE_ALL_BLOCKS,
        "train_relevance_all_blocks": TRAIN_RELEVANCE_ALL_BLOCKS,
        "relevance_loss_weight": RELEVANCE_LOSS_WEIGHT,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "min_class_count": MIN_CLASS_COUNT,
        "use_pair_prior": USE_PAIR_PRIOR,
        "pair_prior_weight": PAIR_PRIOR_WEIGHT,
        "pair_prior_smoothing": PAIR_PRIOR_SMOOTHING,
        "use_action_head_for_prediction": USE_ACTION_HEAD_FOR_PREDICTION,
        "use_action_ensemble": USE_ACTION_ENSEMBLE,
        "use_seed_ensemble": USE_SEED_ENSEMBLE,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "verb_head_weight": VERB_HEAD_WEIGHT,
        "noun_head_weight": NOUN_HEAD_WEIGHT,
        "action_head_weight": ACTION_HEAD_WEIGHT,
        "action_loss_weight": ACTION_LOSS_WEIGHT,
        "num_actions": int(num_actions),
    }


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_temporal_encodings(x: Any, y: dict[str, Any]) -> dict[str, Any]:
    temporal_x, temporal_mask = build_temporal_windows(x, y["meta"])
    valid_action_mask = valid_action_mask_from_labels(y)
    if not valid_action_mask.any():
        raise RuntimeError("No valid relevant action blocks are available for action label encoding.")

    verb_to_encoded, encoded_to_verb = make_label_encoding(y["verb"][valid_action_mask])
    noun_to_encoded, encoded_to_noun = make_label_encoding(y["noun"][valid_action_mask])

    y_verb_encoded = np.full(len(y["verb"]), -1, dtype=np.int64)
    y_noun_encoded = np.full(len(y["noun"]), -1, dtype=np.int64)
    y_verb_encoded[valid_action_mask] = encode_labels(y["verb"][valid_action_mask], verb_to_encoded)
    y_noun_encoded[valid_action_mask] = encode_labels(y["noun"][valid_action_mask], noun_to_encoded)
    action_pairs = [
        (int(verb_id), int(noun_id))
        for verb_id, noun_id in zip(
            y["verb"][valid_action_mask],
            y["noun"][valid_action_mask],
            strict=True,
        )
    ]
    unique_action_pairs = sorted(set(action_pairs))
    action_to_encoded = {pair: index for index, pair in enumerate(unique_action_pairs)}
    encoded_to_action = {index: pair for pair, index in action_to_encoded.items()}
    y_action_encoded = np.full(len(y["verb"]), -1, dtype=np.int64)
    y_action_encoded[valid_action_mask] = np.asarray(
        [
            action_to_encoded[(int(verb_id), int(noun_id))]
            for verb_id, noun_id in zip(
                y["verb"][valid_action_mask],
                y["noun"][valid_action_mask],
                strict=True,
            )
        ],
        dtype=np.int64,
    )

    return {
        "temporal_x": temporal_x,
        "temporal_mask": temporal_mask,
        "verb_to_encoded": verb_to_encoded,
        "encoded_to_verb": encoded_to_verb,
        "noun_to_encoded": noun_to_encoded,
        "encoded_to_noun": encoded_to_noun,
        "encoded_to_action": encoded_to_action,
        "y_verb_encoded": y_verb_encoded,
        "y_noun_encoded": y_noun_encoded,
        "y_action_encoded": y_action_encoded,
        "y_relevant": np.asarray(y["relevant"], dtype=np.int64),
        "valid_action_mask": valid_action_mask,
    }


def make_temporal_loader(
    indices: Any,
    encodings: dict[str, Any],
    shuffle: bool,
    seed: int | None = None,
) -> Any:
    dataset = TensorDataset(
        torch.from_numpy(encodings["temporal_x"][indices]).float(),
        torch.from_numpy(encodings["temporal_mask"][indices]).bool(),
        torch.from_numpy(encodings["y_verb_encoded"][indices]).long(),
        torch.from_numpy(encodings["y_noun_encoded"][indices]).long(),
        torch.from_numpy(encodings["y_action_encoded"][indices]).long(),
        torch.from_numpy(encodings["y_relevant"][indices]).long(),
        torch.from_numpy(encodings["valid_action_mask"][indices]).bool(),
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=TEMPORAL_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=TEMPORAL_NUM_WORKERS,
        generator=generator,
    )


def make_temporal_model(x: Any, encodings: dict[str, Any], device: torch.device) -> Any:
    return TemporalWindowClassifier(
        input_dim=x.shape[1],
        num_verbs=len(encodings["encoded_to_verb"]),
        num_nouns=len(encodings["encoded_to_noun"]),
        num_actions=len(encodings["encoded_to_action"]),
    ).to(device)


def save_label_maps(model_dir: Any, encodings: dict[str, Any]) -> None:
    write_json(model_dir / "label_maps.json", {
        "encoded_to_verb": {str(k): v for k, v in encodings["encoded_to_verb"].items()},
        "encoded_to_noun": {str(k): v for k, v in encodings["encoded_to_noun"].items()},
        "encoded_to_action": {
            str(k): [int(verb_id), int(noun_id)]
            for k, (verb_id, noun_id) in encodings["encoded_to_action"].items()
        },
    })


def compute_temporal_batch_loss(
    verb_loss_fn: Any,
    noun_loss_fn: Any,
    action_loss_fn: Any,
    relevance_loss_fn: Any,
    verb_logits: Any,
    noun_logits: Any,
    action_logits: Any,
    relevance_logits: Any,
    batch_verb: Any,
    batch_noun: Any,
    batch_action: Any,
    batch_relevant: Any,
    batch_valid_action: Any,
) -> Any:
    relevance_loss = relevance_loss_fn(relevance_logits, batch_relevant)
    if batch_valid_action.any():
        verb_loss = verb_loss_fn(verb_logits[batch_valid_action], batch_verb[batch_valid_action])
        noun_loss = noun_loss_fn(noun_logits[batch_valid_action], batch_noun[batch_valid_action])
        action_loss = action_loss_fn(
            action_logits[batch_valid_action],
            batch_action[batch_valid_action],
        )
    else:
        zero = action_logits.sum() * 0.0
        verb_loss = zero
        noun_loss = zero
        action_loss = zero
    return (
        RELEVANCE_LOSS_WEIGHT * relevance_loss
        + verb_loss
        + noun_loss
        + ACTION_LOSS_WEIGHT * action_loss
    )


def predict_actions_from_all_heads(
    verb_logits: Any,
    noun_logits: Any,
    action_logits: Any,
    encodings: dict[str, Any],
    pair_prior: Any | None = None,
) -> tuple[Any, Any]:
    encoded_to_action = encodings["encoded_to_action"]
    verb_to_encoded = encodings["verb_to_encoded"]
    noun_to_encoded = encodings["noun_to_encoded"]

    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    verb_log_probs = torch.log_softmax(verb_logits, dim=-1)
    noun_log_probs = torch.log_softmax(noun_logits, dim=-1)

    action_scores = ACTION_HEAD_WEIGHT * action_log_probs.clone()
    for encoded_action, (verb_id, noun_id) in encoded_to_action.items():
        verb_index = verb_to_encoded[int(verb_id)]
        noun_index = noun_to_encoded[int(noun_id)]
        action_scores[:, encoded_action] += (
            VERB_HEAD_WEIGHT * verb_log_probs[:, verb_index]
            + NOUN_HEAD_WEIGHT * noun_log_probs[:, noun_index]
        )
        if USE_PAIR_PRIOR and pair_prior is not None:
            action_scores[:, encoded_action] += (
                PAIR_PRIOR_WEIGHT * pair_prior[verb_index, noun_index]
            )

    best_actions = action_scores.argmax(dim=1).cpu().numpy().tolist()
    batch_pred_pairs = [encoded_to_action[int(action)] for action in best_actions]
    return (
        [verb_to_encoded[int(verb_id)] for verb_id, _ in batch_pred_pairs],
        [noun_to_encoded[int(noun_id)] for _, noun_id in batch_pred_pairs],
    )


def predict_single_model(
    model: Any,
    loader: Any,
    encodings: dict[str, Any],
    pair_prior: Any,
    device: torch.device,
) -> tuple[Any, Any, Any, Any]:
    pred_verbs = []
    pred_nouns = []
    pred_relevant = []
    p_relevant = []
    encoded_to_action = encodings["encoded_to_action"]
    encoded_to_verb = encodings["encoded_to_verb"]
    encoded_to_noun = encodings["encoded_to_noun"]
    verb_to_encoded = encodings["verb_to_encoded"]
    noun_to_encoded = encodings["noun_to_encoded"]

    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask, _, _, _, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            verb_logits, noun_logits, action_logits, relevance_logits = model(batch_x, batch_mask)
            relevance_probs = torch.softmax(relevance_logits, dim=-1)[:, 1]
            pred_relevant.extend((relevance_probs >= RELEVANCE_THRESHOLD).long().cpu().numpy().tolist())
            p_relevant.extend(relevance_probs.cpu().numpy().tolist())
            if USE_ACTION_HEAD_FOR_PREDICTION:
                batch_pred_verbs, batch_pred_nouns = predict_actions_from_all_heads(
                    verb_logits,
                    noun_logits,
                    action_logits,
                    encodings,
                    pair_prior,
                )
            elif USE_ACTION_ENSEMBLE:
                num_nouns = len(encoded_to_noun)
                verb_log_probs = torch.log_softmax(verb_logits, dim=-1)
                noun_log_probs = torch.log_softmax(noun_logits, dim=-1)
                action_head_log_probs = torch.log_softmax(action_logits, dim=-1)
                action_scores = (
                    verb_log_probs[:, :, None]
                    + noun_log_probs[:, None, :]
                    + PAIR_PRIOR_WEIGHT * pair_prior[None, :, :]
                )
                flat_action_scores = action_scores.flatten(start_dim=1)
                for encoded_action, (verb_id, noun_id) in encoded_to_action.items():
                    flat_index = (
                        verb_to_encoded[int(verb_id)] * num_nouns
                        + noun_to_encoded[int(noun_id)]
                    )
                    flat_action_scores[:, flat_index] += (
                        ACTION_HEAD_WEIGHT * action_head_log_probs[:, encoded_action]
                    )
                best_actions = flat_action_scores.argmax(dim=1)
                batch_pred_verbs = best_actions // num_nouns
                batch_pred_nouns = best_actions % num_nouns
            elif USE_PAIR_PRIOR:
                num_nouns = len(encoded_to_noun)
                verb_log_probs = torch.log_softmax(verb_logits, dim=-1)
                noun_log_probs = torch.log_softmax(noun_logits, dim=-1)
                action_scores = (
                    verb_log_probs[:, :, None]
                    + noun_log_probs[:, None, :]
                    + PAIR_PRIOR_WEIGHT * pair_prior[None, :, :]
                )
                best_actions = action_scores.flatten(start_dim=1).argmax(dim=1)
                batch_pred_verbs = best_actions // num_nouns
                batch_pred_nouns = best_actions % num_nouns
            else:
                batch_pred_verbs = verb_logits.argmax(dim=1)
                batch_pred_nouns = noun_logits.argmax(dim=1)

            if torch.is_tensor(batch_pred_verbs):
                pred_verbs.extend(batch_pred_verbs.cpu().numpy().tolist())
                pred_nouns.extend(batch_pred_nouns.cpu().numpy().tolist())
            else:
                pred_verbs.extend(batch_pred_verbs)
                pred_nouns.extend(batch_pred_nouns)

    return (
        decode_labels(pred_verbs, encoded_to_verb),
        decode_labels(pred_nouns, encoded_to_noun),
        np.asarray(pred_relevant, dtype=np.int64),
        np.asarray(p_relevant, dtype=np.float32),
    )


def predict_ensemble(
    models: list[Any],
    loader: Any,
    encodings: dict[str, Any],
    pair_prior: Any,
    device: torch.device,
) -> tuple[Any, Any, Any, Any]:
    pred_verbs = []
    pred_nouns = []
    pred_relevant = []
    p_relevant = []
    for model in models:
        model.eval()

    with torch.no_grad():
        for batch_x, batch_mask, _, _, _, _, _ in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            verb_logits_from_all_models = []
            noun_logits_from_all_models = []
            action_logits_from_all_models = []
            relevance_logits_from_all_models = []
            for model in models:
                verb_logits, noun_logits, action_logits, relevance_logits = model(batch_x, batch_mask)
                verb_logits_from_all_models.append(verb_logits)
                noun_logits_from_all_models.append(noun_logits)
                action_logits_from_all_models.append(action_logits)
                relevance_logits_from_all_models.append(relevance_logits)

            avg_verb_logits = torch.stack(verb_logits_from_all_models, dim=0).mean(dim=0)
            avg_noun_logits = torch.stack(noun_logits_from_all_models, dim=0).mean(dim=0)
            avg_action_logits = torch.stack(action_logits_from_all_models, dim=0).mean(dim=0)
            avg_relevance_logits = torch.stack(relevance_logits_from_all_models, dim=0).mean(dim=0)
            relevance_probs = torch.softmax(avg_relevance_logits, dim=-1)[:, 1]
            pred_relevant.extend((relevance_probs >= RELEVANCE_THRESHOLD).long().cpu().numpy().tolist())
            p_relevant.extend(relevance_probs.cpu().numpy().tolist())
            batch_pred_verbs, batch_pred_nouns = predict_actions_from_all_heads(
                avg_verb_logits,
                avg_noun_logits,
                avg_action_logits,
                encodings,
                pair_prior,
            )
            pred_verbs.extend(batch_pred_verbs)
            pred_nouns.extend(batch_pred_nouns)

    return (
        decode_labels(pred_verbs, encodings["encoded_to_verb"]),
        decode_labels(pred_nouns, encodings["encoded_to_noun"]),
        np.asarray(pred_relevant, dtype=np.int64),
        np.asarray(p_relevant, dtype=np.float32),
    )


def train_single_seed(
    seed: int,
    x: Any,
    y: dict[str, Any],
    splits: dict[str, Any],
    encodings: dict[str, Any],
    device: torch.device,
    output_root: Any,
) -> tuple[Any, dict[str, Any]]:
    set_training_seed(seed)

    model = make_temporal_model(x, encodings, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TEMPORAL_LEARNING_RATE,
        weight_decay=TEMPORAL_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    verb_loss_fn = nn.CrossEntropyLoss()
    noun_loss_fn = nn.CrossEntropyLoss()
    action_loss_fn = nn.CrossEntropyLoss()
    relevance_loss_fn = nn.CrossEntropyLoss()

    train_loader = make_temporal_loader(splits["train"], encodings, shuffle=True, seed=seed)
    val_loader = make_temporal_loader(splits["val"], encodings, shuffle=False)

    if getattr(output_root, "name", "") == "temporal_window":
        model_dir = output_root
    else:
        model_dir = output_root / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "classifier.pt"
    best_val_loss = math.inf
    stale_epochs = 0
    history = []

    for epoch in progress(range(1, TEMPORAL_EPOCHS + 1), desc=f"Training seed {seed}"):
        model.train()
        train_loss = 0.0
        train_count = 0
        for (
            batch_x,
            batch_mask,
            batch_verb,
            batch_noun,
            batch_action,
            batch_relevant,
            batch_valid_action,
        ) in progress(
            train_loader,
            desc=f"Seed {seed} epoch {epoch:03d} train",
            leave=False,
        ):
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_verb = batch_verb.to(device)
            batch_noun = batch_noun.to(device)
            batch_action = batch_action.to(device)
            batch_relevant = batch_relevant.to(device)
            batch_valid_action = batch_valid_action.to(device)

            optimizer.zero_grad(set_to_none=True)
            verb_logits, noun_logits, action_logits, relevance_logits = model(batch_x, batch_mask)
            loss = compute_temporal_batch_loss(
                verb_loss_fn,
                noun_loss_fn,
                action_loss_fn,
                relevance_loss_fn,
                verb_logits,
                noun_logits,
                action_logits,
                relevance_logits,
                batch_verb,
                batch_noun,
                batch_action,
                batch_relevant,
                batch_valid_action,
            )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for (
                batch_x,
                batch_mask,
                batch_verb,
                batch_noun,
                batch_action,
                batch_relevant,
                batch_valid_action,
            ) in progress(
                val_loader,
                desc=f"Seed {seed} epoch {epoch:03d} val",
                leave=False,
            ):
                batch_x = batch_x.to(device)
                batch_mask = batch_mask.to(device)
                batch_verb = batch_verb.to(device)
                batch_noun = batch_noun.to(device)
                batch_action = batch_action.to(device)
                batch_relevant = batch_relevant.to(device)
                batch_valid_action = batch_valid_action.to(device)
                verb_logits, noun_logits, action_logits, relevance_logits = model(batch_x, batch_mask)
                loss = compute_temporal_batch_loss(
                    verb_loss_fn,
                    noun_loss_fn,
                    action_loss_fn,
                    relevance_loss_fn,
                    verb_logits,
                    noun_logits,
                    action_logits,
                    relevance_logits,
                    batch_verb,
                    batch_noun,
                    batch_action,
                    batch_relevant,
                    batch_valid_action,
                )
                val_loss += float(loss.item()) * len(batch_x)
                val_count += len(batch_x)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_loss / max(val_count, 1),
        }
        scheduler.step(epoch_record["val_loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record["learning_rate"] = current_lr
        history.append(epoch_record)
        print(
            f"temporal seed {seed} epoch {epoch:03d}: "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"lr={current_lr:.6f}"
        )

        if epoch_record["val_loss"] < best_val_loss:
            best_val_loss = epoch_record["val_loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoded_to_verb": encodings["encoded_to_verb"],
                    "encoded_to_noun": encodings["encoded_to_noun"],
                    "encoded_to_action": {
                        str(k): [int(verb_id), int(noun_id)]
                        for k, (verb_id, noun_id) in encodings["encoded_to_action"].items()
                    },
                    "config": temporal_config(x.shape[1], len(encodings["encoded_to_action"])),
                    "seed": seed,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= TEMPORAL_EARLY_STOP_PATIENCE:
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    pair_prior = build_pair_prior(
        encodings["y_verb_encoded"],
        encodings["y_noun_encoded"],
        y["relevant"],
        splits["train"],
        len(encodings["encoded_to_verb"]),
        len(encodings["encoded_to_noun"]),
        device,
    )
    pred_verb, pred_noun, pred_relevant, _ = predict_single_model(
        model,
        val_loader,
        encodings,
        pair_prior,
        device,
    )
    val_metrics = compute_pipeline_metrics(
        y,
        splits["val"],
        pred_verb,
        pred_noun,
        pred_relevant,
    )
    metrics = {
        "seed": seed,
        "best_model_path": str(best_path),
        "best_val_loss": best_val_loss,
        "val": val_metrics,
        "history": history,
    }
    return best_path, metrics


def load_temporal_model(model_path: Any, x: Any, encodings: dict[str, Any], device: torch.device) -> Any:
    model = make_temporal_model(x, encodings, device)
    checkpoint = torch.load(model_path, map_location=device)
    if not any(key.startswith("relevance_head.") for key in checkpoint["model_state_dict"]):
        raise RuntimeError(
            f"{model_path} was created before the temporal relevance head was added. "
            "Retrain the temporal classifier to produce a compatible checkpoint."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def train_temporal_classifier(
    x: Any,
    y: dict[str, Any],
    splits: dict[str, Any],
    verb_map: dict[int, str],
    noun_map: dict[int, str],
    run_timestamp: str,
) -> dict[str, Any]:
    set_training_seed(RANDOM_SEED)
    device = get_torch_device()
    print(f"Using PyTorch device: {device}")
    run_name = "temporal_window_ensemble" if USE_SEED_ENSEMBLE else "temporal_window"
    classifier_name = "temporal_window_seed_ensemble" if USE_SEED_ENSEMBLE else "temporal_window"
    report_dir = REPORTS_DIR / run_name
    prediction_dir = PREDICTIONS_DIR / run_name
    model_root = MODELS_DIR / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    encodings = build_temporal_encodings(x, y)
    pair_prior = build_pair_prior(
        encodings["y_verb_encoded"],
        encodings["y_noun_encoded"],
        y["relevant"],
        splits["train"],
        len(encodings["encoded_to_verb"]),
        len(encodings["encoded_to_noun"]),
        device,
    )

    result: dict[str, Any] = {
        "classifier": classifier_name,
        "device": str(device),
        "run_timestamp": run_timestamp,
        "seeds": ENSEMBLE_SEEDS if USE_SEED_ENSEMBLE else [RANDOM_SEED],
        "per_seed": {},
        "ensemble": {},
        "config": temporal_config(x.shape[1], len(encodings["encoded_to_action"])),
    }

    if USE_SEED_ENSEMBLE:
        best_paths = []
        for seed in ENSEMBLE_SEEDS:
            best_path, seed_metrics = train_single_seed(
                seed,
                x,
                y,
                splits,
                encodings,
                device,
                model_root,
            )
            best_paths.append(best_path)
            model = load_temporal_model(best_path, x, encodings, device)
            test_loader = make_temporal_loader(splits["test"], encodings, shuffle=False)
            pred_verb, pred_noun, pred_relevant, _ = predict_single_model(
                model,
                test_loader,
                encodings,
                pair_prior,
                device,
            )
            seed_metrics["test"] = compute_pipeline_metrics(
                y,
                splits["test"],
                pred_verb,
                pred_noun,
                pred_relevant,
            )
            result["per_seed"][str(seed)] = seed_metrics

        save_label_maps(model_root, encodings)
        models = [
            load_temporal_model(best_path, x, encodings, device)
            for best_path in best_paths
        ]
        for split_name, indices in (("val", splits["val"]), ("test", splits["test"])):
            loader = make_temporal_loader(indices, encodings, shuffle=False)
            pred_verb, pred_noun, pred_relevant, p_relevant = predict_ensemble(
                models,
                loader,
                encodings,
                pair_prior,
                device,
            )
            result["ensemble"][split_name] = compute_pipeline_metrics(
                y,
                indices,
                pred_verb,
                pred_noun,
                pred_relevant,
            )
            if split_name == "test":
                test_metas = [y["meta"][int(i)] for i in indices]
                write_predictions_csv(
                    prediction_dir / f"test_block_predictions_{run_timestamp}.csv",
                    test_metas,
                    pred_verb,
                    pred_noun,
                    pred_relevant,
                    verb_map,
                    noun_map,
                    p_relevant=p_relevant,
                )
    else:
        single_model_dir = MODELS_DIR / "temporal_window"
        best_path, seed_metrics = train_single_seed(
            RANDOM_SEED,
            x,
            y,
            splits,
            encodings,
            device,
            single_model_dir,
        )
        save_label_maps(single_model_dir, encodings)
        model = load_temporal_model(best_path, x, encodings, device)
        result["history"] = seed_metrics["history"]
        result["splits"] = {}
        for split_name, indices in (("val", splits["val"]), ("test", splits["test"])):
            loader = make_temporal_loader(indices, encodings, shuffle=False)
            pred_verb, pred_noun, pred_relevant, p_relevant = predict_single_model(
                model,
                loader,
                encodings,
                pair_prior,
                device,
            )
            result["splits"][split_name] = compute_pipeline_metrics(
                y,
                indices,
                pred_verb,
                pred_noun,
                pred_relevant,
            )
            if split_name == "test":
                test_metas = [y["meta"][int(i)] for i in indices]
                write_predictions_csv(
                    prediction_dir / f"test_block_predictions_{run_timestamp}.csv",
                    test_metas,
                    pred_verb,
                    pred_noun,
                    pred_relevant,
                    verb_map,
                    noun_map,
                    p_relevant=p_relevant,
                )
        result["per_seed"][str(RANDOM_SEED)] = seed_metrics

    write_json(report_dir / f"metrics_{run_timestamp}.json", result)
    return result


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    run_timestamp = make_run_timestamp()
    run_name = "temporal_window_ensemble" if USE_SEED_ENSEMBLE else "temporal_window"

    for directory in (MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / run_name).mkdir(parents=True, exist_ok=True)

    verb_map = read_class_map(VERB_CLASSES_CSV, "verb_id")
    noun_map = read_class_map(NOUN_CLASSES_CSV, "noun_id")
    annotations_by_video = read_annotations(ANNOTATIONS_CSV)
    x, y = build_dataset(annotations_by_video)
    if TRAIN_RELEVANCE_ALL_BLOCKS:
        y["filter_summary"] = {
            "source_blocks": int(len(x)),
            "kept_blocks": int(len(x)),
            "dropped_blocks": 0,
            "mode": "all_blocks_for_relevance_training",
        }
    else:
        x, y = filter_relevant_non_singleton_blocks(x, y)
    splits = stratified_block_split(y["action"])
    dataset_summary = build_dataset_summary(x, y, splits)
    dataset_summary["filter_summary"] = y["filter_summary"]
    dataset_summary["temporal_window"] = {
        "radius": TEMPORAL_WINDOW_RADIUS,
        "size": TEMPORAL_WINDOW_SIZE,
        "padding": "zero for missing neighbors or video boundaries",
        "center_only_prediction": True,
    }
    dataset_summary["split_note"] = (
        "Splits are block-level over full temporal blocks when "
        "TRAIN_RELEVANCE_ALL_BLOCKS is enabled; action losses are masked to "
        "valid relevant action blocks."
    )
    print_dataset_summary(dataset_summary)
    dataset_summary["run_timestamp"] = run_timestamp
    write_json(
        REPORTS_DIR / run_name / f"dataset_summary_{run_timestamp}.json",
        dataset_summary,
    )

    results = [train_temporal_classifier(x, y, splits, verb_map, noun_map, run_timestamp)]

    write_json(
        REPORTS_DIR / run_name / f"all_metrics_{run_timestamp}.json",
        {"run_timestamp": run_timestamp, "results": results},
    )
    print(f"Finished. Reports: {REPORTS_DIR / run_name}. Predictions: {PREDICTIONS_DIR / run_name}.")


if __name__ == "__main__":
    main()
