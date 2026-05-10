import ast
import csv
import json
import logging
import pickle
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ANNOTATIONS_CSV = "../data/annotations_train_test.csv"
EMBEDDINGS_DIR = "../embeddings"
EMBEDDINGS_SUBDIRS = ("train", "test")
OUTPUT_DIR = "runs/temporal_vjepa2"

BLOCK_SIZE = 64
MIN_OVERLAP_RATIO = 0.25

SEQ_LEN = 32
STRIDE = 16

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42

HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2
MAX_LEN = 512

GRAD_CLIP = 1.0
USE_RELEVANT_ONLY = False

ACTION_THRESHOLD = 0.4
MERGE_GAP = 1

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

VIDEO_ID_COL = "video_id"
START_FRAME_COL = "start_frame"
STOP_FRAME_COL = "stop_frame"
VERB_CLASS_COL = "verb_class"
VERB_NAME_COL = "verb"
NOUN_NAME_COL = "noun"
NOUN_CLASS_COL = "noun_class"
NOUN_CLASS_LIST_COL = "all_noun_classes"
RELEVANT_COL = "relevant"


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_embedding_file(path):
    if path.suffix != ".pkl":
        raise ValueError(f"Expected a .pkl embedding file, got {path}.")

    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(f"Expected {path} to be a pickled dict with an 'embeddings' key.")

    array = payload["embeddings"]
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected embedding array [num_blocks, embedding_dim] in {path}, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def load_embedding_index(embeddings_dir):
    train_dir = embeddings_dir / "train"
    test_dir = embeddings_dir / "test"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected embedding train directory at {train_dir}.")
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected embedding test directory at {test_dir}.")

    paths = sorted(train_dir.glob("*.pkl")) + sorted(test_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl embedding files found in {train_dir} or {test_dir}.")

    by_video = {}
    duplicates = defaultdict(list)
    for path in paths:
        video_id = path.stem
        if video_id in by_video:
            duplicates[video_id].append(path)
            continue
        by_video[video_id] = path

    if duplicates:
        examples = ", ".join(f"{vid}: {by_video[vid]} and {extra[0]}" for vid, extra in list(duplicates.items())[:5])
        raise ValueError(f"Multiple embedding files found for the same video id: {examples}")
    return by_video, paths


def parse_first_noun_class(value):
    if pd.isna(value):
        return -100
    if isinstance(value, (list, tuple)):
        values = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return -100
        try:
            values = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse {NOUN_CLASS_LIST_COL} value {value!r} as a list.") from exc
    if not values:
        return -100
    first_value = values[0]
    if pd.isna(first_value):
        return -100
    return int(first_value)


def validate_and_prepare_annotations(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found at {csv_path}.")

    df = pd.read_csv(csv_path)
    logging.info("Loaded annotations: %s rows from %s", len(df), csv_path)
    logging.info("Annotation columns: %s", list(df.columns))

    required_columns = [VIDEO_ID_COL, START_FRAME_COL, STOP_FRAME_COL, VERB_CLASS_COL]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Annotation CSV {csv_path} is missing required columns {missing}. "
            f"Expected video id, start frame, stop frame, and verb class columns."
        )

    if NOUN_CLASS_COL in df.columns:
        df[NOUN_CLASS_COL] = df[NOUN_CLASS_COL].fillna(-100).astype(int)
        noun_source = NOUN_CLASS_COL
    elif NOUN_CLASS_LIST_COL in df.columns:
        df[NOUN_CLASS_COL] = df[NOUN_CLASS_LIST_COL].apply(parse_first_noun_class).astype(int)
        noun_source = NOUN_CLASS_LIST_COL
    else:
        raise ValueError(
            f"Annotation CSV {csv_path} has no {NOUN_CLASS_COL!r} column and no list-like "
            f"{NOUN_CLASS_LIST_COL!r} column to derive noun classes from."
        )

    for col in [START_FRAME_COL, STOP_FRAME_COL, VERB_CLASS_COL]:
        if df[col].isna().any():
            raise ValueError(f"Annotation CSV {csv_path} contains missing values in required column {col!r}.")
        df[col] = df[col].astype(int)

    if USE_RELEVANT_ONLY:
        if RELEVANT_COL not in df.columns:
            logging.warning("USE_RELEVANT_ONLY is True but column %r is absent; no relevance filter applied.", RELEVANT_COL)
        else:
            before = len(df)
            df = df[df[RELEVANT_COL].astype(bool)].copy()
            logging.info("Filtered to relevant annotations: %s -> %s rows", before, len(df))

    logging.info("Using noun class source: %s", noun_source)
    present_label_columns = [
        col for col in [VERB_CLASS_COL, NOUN_CLASS_COL, NOUN_CLASS_LIST_COL, VERB_NAME_COL, NOUN_NAME_COL, RELEVANT_COL]
        if col in df.columns
    ]
    logging.info("Relevant label columns present: %s", present_label_columns)
    return df


def analyze_workspace():
    csv_path = resolve_path(ANNOTATIONS_CSV)
    embeddings_dir = resolve_path(EMBEDDINGS_DIR)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found at {embeddings_dir}.")

    df = validate_and_prepare_annotations(csv_path)
    embedding_files, all_paths = load_embedding_index(embeddings_dir)
    first_path = all_paths[0]
    first_array = load_embedding_file(first_path)

    logging.info("Discovered annotation CSV: %s", csv_path)
    logging.info("Using embeddings from %s/train and %s/test as one combined video index.", embeddings_dir, embeddings_dir)
    logging.info("Embedding format: pickled dict files named <video_id>.pkl with key 'embeddings'.")
    logging.info("Embedding file counts: train=%s test=%s", len(list((embeddings_dir / "train").glob("*.pkl"))), len(list((embeddings_dir / "test").glob("*.pkl"))))
    logging.info("Sample embedding shape: %s from %s", tuple(first_array.shape), first_path)
    if first_array.shape[0] > 0:
        logging.info("Sample embedding dim: %s", first_array.shape[1])

    annotated_video_ids = set(df[VIDEO_ID_COL].astype(str).unique())
    missing_videos = sorted(annotated_video_ids - set(embedding_files.keys()))
    if missing_videos:
        logging.warning(
            "Missing embeddings for %s annotated videos; examples: %s",
            len(missing_videos),
            missing_videos[:20],
        )

    return csv_path, embeddings_dir, df, embedding_files, first_array.shape[1]


def build_id_to_name(df, class_col, name_col, max_classes):
    id_to_name = {str(i): str(i) for i in range(max_classes)}
    if name_col not in df.columns:
        return id_to_name

    usable = df[[class_col, name_col]].dropna()
    usable = usable[usable[class_col].astype(int) >= 0]
    for class_id, group in usable.groupby(class_col):
        class_id = int(class_id)
        if class_id >= max_classes:
            continue
        names = group[name_col].astype(str)
        if len(names) > 0:
            id_to_name[str(class_id)] = names.mode().iloc[0]
    return id_to_name


def make_window_starts(num_blocks):
    if num_blocks <= 0:
        return []
    starts = list(range(0, max(num_blocks - SEQ_LEN + 1, 1), STRIDE))
    last_needed_start = max(num_blocks - SEQ_LEN, 0)
    if starts[-1] != last_needed_start:
        starts.append(last_needed_start)
    return sorted(set(starts))


def build_block_labels(video_annotations, num_blocks):
    verb_targets = np.full(num_blocks, -100, dtype=np.int64)
    noun_targets = np.full(num_blocks, -100, dtype=np.int64)
    action_targets = np.zeros(num_blocks, dtype=np.float32)

    records = video_annotations.to_dict("records")
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        block_end = block_start + BLOCK_SIZE - 1
        best_overlap = 0
        best_row = None

        for row in records:
            start_frame = int(row[START_FRAME_COL])
            stop_frame = int(row[STOP_FRAME_COL])
            overlap = max(
                0,
                min(block_end, stop_frame) - max(block_start, start_frame) + 1,
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = row

        if best_row is not None and best_overlap / BLOCK_SIZE >= MIN_OVERLAP_RATIO:
            verb_targets[block_idx] = int(best_row[VERB_CLASS_COL])
            noun_class = int(best_row[NOUN_CLASS_COL])
            noun_targets[block_idx] = noun_class if noun_class >= 0 else -100
            action_targets[block_idx] = 1.0

    return verb_targets, noun_targets, action_targets


class TemporalEmbeddingDataset(Dataset):
    def __init__(self, annotations_df, embedding_files, video_ids):
        self.annotations_df = annotations_df
        self.embedding_files = embedding_files
        self.video_ids = sorted(str(video_id) for video_id in video_ids)
        self.video_data = {}
        self.windows = []

        for video_id in self.video_ids:
            path = self.embedding_files.get(video_id)
            if path is None:
                logging.warning("Skipping %s because no embedding file was found.", video_id)
                continue

            embeddings = load_embedding_file(path)
            video_annotations = self.annotations_df[self.annotations_df[VIDEO_ID_COL].astype(str) == video_id]
            verb_targets, noun_targets, action_targets = build_block_labels(video_annotations, len(embeddings))
            self.video_data[video_id] = {
                "embeddings": embeddings,
                "verb_targets": verb_targets,
                "noun_targets": noun_targets,
                "action_targets": action_targets,
            }

            for start_idx in make_window_starts(len(embeddings)):
                self.windows.append((video_id, start_idx))

        if not self.windows:
            raise ValueError("Dataset has no temporal windows. Check split video ids and embedding availability.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        video_id, start_idx = self.windows[index]
        data = self.video_data[video_id]
        embeddings = data["embeddings"]
        end_idx = min(start_idx + SEQ_LEN, len(embeddings))
        real_len = end_idx - start_idx
        embedding_dim = embeddings.shape[1]

        window_embeddings = np.zeros((SEQ_LEN, embedding_dim), dtype=np.float32)
        verb_targets = np.full(SEQ_LEN, -100, dtype=np.int64)
        noun_targets = np.full(SEQ_LEN, -100, dtype=np.int64)
        action_targets = np.zeros(SEQ_LEN, dtype=np.float32)
        padding_mask = np.ones(SEQ_LEN, dtype=bool)

        if real_len > 0:
            window_embeddings[:real_len] = embeddings[start_idx:end_idx]
            verb_targets[:real_len] = data["verb_targets"][start_idx:end_idx]
            noun_targets[:real_len] = data["noun_targets"][start_idx:end_idx]
            action_targets[:real_len] = data["action_targets"][start_idx:end_idx]
            padding_mask[:real_len] = False

        return {
            "embeddings": torch.from_numpy(window_embeddings),
            "verb_targets": torch.from_numpy(verb_targets),
            "noun_targets": torch.from_numpy(noun_targets),
            "action_targets": torch.from_numpy(action_targets),
            "padding_mask": torch.from_numpy(padding_mask),
            "video_id": video_id,
            "block_start_idx": int(start_idx),
        }


class TemporalActionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_verbs,
        num_nouns,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.verb_head = nn.Linear(hidden_dim, num_verbs)
        self.noun_head = nn.Linear(hidden_dim, num_nouns)
        self.actionness_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding_mask=None):
        batch_size, seq_len, _ = x.shape

        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pos_embedding.shape[1]}"
            )

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :seq_len]
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)

        return {
            "verb_logits": self.verb_head(h),
            "noun_logits": self.noun_head(h),
            "actionness_logits": self.actionness_head(h).squeeze(-1),
        }


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def zero_loss_like(outputs):
    return outputs["actionness_logits"].sum() * 0.0


def compute_loss(outputs, batch):
    padding_mask = batch["padding_mask"]
    non_padded = ~padding_mask

    if non_padded.any():
        actionness_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["actionness_logits"][non_padded],
            batch["action_targets"][non_padded],
        )
    else:
        actionness_loss = zero_loss_like(outputs)

    verb_mask = batch["verb_targets"] != -100
    if verb_mask.any():
        verb_loss = nn.functional.cross_entropy(
            outputs["verb_logits"].reshape(-1, outputs["verb_logits"].shape[-1]),
            batch["verb_targets"].reshape(-1),
            ignore_index=-100,
        )
    else:
        verb_loss = zero_loss_like(outputs)

    noun_mask = batch["noun_targets"] != -100
    if noun_mask.any():
        noun_loss = nn.functional.cross_entropy(
            outputs["noun_logits"].reshape(-1, outputs["noun_logits"].shape[-1]),
            batch["noun_targets"].reshape(-1),
            ignore_index=-100,
        )
    else:
        noun_loss = zero_loss_like(outputs)

    return actionness_loss + verb_loss + noun_loss


def topk_accuracy(logits, targets, mask, k):
    if not mask.any():
        return None
    _, pred = logits[mask].topk(k, dim=-1)
    correct = pred.eq(targets[mask].unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, num_verbs, num_nouns):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    action_correct = 0
    action_total = 0
    verb_correct = 0
    verb_total = 0
    noun_correct = 0
    noun_total = 0
    joint_correct = 0
    joint_total = 0
    verb_top5_correct = 0
    verb_top5_total = 0
    noun_top5_correct = 0
    noun_top5_total = 0

    for batch in tqdm(loader, desc="validate", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["embeddings"], batch["padding_mask"])
        loss = compute_loss(outputs, batch)
        batch_size = batch["embeddings"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        non_padded = ~batch["padding_mask"]
        action_pred = (torch.sigmoid(outputs["actionness_logits"]) >= 0.5).float()
        action_correct += (action_pred[non_padded] == batch["action_targets"][non_padded]).sum().item()
        action_total += non_padded.sum().item()

        verb_mask = batch["verb_targets"] != -100
        if verb_mask.any():
            verb_pred = outputs["verb_logits"].argmax(dim=-1)
            verb_correct += (verb_pred[verb_mask] == batch["verb_targets"][verb_mask]).sum().item()
            verb_total += verb_mask.sum().item()
            if num_verbs >= 5:
                top5 = outputs["verb_logits"][verb_mask].topk(5, dim=-1).indices
                verb_top5_correct += top5.eq(batch["verb_targets"][verb_mask].unsqueeze(-1)).any(dim=-1).sum().item()
                verb_top5_total += verb_mask.sum().item()

        noun_mask = batch["noun_targets"] != -100
        if noun_mask.any():
            noun_pred = outputs["noun_logits"].argmax(dim=-1)
            noun_correct += (noun_pred[noun_mask] == batch["noun_targets"][noun_mask]).sum().item()
            noun_total += noun_mask.sum().item()
            if num_nouns >= 5:
                top5 = outputs["noun_logits"][noun_mask].topk(5, dim=-1).indices
                noun_top5_correct += top5.eq(batch["noun_targets"][noun_mask].unsqueeze(-1)).any(dim=-1).sum().item()
                noun_top5_total += noun_mask.sum().item()

        joint_mask = verb_mask & noun_mask
        if joint_mask.any():
            verb_pred = outputs["verb_logits"].argmax(dim=-1)
            noun_pred = outputs["noun_logits"].argmax(dim=-1)
            joint = (verb_pred[joint_mask] == batch["verb_targets"][joint_mask]) & (
                noun_pred[joint_mask] == batch["noun_targets"][joint_mask]
            )
            joint_correct += joint.sum().item()
            joint_total += joint_mask.sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "actionness_acc": action_correct / action_total if action_total else None,
        "verb_top1_acc": verb_correct / verb_total if verb_total else None,
        "noun_top1_acc": noun_correct / noun_total if noun_total else None,
        "joint_top1_acc": joint_correct / joint_total if joint_total else None,
        "verb_top5_acc": verb_top5_correct / verb_top5_total if verb_top5_total else None,
        "noun_top5_acc": noun_top5_correct / noun_top5_total if noun_top5_total else None,
    }


def participant_id_from_video(video_id):
    return str(video_id).split("_", 1)[0]


def make_participant_split(video_ids):
    participants = sorted({participant_id_from_video(video_id) for video_id in video_ids})
    rng = random.Random(SEED)
    rng.shuffle(participants)

    n_participants = len(participants)
    if n_participants < 3:
        raise ValueError(f"Need at least 3 participants for train/val/test split, found {n_participants}.")

    n_train = max(1, int(n_participants * 0.8))
    n_val = max(1, int(n_participants * 0.1))
    if n_train + n_val >= n_participants:
        n_train = n_participants - 2
        n_val = 1

    train_participants = set(participants[:n_train])
    val_participants = set(participants[n_train:n_train + n_val])
    test_participants = set(participants[n_train + n_val:])

    split = {"train": [], "val": [], "test": []}
    for video_id in sorted(video_ids):
        participant_id = participant_id_from_video(video_id)
        if participant_id in train_participants:
            split["train"].append(video_id)
        elif participant_id in val_participants:
            split["val"].append(video_id)
        elif participant_id in test_participants:
            split["test"].append(video_id)

    seen = {}
    for split_name, split_videos in split.items():
        for video_id in split_videos:
            participant_id = participant_id_from_video(video_id)
            if participant_id in seen and seen[participant_id] != split_name:
                raise ValueError(f"Participant leakage detected for {participant_id}.")
            seen[participant_id] = split_name

    return split, {
        "train": sorted(train_participants),
        "val": sorted(val_participants),
        "test": sorted(test_participants),
    }


def config_dict():
    return {
        "annotations_csv": ANNOTATIONS_CSV,
        "embeddings_dir": EMBEDDINGS_DIR,
        "output_dir": OUTPUT_DIR,
        "block_size": BLOCK_SIZE,
        "min_overlap_ratio": MIN_OVERLAP_RATIO,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "grad_clip": GRAD_CLIP,
        "use_relevant_only": USE_RELEVANT_ONLY,
        "action_threshold": ACTION_THRESHOLD,
        "merge_gap": MERGE_GAP,
        "device": DEVICE,
    }


def save_checkpoint(path, model, epoch, validation_metrics, num_verbs, num_nouns, embedding_dim, verb_id_to_name, noun_id_to_name):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config_dict(),
        "num_verbs": num_verbs,
        "num_nouns": num_nouns,
        "embedding_dim": embedding_dim,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "verb_id_to_name": verb_id_to_name,
        "noun_id_to_name": noun_id_to_name,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    use_amp = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            outputs = model(batch["embeddings"], batch["padding_mask"])
            loss = compute_loss(outputs, batch)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        scheduler.step()
        batch_size = batch["embeddings"].shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def predict_video(model, embedding_array, video_id, id_to_verb=None, id_to_noun=None):
    model.eval()
    id_to_verb = id_to_verb or {}
    id_to_noun = id_to_noun or {}

    if isinstance(embedding_array, dict):
        embedding_array = embedding_array["embeddings"]
    embeddings = np.asarray(embedding_array, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embedding_array [num_blocks, embedding_dim], got shape {embeddings.shape}.")

    num_blocks = embeddings.shape[0]
    if num_blocks == 0:
        return []

    device = next(model.parameters()).device
    starts = make_window_starts(num_blocks)
    verb_sum = None
    noun_sum = None
    action_sum = None
    counts = np.zeros(num_blocks, dtype=np.float32)

    with torch.no_grad():
        for start_idx in starts:
            end_idx = min(start_idx + SEQ_LEN, num_blocks)
            real_len = end_idx - start_idx
            window = np.zeros((SEQ_LEN, embeddings.shape[1]), dtype=np.float32)
            padding_mask = np.ones(SEQ_LEN, dtype=bool)
            window[:real_len] = embeddings[start_idx:end_idx]
            padding_mask[:real_len] = False

            x = torch.from_numpy(window).unsqueeze(0).to(device)
            mask = torch.from_numpy(padding_mask).unsqueeze(0).to(device)
            outputs = model(x, mask)

            verb_logits = outputs["verb_logits"][0, :real_len].detach().cpu().numpy()
            noun_logits = outputs["noun_logits"][0, :real_len].detach().cpu().numpy()
            action_logits = outputs["actionness_logits"][0, :real_len].detach().cpu().numpy()

            if verb_sum is None:
                verb_sum = np.zeros((num_blocks, verb_logits.shape[-1]), dtype=np.float32)
                noun_sum = np.zeros((num_blocks, noun_logits.shape[-1]), dtype=np.float32)
                action_sum = np.zeros(num_blocks, dtype=np.float32)

            block_slice = slice(start_idx, end_idx)
            verb_sum[block_slice] += verb_logits
            noun_sum[block_slice] += noun_logits
            action_sum[block_slice] += action_logits
            counts[block_slice] += 1.0

    valid = counts > 0
    verb_avg = verb_sum[valid] / counts[valid, None]
    noun_avg = noun_sum[valid] / counts[valid, None]
    action_avg = action_sum[valid] / counts[valid]

    block_events = []
    valid_block_indices = np.where(valid)[0]
    for local_idx, block_idx in enumerate(valid_block_indices):
        action_probability = float(1.0 / (1.0 + np.exp(-action_avg[local_idx])))
        if action_probability < ACTION_THRESHOLD:
            continue

        verb_probs = torch.softmax(torch.from_numpy(verb_avg[local_idx]), dim=-1).numpy()
        noun_probs = torch.softmax(torch.from_numpy(noun_avg[local_idx]), dim=-1).numpy()
        verb_id = int(verb_probs.argmax())
        noun_id = int(noun_probs.argmax())
        verb_name = id_to_verb.get(str(verb_id), id_to_verb.get(verb_id, str(verb_id)))
        noun_name = id_to_noun.get(str(noun_id), id_to_noun.get(noun_id, str(noun_id)))

        block_events.append({
            "video_id": video_id,
            "start_block": int(block_idx),
            "end_block": int(block_idx),
            "start_frame": int(block_idx * BLOCK_SIZE),
            "end_frame": int((block_idx + 1) * BLOCK_SIZE - 1),
            "verb_id": verb_id,
            "verb": str(verb_name),
            "verb_confidence": float(verb_probs[verb_id]),
            "noun_id": noun_id,
            "noun": str(noun_name),
            "noun_confidence": float(noun_probs[noun_id]),
            "action_probability": action_probability,
            "_count": 1,
        })

    merged = []
    for event in block_events:
        if (
            merged
            and event["verb_id"] == merged[-1]["verb_id"]
            and event["noun_id"] == merged[-1]["noun_id"]
            and event["start_block"] - merged[-1]["end_block"] - 1 <= MERGE_GAP
        ):
            previous = merged[-1]
            previous_count = previous["_count"]
            new_count = previous_count + 1
            previous["end_block"] = event["end_block"]
            previous["end_frame"] = event["end_frame"]
            previous["verb_confidence"] = (
                previous["verb_confidence"] * previous_count + event["verb_confidence"]
            ) / new_count
            previous["noun_confidence"] = (
                previous["noun_confidence"] * previous_count + event["noun_confidence"]
            ) / new_count
            previous["action_probability"] = (
                previous["action_probability"] * previous_count + event["action_probability"]
            ) / new_count
            previous["_count"] = new_count
        else:
            merged.append(event)

    for event in merged:
        event.pop("_count", None)
    return merged


def write_llm_prompts(events, output_dir):
    prompt_dir = Path(output_dir) / "llm_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    events_by_video = defaultdict(list)
    for event in events:
        events_by_video[event["video_id"]].append(event)

    for video_id, video_events in events_by_video.items():
        video_events = sorted(video_events, key=lambda item: item["start_frame"])
        lines = [
            "You are given a sequence of detected cooking actions from an egocentric video.",
            "Write a concise, cohesive summary of what happens.",
            "Do not invent actions that are not supported by the detections.",
            "Use uncertainty-aware language when confidence is low.",
            "",
            "Detected actions:",
        ]
        for idx, event in enumerate(video_events, start=1):
            confidence = (
                event["action_probability"] + event["verb_confidence"] + event["noun_confidence"]
            ) / 3.0
            lines.append(
                f"{idx}. frames {event['start_frame']}-{event['end_frame']}: "
                f"{event['verb']} {event['noun']}, confidence {confidence:.2f}"
            )

        safe_video_id = str(video_id).replace("/", "_")
        (prompt_dir / f"{safe_video_id}.txt").write_text("\n".join(lines) + "\n")


def main():
    setup_logging()
    set_seed(SEED)

    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path, embeddings_dir, annotations_df, embedding_files, embedding_dim = analyze_workspace()

    annotated_video_ids = set(annotations_df[VIDEO_ID_COL].astype(str).unique())
    available_video_ids = sorted(annotated_video_ids & set(embedding_files.keys()))
    if not available_video_ids:
        raise ValueError(f"No annotated videos have matching embeddings under {embeddings_dir}.")

    num_verbs = int(annotations_df[VERB_CLASS_COL].max()) + 1
    valid_nouns = annotations_df.loc[annotations_df[NOUN_CLASS_COL] >= 0, NOUN_CLASS_COL]
    if valid_nouns.empty:
        raise ValueError(f"No usable noun classes found in {csv_path}.")
    num_nouns = int(valid_nouns.max()) + 1
    verb_id_to_name = build_id_to_name(annotations_df, VERB_CLASS_COL, VERB_NAME_COL, num_verbs)
    noun_id_to_name = build_id_to_name(annotations_df, NOUN_CLASS_COL, NOUN_NAME_COL, num_nouns)

    split, participant_split = make_participant_split(available_video_ids)
    split_payload = {
        "participants": participant_split,
        "videos": split,
    }
    (output_dir / "participant_split.json").write_text(json.dumps(split_payload, indent=2, sort_keys=True))
    (output_dir / "config.json").write_text(json.dumps(config_dict(), indent=2, sort_keys=True))

    logging.info("Available annotated videos with embeddings: %s", len(available_video_ids))
    logging.info("Participant split sizes: train=%s val=%s test=%s", len(participant_split["train"]), len(participant_split["val"]), len(participant_split["test"]))
    logging.info("Video split sizes: train=%s val=%s test=%s", len(split["train"]), len(split["val"]), len(split["test"]))
    logging.info("num_verbs=%s num_nouns=%s embedding_dim=%s", num_verbs, num_nouns, embedding_dim)

    train_dataset = TemporalEmbeddingDataset(annotations_df, embedding_files, split["train"])
    val_dataset = TemporalEmbeddingDataset(annotations_df, embedding_files, split["val"])
    test_dataset = TemporalEmbeddingDataset(annotations_df, embedding_files, split["test"])

    pin_memory = str(DEVICE).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = TemporalActionClassifier(
        embedding_dim=embedding_dim,
        num_verbs=num_verbs,
        num_nouns=num_nouns,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_actionness_acc",
        "val_verb_top1_acc",
        "val_noun_top1_acc",
        "val_joint_top1_acc",
        "val_verb_top5_acc",
        "val_noun_top5_acc",
        "lr",
    ]

    best_val_loss = float("inf")
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_metrics = evaluate(model, val_loader, DEVICE, num_verbs, num_nouns)
            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_actionness_acc": val_metrics["actionness_acc"],
                "val_verb_top1_acc": val_metrics["verb_top1_acc"],
                "val_noun_top1_acc": val_metrics["noun_top1_acc"],
                "val_joint_top1_acc": val_metrics["joint_top1_acc"],
                "val_verb_top5_acc": val_metrics["verb_top5_acc"],
                "val_noun_top5_acc": val_metrics["noun_top5_acc"],
                "lr": current_lr,
            }
            writer.writerow(row)
            f.flush()

            logging.info(
                "epoch=%03d train_loss=%.4f val_loss=%.4f action_acc=%s verb_acc=%s noun_acc=%s joint_acc=%s",
                epoch,
                train_loss,
                val_metrics["loss"],
                None if val_metrics["actionness_acc"] is None else round(val_metrics["actionness_acc"], 4),
                None if val_metrics["verb_top1_acc"] is None else round(val_metrics["verb_top1_acc"], 4),
                None if val_metrics["noun_top1_acc"] is None else round(val_metrics["noun_top1_acc"], 4),
                None if val_metrics["joint_top1_acc"] is None else round(val_metrics["joint_top1_acc"], 4),
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    output_dir / "best_model.pt",
                    model,
                    epoch,
                    val_metrics,
                    num_verbs,
                    num_nouns,
                    embedding_dim,
                    verb_id_to_name,
                    noun_id_to_name,
                )

            save_checkpoint(
                output_dir / "last_model.pt",
                model,
                epoch,
                val_metrics,
                num_verbs,
                num_nouns,
                embedding_dim,
                verb_id_to_name,
                noun_id_to_name,
            )

    best_checkpoint = load_checkpoint(output_dir / "best_model.pt", DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    all_events = []
    for video_id in tqdm(test_dataset.video_data.keys(), desc="predict test"):
        embeddings = test_dataset.video_data[video_id]["embeddings"]
        events = predict_video(
            model,
            embeddings,
            video_id,
            id_to_verb=best_checkpoint.get("verb_id_to_name"),
            id_to_noun=best_checkpoint.get("noun_id_to_name"),
        )
        all_events.extend(events)

    predictions_path = output_dir / "test_predictions.jsonl"
    with predictions_path.open("w") as f:
        for event in all_events:
            f.write(json.dumps(event) + "\n")

    write_llm_prompts(all_events, output_dir)
    logging.info("Saved best model, last model, metrics, predictions, and LLM prompts under %s", output_dir)


if __name__ == "__main__":
    main()
