import csv
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BlockExample:
    embedding: torch.Tensor
    label: float
    video_id: str
    block_index: int


class BlockEmbeddingDataset(Dataset[BlockExample]):
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        video_ids: list[str],
        block_indices: list[int],
        indices: list[int],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.labels = labels
        self.video_ids = video_ids
        self.block_indices = block_indices
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> BlockExample:
        idx = self.indices[item]
        embedding = self.embeddings[idx]
        if self.mean is not None and self.std is not None:
            embedding = (embedding - self.mean) / self.std
        return BlockExample(
            embedding=embedding,
            label=float(self.labels[idx].item()),
            video_id=self.video_ids[idx],
            block_index=self.block_indices[idx],
        )


class RelevanceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def collate_examples(batch: list[BlockExample]) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    embeddings = torch.stack([example.embedding for example in batch])
    labels = torch.tensor([example.label for example in batch], dtype=torch.float32)
    video_ids = [example.video_id for example in batch]
    block_indices = torch.tensor([example.block_index for example in batch], dtype=torch.long)
    return embeddings, labels, video_ids, block_indices


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_relevance_labels(csv_path: Path) -> dict[str, bool]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")

        columns = {name.lower(): name for name in reader.fieldnames}
        if "video_id" not in columns:
            raise ValueError(f"{csv_path} is missing a video_id column")
        if "relevant" not in columns:
            raise ValueError(f"{csv_path} is missing a Relevant/relevant column")

        video_col = columns["video_id"]
        relevant_col = columns["relevant"]
        return {row[video_col].strip(): parse_bool(row[relevant_col]) for row in reader}


def load_block_embeddings(
    embeddings_dir: Path,
    labels_by_video: dict[str, bool],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    pkl_paths = sorted(embeddings_dir.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl embedding files found in {embeddings_dir}")

    arrays: list[np.ndarray] = []
    labels: list[float] = []
    video_ids: list[str] = []
    block_indices: list[int] = []
    missing_labels: list[str] = []

    for pkl_path in pkl_paths:
        video_id = pkl_path.stem
        if video_id not in labels_by_video:
            missing_labels.append(video_id)
            continue

        with pkl_path.open("rb") as f:
            payload = pickle.load(f)

        embeddings = payload["embeddings"] if isinstance(payload, dict) else payload
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"{pkl_path} embeddings must be 2D, got shape {embeddings.shape}")

        arrays.append(embeddings)
        label = float(labels_by_video[video_id])
        labels.extend([label] * len(embeddings))
        video_ids.extend([video_id] * len(embeddings))
        block_indices.extend(range(len(embeddings)))

    if missing_labels:
        raise ValueError(f"Missing labels for embedding files: {', '.join(sorted(missing_labels))}")
    if not arrays:
        raise ValueError(f"No usable embeddings found in {embeddings_dir}")

    embeddings_tensor = torch.from_numpy(np.concatenate(arrays, axis=0)).float()
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return embeddings_tensor, labels_tensor, video_ids, block_indices


def stratified_video_split(
    video_ids: list[str],
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    label_by_video: dict[str, int] = {}
    for video_id, label in zip(video_ids, labels.tolist(), strict=True):
        label_by_video.setdefault(video_id, int(label))

    videos_by_label = {0: [], 1: []}
    for video_id, label in label_by_video.items():
        videos_by_label[label].append(video_id)

    val_videos: set[str] = set()
    for class_videos in videos_by_label.values():
        rng.shuffle(class_videos)
        if len(class_videos) <= 1:
            n_val = 0
        else:
            n_val = max(1, round(len(class_videos) * val_fraction))
            n_val = min(n_val, len(class_videos) - 1)
        val_videos.update(class_videos[:n_val])

    train_indices = [i for i, video_id in enumerate(video_ids) if video_id not in val_videos]
    val_indices = [i for i, video_id in enumerate(video_ids) if video_id in val_videos]
    return train_indices, val_indices


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).long()
    targets = labels.long()

    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())

    total = max(1, len(targets))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.inference_mode()
def evaluate(
    model: RelevanceClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    video_probs: dict[str, list[float]] = {}
    video_labels: dict[str, float] = {}

    for embeddings, labels, video_ids, _ in loader:
        embeddings = embeddings.to(device)
        probs = torch.sigmoid(model(embeddings)).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

        for video_id, prob, label in zip(video_ids, probs.tolist(), labels.tolist(), strict=True):
            video_probs.setdefault(video_id, []).append(prob)
            video_labels[video_id] = label

    block_probs = torch.cat(all_probs)
    block_labels = torch.cat(all_labels)
    block_metrics = compute_metrics(block_probs, block_labels, threshold)

    pooled_probs = torch.tensor([sum(probs) / len(probs) for probs in video_probs.values()])
    pooled_labels = torch.tensor([video_labels[video_id] for video_id in video_probs])
    video_metrics = compute_metrics(pooled_probs, pooled_labels, threshold)
    return block_metrics, video_metrics


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} acc={metrics['accuracy']:.3f} "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"f1={metrics['f1']:.3f} tp={metrics['tp']:.0f} tn={metrics['tn']:.0f} "
        f"fp={metrics['fp']:.0f} fn={metrics['fn']:.0f}"
    )


def train() -> None:
    embeddings_dir = Path("../embeddings/train")
    csv_path = Path("../data/cooking/EPIC_descriptions.csv")
    checkpoint = Path("checkpoint/relevance_classifier.pt")

    epochs = 30
    batch_size = 128
    hidden_dim = 256
    dropout = 0.25
    lr = 1e-3
    weight_decay = 1e-4
    val_fraction = 0.2
    threshold = 0.5
    seed = 7
    class_weights_enabled = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    labels_by_video = load_relevance_labels(csv_path)
    embeddings, labels, video_ids, block_indices = load_block_embeddings(embeddings_dir, labels_by_video)
    train_indices, val_indices = stratified_video_split(video_ids, labels, val_fraction, seed)

    if not train_indices or not val_indices:
        raise ValueError("Train/validation split produced an empty split")

    mean = embeddings[train_indices].mean(dim=0)
    std = embeddings[train_indices].std(dim=0).clamp_min(1e-6)

    train_dataset = BlockEmbeddingDataset(embeddings, labels, video_ids, block_indices, train_indices, mean, std)
    val_dataset = BlockEmbeddingDataset(embeddings, labels, video_ids, block_indices, val_indices, mean, std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_examples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = RelevanceClassifier(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    positive_blocks = float(labels[train_indices].sum().item())
    negative_blocks = float(len(train_indices) - positive_blocks)

    class_weights = torch.ones(2, dtype=torch.float32, device=device)
    if class_weights_enabled:
        total = positive_blocks + negative_blocks
        class_weights[0] = total / max(1.0, 2.0 * negative_blocks)
        class_weights[1] = total / max(1.0, 2.0 * positive_blocks)

    print(f"embeddings_dir={embeddings_dir}")
    print(f"videos={len(set(video_ids))} blocks={len(labels)} input_dim={embeddings.shape[1]}")
    print(
        f"train_blocks={len(train_indices)} val_blocks={len(val_indices)} "
        f"train_pos={int(positive_blocks)} train_neg={int(negative_blocks)} device={device}"
    )

    best_val_f1 = -1.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for batch_embeddings, batch_labels, _, _ in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            weights = torch.where(batch_labels > 0.5, class_weights[1], class_weights[0])

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_embeddings)
            loss = (criterion(logits, batch_labels) * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_batch_size = len(batch_labels)
            total_loss += loss.item() * current_batch_size
            seen += current_batch_size

        block_metrics, video_metrics = evaluate(model, val_loader, device, threshold)
        avg_loss = total_loss / max(1, seen)

        print(
            f"epoch={epoch:03d} loss={avg_loss:.4f} "
            f"{format_metrics('val_block', block_metrics)} "
            f"{format_metrics('val_video', video_metrics)}"
        )

        if video_metrics["f1"] > best_val_f1:
            best_val_f1 = video_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": embeddings.shape[1],
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "mean": mean,
                    "std": std,
                    "threshold": threshold,
                    "labels": {0: "irrelevant", 1: "relevant"},
                    "embeddings_dir": str(embeddings_dir),
                    "csv_path": str(csv_path),
                },
                checkpoint,
            )

    print(f"saved_best_checkpoint={checkpoint}")


if __name__ == "__main__":
    train()
