import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class VideoClassifier(nn.Module):
    """Classifies video embeddings of shape (batch, 18432, 1408) into label set."""

    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.attention_pool = nn.Linear(1408, 1)
        self.classifier = nn.Sequential(
            nn.Linear(1408, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 18432, 1408)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)  # (batch, 18432, 1)
        pooled = (x * attn_weights).sum(dim=1)                        # (batch, 1408)
        return self.classifier(pooled)                                 # (batch, num_classes)


def train(
    model: VideoClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str | None = None,
) -> list[float]:
    """
    Train the classifier.

    Args:
        features: (N, 18432, 1408) tensor of video embeddings.
        labels:   (N,) tensor of integer class indices.
        device:   'cuda', 'mps', or 'cpu'. Auto-detected if None.

    Returns:
        List of per-epoch average loss values.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = model.to(device)
    features, labels = features.to(device), labels.to(device)

    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

    return epoch_losses


@torch.inference_mode()
def predict(
    model: VideoClassifier,
    features: torch.Tensor,
    labels: list[str] | None = None,
    batch_size: int = 8,
    device: str | None = None,
) -> tuple[torch.Tensor, list[str] | None]:
    """
    Run inference on video embeddings.

    Args:
        features: (N, 18432, 1408) tensor of video embeddings.
        labels:   Optional list of class name strings for human-readable output.
        device:   Auto-detected if None.

    Returns:
        (class_indices, class_names) — class_names is None when labels not provided.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = model.to(device).eval()
    dataset = TensorDataset(features.to(device))
    loader = DataLoader(dataset, batch_size=batch_size)

    all_indices = []
    for (x_batch,) in loader:
        logits = model(x_batch)
        all_indices.append(logits.argmax(dim=-1).cpu())

    indices = torch.cat(all_indices)
    class_names = [labels[i] for i in indices.tolist()] if labels else None
    return indices, class_names


if __name__ == "__main__":
    LABELS = ["running", "jumping", "swimming", "cycling", "sitting"]
    N = 16  # number of video samples

    features = torch.randn(N, 18432, 1408)
    targets = torch.randint(0, len(LABELS), (N,))

    model = VideoClassifier(num_classes=len(LABELS))
    train(model, features, targets, num_epochs=5, batch_size=4)

    indices, names = predict(model, features, labels=LABELS)
    print("Predicted classes:", names)
