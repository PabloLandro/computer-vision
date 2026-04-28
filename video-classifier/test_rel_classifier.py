from pathlib import Path

import torch
from torch.utils.data import DataLoader

from relevance_classifier import (
    BlockEmbeddingDataset,
    RelevanceClassifier,
    collate_examples,
    evaluate,
    format_metrics,
    load_block_embeddings,
    load_relevance_labels,
)


def test() -> None:
    embeddings_dir = Path("../embeddings/test")
    csv_path = Path("../data/cooking/EPIC_descriptions.csv")
    checkpoint_path = Path("video-classifier/relevance_classifier.pt")

    batch_size = 128

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = RelevanceClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    threshold = checkpoint["threshold"]

    labels_by_video = load_relevance_labels(csv_path)

    embeddings, labels, video_ids, block_indices = load_block_embeddings(
        embeddings_dir,
        labels_by_video,
    )

    test_indices = list(range(len(labels)))

    test_dataset = BlockEmbeddingDataset(
        embeddings,
        labels,
        video_ids,
        block_indices,
        test_indices,
        mean,
        std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )

    block_metrics, video_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold,
    )

    print(f"embeddings_dir={embeddings_dir}")
    print(f"videos={len(set(video_ids))} blocks={len(labels)} input_dim={embeddings.shape[1]}")
    print(format_metrics("test_block", block_metrics))
    print(format_metrics("test_video", video_metrics))


if __name__ == "__main__":
    test()