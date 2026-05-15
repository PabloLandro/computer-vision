import ast
import itertools
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import wandb
except ImportError:
    wandb = None

BLOCK_SIZE = 64
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

ANNOTATIONS_PATH = REPO_ROOT / "data" / "annotations" / "annotations_train.csv"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint"
CSV_RECORDS_DIR = SCRIPT_DIR / "csv_records"
WANDB_PROJECT = "action-classifier"
USE_WANDB = wandb is not None

BASE_CONFIG = {
    "block_size": BLOCK_SIZE,
    "epochs": 30,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.25,
    "hidden_dims": [512, 256, 128],
    "test_size": 0.2,
    "random_state": 42,
    "patience": 7,
}

PARAM_GRIDS = {
    "verb": {
        "lr": [2e-4, 3e-4, 5e-4],
        "weight_decay": [3e-4, 5e-4, 8e-4],
        "dropout": [0.2, 0.25, 0.3],
        "hidden_dims": [[1024, 512], [1536, 768], [1024, 512, 256]],
        "batch_size": [256],
    },
    "noun": {
        "lr": [2e-4, 3e-4, 5e-4],
        "weight_decay": [3e-4, 5e-4, 1e-3],
        "dropout": [0.3, 0.35, 0.4],
        "hidden_dims": [[1024, 512], [1536, 768], [1024, 512, 256]],
        "batch_size": [128],
    },
}

MAX_RUNS_PER_TASK = 24


def grid_configs(base_config, param_grid, max_runs=None, seed=42):
    keys = list(param_grid)
    values = [param_grid[k] for k in keys]
    configs = []

    for combo in itertools.product(*values):
        cfg = deepcopy(base_config)
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)

    rng = np.random.default_rng(seed)
    rng.shuffle(configs)

    if max_runs:
        return configs[:max_runs]
    return configs


def load_data():
    df = pd.read_csv(ANNOTATIONS_PATH)
    df = df[df["relevant"] == True].copy()

    df["start_block"] = df["start_frame"] // BLOCK_SIZE
    df["stop_block"] = df["stop_frame"] // BLOCK_SIZE

    rows = []

    for _, row in df.iterrows():
        primary_noun_class = ast.literal_eval(row["all_noun_classes"])[0]

        for block in range(row["start_block"], row["stop_block"] + 1):
            rows.append(
                {
                    "video_id": row["video_id"],
                    "block": block,
                    "verb_class": row["verb_class"],
                    "noun_class": primary_noun_class,
                }
            )

    expanded = pd.DataFrame(rows)

    verb_labels = expanded.groupby(["video_id", "block"])["verb_class"].agg(
        lambda x: x.mode()[0]
    )

    noun_labels = expanded.groupby(["video_id", "block"])["noun_class"].agg(
        lambda x: x.mode()[0]
    )

    all_embeddings = []
    all_verb_labels = []
    all_noun_labels = []

    for pkl_path in sorted(EMBEDDINGS_DIR.glob("*.pkl")):
        video_id = pkl_path.stem

        with pkl_path.open("rb") as f:
            payload = pickle.load(f)

        embeddings = payload["embeddings"] if isinstance(payload, dict) else payload

        for block_idx in range(len(embeddings)):
            key = (video_id, block_idx)

            if key not in verb_labels.index:
                continue

            all_embeddings.append(embeddings[block_idx])
            all_verb_labels.append(verb_labels[key])
            all_noun_labels.append(noun_labels[key])

    X = torch.tensor(np.stack(all_embeddings), dtype=torch.float32)

    verb_encoder = LabelEncoder().fit(all_verb_labels)
    noun_encoder = LabelEncoder().fit(all_noun_labels)

    y_verb = torch.tensor(
        verb_encoder.transform(all_verb_labels),
        dtype=torch.long,
    )

    y_noun = torch.tensor(
        noun_encoder.transform(all_noun_labels),
        dtype=torch.long,
    )

    return X, y_verb, y_noun, len(verb_encoder.classes_), len(noun_encoder.classes_)


def make_classifier(input_dim, num_classes, hidden_dims=None, dropout=0.25):
    hidden_dims = hidden_dims or [512, 256, 128]

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, num_classes))

    return nn.Sequential(*layers)


def evaluate_model(model, X_eval, y_eval, device, criterion=None):
    model.eval()

    with torch.no_grad():
        logits = model(X_eval.to(device)).cpu()
        preds = logits.argmax(dim=-1)

        if criterion is not None:
            loss = criterion(logits, y_eval).item()
        else:
            loss = None

    acc = (preds == y_eval).float().mean().item()

    return loss, acc, preds


def train_model(model, X_train, y_train, X_val, y_val, device, task_name, config):
    model = model.to(device)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        train_loss = total_loss / len(y_train)
        val_loss, val_acc, _ = evaluate_model(
            model,
            X_val,
            y_val,
            device,
            criterion,
        )

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"{task_name} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"best={best_val_acc:.4f}"
        )

        if USE_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    f"{task_name}/train_loss": train_loss,
                    f"{task_name}/val_loss": val_loss,
                    f"{task_name}/val_acc": val_acc,
                    f"{task_name}/best_val_acc": best_val_acc,
                    f"{task_name}/best_val_loss": best_val_loss,
                }
            )

        if bad_epochs >= config["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc, best_val_loss


def split_for_task(X, y, config, task_name):
    counts = y.bincount()
    valid_mask = counts[y] >= 2

    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]

    print(
        f"Dropped {(~valid_mask).sum().item()} blocks "
        f"with singleton {task_name} classes"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered,
        y_filtered,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y_filtered,
    )

    return X_train, X_val, y_train, y_val, len(X_filtered)


def run_task(task_name, X, y, num_classes, input_dim, device):
    configs = grid_configs(
        BASE_CONFIG,
        PARAM_GRIDS[task_name],
        max_runs=MAX_RUNS_PER_TASK,
        seed=BASE_CONFIG["random_state"],
    )

    X_train, X_val, y_train, y_val, num_samples = split_for_task(
        X,
        y,
        BASE_CONFIG,
        task_name,
    )

    best = {
        "acc": -1.0,
        "loss": float("inf"),
        "config": None,
        "model": None,
    }

    results = []

    for run_idx, config in enumerate(configs, start=1):
        run_name = f"{task_name}-sweep-{run_idx:03d}"

        print(f"\n=== {run_name}/{len(configs)} ===")
        print({k: config[k] for k in PARAM_GRIDS[task_name]})

        if USE_WANDB:
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                group=f"{task_name}-param-search",
                config={
                    **config,
                    "task": task_name,
                    "num_classes": num_classes,
                    "num_samples": num_samples,
                    "input_dim": input_dim,
                    "device": str(device),
                },
                reinit=True,
            )

        model = make_classifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )

        model, best_acc, best_loss = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            device,
            task_name,
            config,
        )

        results.append(
            {
                "task": task_name,
                "best_val_acc": best_acc,
                "best_val_loss": best_loss,
                **config,
            }
        )

        if best_acc > best["acc"] or (
            best_acc == best["acc"] and best_loss < best["loss"]
        ):
            best = {
                "acc": best_acc,
                "loss": best_loss,
                "config": deepcopy(config),
                "model": deepcopy(model).cpu(),
            }

        if USE_WANDB and wandb.run is not None:
            wandb.summary["best_val_acc"] = best_acc
            wandb.summary["best_val_loss"] = best_loss
            wandb.finish()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["best_val_acc", "best_val_loss"],
        ascending=[False, True],
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = CSV_RECORDS_DIR / f"{task_name}_sweep_results.csv"
    model_path = CHECKPOINT_DIR / f"best_{task_name}_classifier.pt"

    results_df.to_csv(results_path, index=False)
    torch.save(best["model"].state_dict(), model_path)

    print(f"\nBest {task_name}: acc={best['acc']:.4f} loss={best['loss']:.4f}")
    print(best["config"])
    print(f"Saved {results_path} and {model_path}")

    return best, results_df


if __name__ == "__main__":
    X, y_verb, y_noun, n_verb_classes, n_noun_classes = load_data()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    input_dim = X.shape[1]

    print(f"Loaded {len(X)} relevant blocks")
    print(f"Verb classes: {n_verb_classes}  Noun classes: {n_noun_classes}")
    print(f"Device: {device}  Input dim: {input_dim}")

    run_task("verb", X, y_verb, n_verb_classes, input_dim, device)
    run_task("noun", X, y_noun, n_noun_classes, input_dim, device)
