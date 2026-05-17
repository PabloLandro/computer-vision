# Architecture

This project trains and runs classifiers over precomputed video block embeddings. The codebase is organized as a small set of root-level Python scripts:

- `classifier_data.py` contains shared data loading, label alignment, splitting, metrics, prediction writing, and device selection utilities.
- `train_classifiers.py` trains a block-level MLP baseline.
- `train_temporal_classifier.py` trains a temporal-window transformer classifier, optionally as a seed ensemble.
- `infer_temporal_window.py` loads trained temporal checkpoints and writes predictions for unseen embedding files.

## Data Layout

The Python code expects these input directories and files:

- `data/embeddings/*.pkl`: training embedding files, one file per video. Each pickle must contain an `embeddings` key and may contain a `block_size` key.
- `data/unseen/*.pkl`: inference embedding files for the temporal inference script.
- `data/annotations/annotations_train_test.csv`: action annotations with frame spans, verb/noun labels, and a `relevant` flag.
- `data/classes/EPIC_verb_classes.csv`: verb id to class name map.
- `data/classes/EPIC_noun_classes.csv`: noun id to class name map.

Generated artifacts are written below `outputs/`:

- `outputs/models/`: model checkpoints and label maps.
- `outputs/reports/`: dataset summaries and metrics JSON files.
- `outputs/predictions/`: block-level prediction CSV files and inference metrics.

## Shared Data Pipeline

`classifier_data.py` is the foundation for both training scripts and part of inference.

The main data flow is:

1. `read_annotations()` reads `annotations_train_test.csv` into `Annotation` records grouped by video id.
2. `build_dataset()` reads every pickle in `data/embeddings`, normalizes embedding arrays to 2D `float32`, and converts each embedding row into a fixed-size video block.
3. For each block, `best_overlapping_annotation()` chooses the annotation with the largest frame overlap, requiring at least `MIN_ACTION_OVERLAP_FRAMES`.
4. Blocks without enough annotation overlap are labeled as background using `BACKGROUND_ID = -1`.
5. The dataset returns:
   - `x`: a NumPy matrix of embeddings shaped `[num_blocks, embedding_dim]`.
   - `y["verb"]`, `y["noun"]`, `y["relevant"]`, `y["action"]`: label arrays.
   - `y["meta"]`: `BlockMeta` records used when writing predictions.

Splitting is block-level and mostly stratified by action pair through `stratified_block_split()`. The helper `video_split()` also exists but is not used by the current training entry points.

Label handling is centralized through `make_label_encoding()`, `encode_labels()`, and `decode_labels()`, because PyTorch classifiers need contiguous class ids while reports and prediction CSVs should use original class ids and names.

## MLP Baseline Training

`train_classifiers.py` is the entry point for the block-level baseline:

```bash
python train_classifiers.py
```

Its pipeline is:

1. Load class maps, annotations, and embeddings through `classifier_data.py`.
2. Filter to relevant, non-background blocks with `filter_relevant_non_singleton_blocks()`.
3. Drop verb or noun classes with fewer than `MIN_CLASS_COUNT` examples.
4. Split by `y["action"]`.
5. Train `MultiHeadMLP` through `train_mlp_classifier()`.

`MultiHeadMLP` has separate verb and noun backbones. It emits three heads:

- verb logits
- noun logits
- action-pair logits

Training optimizes verb loss, noun loss, and weighted action loss. Prediction currently uses the action head when `USE_ACTION_HEAD_FOR_PREDICTION = True`, then maps the predicted action pair back to verb and noun ids.

Artifacts:

- Model checkpoint: `outputs/models/mlp/classifier.pt`
- Label maps: `outputs/models/mlp/label_maps.json`
- Metrics: `outputs/reports/mlp/metrics_<timestamp>.json`
- Dataset summary: `outputs/reports/block/dataset_summary_<timestamp>.json`
- Test predictions: `outputs/predictions/mlp/test_block_predictions_<timestamp>.csv`

## Temporal Classifier Training

`train_temporal_classifier.py` is the main temporal model entry point:

```bash
python train_temporal_classifier.py
```

It starts from the same filtered block dataset as the MLP baseline, then builds local temporal context with `build_temporal_windows()`.

With the default settings:

- `TEMPORAL_WINDOW_RADIUS = 3`
- `TEMPORAL_WINDOW_SIZE = 7`
- each training sample contains the center block plus up to three neighboring blocks on each side
- missing neighbors at video boundaries are zero-filled and marked invalid in a mask
- predictions are made from the encoded center position only

`TemporalWindowClassifier` has:

- a linear projection from embedding dimension to `TEMPORAL_MODEL_DIM`
- learned positional embeddings over the temporal window
- a `torch.nn.TransformerEncoder`
- separate verb, noun, and action heads

The default configuration uses a seed ensemble:

- `USE_SEED_ENSEMBLE = True`
- `ENSEMBLE_SEEDS = [42, 123, 999]`

Each seed trains a separate checkpoint under `outputs/models/temporal_window_ensemble/seed_<seed>/classifier.pt`. Ensemble prediction averages action logits from all seed models and decodes the best action pair.

Artifacts:

- Ensemble checkpoints: `outputs/models/temporal_window_ensemble/seed_<seed>/classifier.pt`
- Ensemble label maps: `outputs/models/temporal_window_ensemble/label_maps.json`
- Metrics: `outputs/reports/temporal_window_ensemble/metrics_<timestamp>.json`
- Dataset summary: `outputs/reports/temporal_window_ensemble/dataset_summary_<timestamp>.json`
- Test predictions: `outputs/predictions/temporal_window_ensemble/test_block_predictions_<timestamp>.csv`

If `USE_SEED_ENSEMBLE` is disabled, the script writes a single model under `outputs/models/temporal_window/`.

## Temporal Inference

`infer_temporal_window.py` runs trained temporal checkpoints over files in `data/unseen`:

```bash
python infer_temporal_window.py
```

The default model root is:

```text
outputs/models/temporal_window_ensemble
```

The inference pipeline is:

1. Load label maps from `label_maps.json` or fall back to a checkpoint.
2. Validate expected checkpoint paths for the configured seeds.
3. For each `.pkl` file in `data/unseen`, load embeddings with `load_video_blocks()`.
4. Build temporal windows using the same `build_temporal_windows()` function as training.
5. Load each `TemporalWindowClassifier` checkpoint.
6. Average action and relevance logits across models, decode the raw action prediction, and gate accepted actions with the predicted relevance probability.
7. Write prediction and metrics CSV files under `outputs/predictions/temporal_window_inference/`.

Inference can evaluate predictions when annotations exist for the unseen video id in `annotations_train_test.csv`. Reports include oracle action metrics using ground-truth relevance for debugging plus deployed pipeline metrics using predicted relevance.

## Important Conventions

- Background labels use `BACKGROUND_ID = -1` and render as `<background>`.
- Blocks are derived from embedding row index and `block_size`, defaulting to `DEFAULT_BLOCK_SIZE = 64`.
- Annotation alignment is based on maximum frame overlap, with a minimum overlap of half the default block size.
- Training currently filters to relevant action blocks and excludes singleton verb/noun classes.
- Prediction CSVs include video id, block index, frame range, true action, predicted action, true relevance, and predicted relevance.
- Device selection prefers Apple MPS, then CUDA, then CPU via `get_torch_device()`.

## Dependency Surface

Runtime dependencies are listed in `requirements.txt`:

- NumPy for array handling and pickle compatibility.
- PyTorch for model training and inference.
- scikit-learn for metrics.
- tqdm for progress bars.
- joblib is listed but is not used by the current Python files.

## High-Level Flow

```text
data/annotations + data/classes + data/embeddings
        |
        v
classifier_data.py
        |
        +--> train_classifiers.py ---------> outputs/models/mlp
        |                                   outputs/reports/mlp
        |                                   outputs/predictions/mlp
        |
        +--> train_temporal_classifier.py -> outputs/models/temporal_window_ensemble
                                            outputs/reports/temporal_window_ensemble
                                            outputs/predictions/temporal_window_ensemble

data/unseen + outputs/models/temporal_window_ensemble
        |
        v
infer_temporal_window.py -----------------> outputs/predictions/temporal_window_inference
```
