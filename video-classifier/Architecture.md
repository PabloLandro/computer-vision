# Architecture

This project trains and runs classifiers over precomputed video block embeddings. The active classifier code lives in `video-classifier/`:

- `classifier_data.py` contains shared data loading, label alignment, splitting, metrics, prediction writing, and device selection utilities.
- `train_classifiers.py` trains a block-level MLP baseline.
- `train_temporal_classifier.py` trains a temporal-window transformer classifier, optionally as a seed ensemble, with an action classifier and a relevance classifier.
- `infer_temporal_window.py` loads trained temporal checkpoints, writes predictions for unseen embedding files, and can optionally ask a local Ollama model to turn predicted actions into a recipe.

## Data Layout

The Python code expects these input directories and files:

- `embeddings/*.pkl`: training embedding files, one file per video. Each pickle must contain an `embeddings` key and may contain a `block_size` key.
- `data/unseen/*.pkl`: inference embedding files for the temporal inference script.
- `data/annotations/annotations_train_test.csv`: action annotations with frame spans, verb/noun labels, and a `relevant` flag.
- `data/classes/EPIC_verb_classes.csv`: verb id to class name map.
- `data/classes/EPIC_noun_classes.csv`: noun id to class name map.

Generated artifacts are written below `video-classifier/outputs/`:

- `video-classifier/outputs/models/`: model checkpoints and label maps.
- `video-classifier/outputs/reports/`: dataset summaries and metrics JSON files.
- `video-classifier/outputs/predictions/`: block-level prediction CSV files and inference metrics.

## Shared Data Pipeline

`classifier_data.py` is the foundation for both training scripts and part of inference.

The main data flow is:

1. `read_annotations()` reads `annotations_train_test.csv` into `Annotation` records grouped by video id.
2. `build_dataset()` reads every pickle in repo-root `embeddings/`, normalizes embedding arrays to 2D `float32`, and converts each embedding row into a fixed-size video block.
3. For each block, `best_overlapping_annotation()` chooses the annotation with the largest frame overlap, requiring at least `MIN_ACTION_OVERLAP_FRAMES`.
4. Blocks without enough annotation overlap are labeled as background using `BACKGROUND_ID = -1`.
5. The dataset returns:
   - `x`: a NumPy matrix of embeddings shaped `[num_blocks, embedding_dim]`.
   - `y["verb"]`, `y["noun"]`, `y["relevant"]`, `y["action"]`: label arrays.
   - `y["meta"]`: `BlockMeta` records used when writing predictions.
   - `y["skipped_files"]`: embedding files skipped because they were missing the `embeddings` key.

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
3. Iteratively drop verb or noun classes with fewer than `MIN_CLASS_COUNT` examples.
4. Split by `y["action"]`.
5. Train `MultiHeadMLP` through `train_mlp_classifier()`.

`MultiHeadMLP` has separate verb and noun backbones. It emits three heads:

- verb logits
- noun logits
- action-pair logits

Training optimizes verb loss, noun loss, and weighted action loss. Prediction currently uses the action head when `USE_ACTION_HEAD_FOR_PREDICTION = True`, then maps the predicted action pair back to verb and noun ids. Pair-prior and action-ensemble prediction paths still exist behind flags, but they are disabled by default.

Artifacts:

- Model checkpoint: `video-classifier/outputs/models/mlp/classifier.pt`
- Label maps: `video-classifier/outputs/models/mlp/label_maps.json`
- Metrics: `video-classifier/outputs/reports/mlp/metrics_<timestamp>.json`
- Dataset summary: `video-classifier/outputs/reports/block/dataset_summary_<timestamp>.json`
- Combined metrics: `video-classifier/outputs/reports/block/all_metrics_<timestamp>.json`
- Test predictions: `video-classifier/outputs/predictions/mlp/test_block_predictions_<timestamp>.csv`

## Temporal Classifier Training

`train_temporal_classifier.py` is the main temporal model entry point:

```bash
python train_temporal_classifier.py
```

It loads the same block dataset as the MLP baseline, but the default training mode is different:

- `TRAIN_RELEVANCE_ALL_BLOCKS = True` keeps all blocks so the relevance head can learn foreground/background and irrelevant-action decisions.
- Action labels are only encoded for relevant, non-background blocks.
- Verb, noun, and action losses are masked to valid action blocks.
- Relevance loss is trained on every sampled block.
- If `TRAIN_RELEVANCE_ALL_BLOCKS` is disabled, the script falls back to the relevant non-singleton filtering used by the MLP baseline.

The script builds local temporal context with `build_temporal_windows()`.

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
- separate verb, noun, action, and relevance heads

The default temporal prediction path averages or scores action candidates using the action head together with verb and noun log-probabilities, plus the pair prior when `USE_PAIR_PRIOR = True`. The relevance head produces `p_relevant`; accepted predictions are gated by `RELEVANCE_THRESHOLD = 0.5`.

The default configuration uses a seed ensemble:

- `USE_SEED_ENSEMBLE = True`
- `ENSEMBLE_SEEDS = [42, 123, 999]`

Each seed trains a separate checkpoint under `video-classifier/outputs/models/temporal_window_ensemble/seed_<seed>/classifier.pt`. Ensemble prediction averages verb, noun, action, and relevance logits from all seed models, scores valid action pairs, and applies the predicted relevance gate.

Weights & Biases logging is optional. If `wandb` is importable, `USE_WANDB` is true and each temporal seed run is logged to the `action-classifier` project.

Artifacts:

- Ensemble checkpoints: `video-classifier/outputs/models/temporal_window_ensemble/seed_<seed>/classifier.pt`
- Ensemble label maps: `video-classifier/outputs/models/temporal_window_ensemble/label_maps.json`
- Metrics: `video-classifier/outputs/reports/temporal_window_ensemble/metrics_<timestamp>.json`
- Combined metrics: `video-classifier/outputs/reports/temporal_window_ensemble/all_metrics_<timestamp>.json`
- Dataset summary: `video-classifier/outputs/reports/temporal_window_ensemble/dataset_summary_<timestamp>.json`
- Test predictions: `video-classifier/outputs/predictions/temporal_window_ensemble/test_block_predictions_<timestamp>.csv`

If `USE_SEED_ENSEMBLE` is disabled, the script writes a single model under `video-classifier/outputs/models/temporal_window/` and corresponding reports/predictions under `temporal_window`.

## Temporal Inference

`infer_temporal_window.py` runs trained temporal checkpoints over files in `data/unseen`:

```bash
python infer_temporal_window.py
```

The default model root is:

```text
video-classifier/outputs/models/temporal_window_ensemble
```

The inference pipeline is:

1. Load label maps from `label_maps.json` or fall back to a checkpoint.
2. Validate expected checkpoint paths for the configured seeds.
3. For each `.pkl` file in `data/unseen`, load embeddings with `load_video_blocks()`.
4. Build temporal windows using the same `build_temporal_windows()` function as training.
5. Load each `TemporalWindowClassifier` checkpoint and require relevance-head weights.
6. Average action and relevance logits across models, decode the raw action prediction, and gate accepted actions with the predicted relevance probability.
7. Write prediction CSV and metrics CSV files under `video-classifier/outputs/predictions/temporal_window_inference/`.

Inference can evaluate predictions when annotations exist for the unseen video id in `annotations_train_test.csv`. Reports include oracle action metrics using ground-truth relevance for debugging plus deployed pipeline metrics using predicted relevance.

When `GENERATE_RECIPE = True`, inference also tries to connect to Ollama at `http://localhost:11434`, prompts the user to choose a local model, collapses consecutive predicted relevant actions, and asks Ollama to generate a concise recipe. If Ollama is unavailable or no relevant actions are detected, recipe generation is skipped.

## Important Conventions

- Background labels use `BACKGROUND_ID = -1` and render as `<background>`.
- Blocks are derived from embedding row index and `block_size`, defaulting to `DEFAULT_BLOCK_SIZE = 64`.
- Annotation alignment is based on maximum frame overlap, with a minimum overlap of half the default block size.
- `KEEP_ONLY_RELEVANT_ACTIONS = False`, so annotation loading keeps both relevant and irrelevant annotations by default.
- MLP training filters to relevant action blocks and excludes singleton verb/noun classes.
- Temporal training keeps all blocks by default for relevance training, while masking action losses to valid relevant action blocks.
- Prediction CSVs include video id, block index, frame range, true action, raw predicted action, accepted predicted action, true relevance, predicted relevance, and `p_relevant`.
- Device selection prefers Apple MPS, then CUDA, then CPU via `get_torch_device()`.

## Dependency Surface

Runtime dependencies are listed in repo-root `requirements.txt`:

- `torch`
- `numpy`
- `torchcodec`
- `transformers`
- `pillow`
- `torchvision`
- `wandb`

The classifier files also import `scikit-learn` and `tqdm`, but those packages are not currently listed in `requirements.txt`. `wandb` is optional in `train_temporal_classifier.py`; if it is missing, temporal training still runs without W&B logging.

## High-Level Flow

```text
data/annotations + data/classes + embeddings
        |
        v
video-classifier/classifier_data.py
        |
        +--> video-classifier/train_classifiers.py ---------> video-classifier/outputs/models/mlp
        |                                                    video-classifier/outputs/reports/mlp
        |                                                    video-classifier/outputs/predictions/mlp
        |
        +--> video-classifier/train_temporal_classifier.py -> video-classifier/outputs/models/temporal_window_ensemble
                                                             video-classifier/outputs/reports/temporal_window_ensemble
                                                             video-classifier/outputs/predictions/temporal_window_ensemble

data/unseen + video-classifier/outputs/models/temporal_window_ensemble
        |
        v
video-classifier/infer_temporal_window.py -----------------> video-classifier/outputs/predictions/temporal_window_inference
        |
        +--> optional local Ollama recipe generation
```
