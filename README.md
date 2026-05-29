# Extracting Recipes from Cooking Videos using V-JEPA2 Embeddings

Computer Vision course project by Michelangelo Bettini and Pablo Landrove Perez-Gorgoroso.

This repository builds a pipeline that turns egocentric cooking videos into a sequence of cooking actions, then optionally asks a local LLM to convert those actions into readable recipe steps. The core idea is to use a frozen V-JEPA2 video backbone to embed short video blocks, then train lightweight classifiers over those embeddings instead of fine-tuning a full video model.

## Project Goal

The original goal was broad video captioning from V-JEPA2 embeddings. In practice, V-JEPA2 produces block-level representations and still requires a task-specific classifier, so the project scope was narrowed to cooking videos from EPIC-KITCHENS-100.

The final task is:

1. Split a cooking video into fixed-length frame blocks.
2. Embed each block with V-JEPA2.
3. Predict whether each block contains recipe-relevant cooking activity.
4. Predict the action as a verb, noun, and verb-noun pair.
5. Collapse the predicted event sequence into recipe-like instructions with an LLM.

## Dataset and Embeddings

The classifier pipeline uses precomputed embeddings from EPIC-KITCHENS-100 egocentric cooking videos.

From the poster summary:

- 408 embedded videos
- 170,299 temporal blocks
- 1024-dimensional V-JEPA2 embedding per 64-frame block
- 116,895 blocks overlapping an annotation
- 53,404 background or unannotated blocks
- 7,008 dataset action pairs
- 2,721 action-head classes after filtering

The data is noisy for recipe extraction: many visible actions are background or non-recipe actions, such as washing dishes, opening cupboards, or handling utensils. The pipeline therefore includes a relevance classifier in addition to action classification.

## Pipeline

The active classifier pipeline is under `video-classifier/`.

```text
video
  |
  v
center crop / preprocessing
  |
  v
64-frame blocks
  |
  v
frozen V-JEPA2 embeddings
  |
  v
7-block temporal window
  |
  v
Transformer encoder
  |
  +--> relevance head
  +--> verb head
  +--> noun head
  +--> action-pair head
  |
  v
predicted action sequence
  |
  v
optional local LLM recipe generation
```

Each temporal classifier sample contains the target block plus up to three neighboring blocks on either side. This gives the model roughly 15 seconds of local context, which is useful because a single 64-frame block is often too short to identify a cooking action reliably.

## Models

The repository contains two classifier entry points:

- `video-classifier/train_classifiers.py`: block-level MLP baseline over individual embeddings.
- `video-classifier/train_temporal_classifier.py`: temporal-window Transformer classifier.

The temporal model is the main model. It projects each 1024-D embedding to a 768-D model dimension, adds learned positional embeddings, and passes the 7-block window through a 2-layer, 8-head `torch.nn.TransformerEncoder`. Only the center token is classified.

The model has four heads:

- relevance: predicts whether the block is recipe-relevant
- verb: predicts the action verb
- noun: predicts the action noun
- action pair: predicts the verb-noun pair directly

Temporal training keeps all blocks by default so the relevance head can learn background and irrelevant actions. Verb, noun, and action-pair losses are masked to relevant annotated action blocks.

The default temporal setup trains a 3-model seed ensemble with seeds `42`, `123`, and `999`. Ensemble inference averages model logits and applies a relevance gate before accepting predicted actions.

## Results

The poster reports the following ensemble results:

| Metric | Val | Test |
| --- | ---: | ---: |
| Relevance accuracy | 91.6% | 92.3% |
| Relevance F1 | 84.8% | 86.1% |
| Verb accuracy | 94.4% | 95.6% |
| Noun accuracy | 98.3% | 98.7% |
| Exact action | 93.8% | 95.3% |

The MLP baseline was useful as a comparison point, but the temporal Transformer improved action labeling by using neighboring-block context.

## Repository Layout

```text
.
├── data/
│   ├── annotations/
│   ├── classes/
│   └── unseen/
├── embeddings/
├── preprocess_pipeline/
├── video-classifier/
│   ├── Architecture.md
│   ├── classifier_data.py
│   ├── train_classifiers.py
│   ├── train_temporal_classifier.py
│   ├── infer_temporal_window.py
│   └── outputs/
├── poster/
│   └── poster.tex
├── presentation_notes.md
└── requirements.txt
```

Key expected inputs:

- `embeddings/*.pkl`: training embeddings, one file per video
- `data/unseen/*.pkl`: unseen embeddings for inference
- `data/annotations/annotations_train_test.csv`: frame-level action annotations
- `data/classes/EPIC_verb_classes.csv`: verb id to class name map
- `data/classes/EPIC_noun_classes.csv`: noun id to class name map

Each embedding pickle must contain an `embeddings` key and may contain a `block_size` key.

Generated outputs are written under `video-classifier/outputs/`:

- `models/`: checkpoints and label maps
- `reports/`: metrics and dataset summaries
- `predictions/`: prediction CSVs and inference metrics

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The classifier code also imports `scikit-learn` and `tqdm`. If they are not already installed in the environment, install them as well:

```bash
pip install scikit-learn tqdm
```

## Training

Train the MLP baseline:

```bash
python3 video-classifier/train_classifiers.py
```

Train the temporal-window model:

```bash
python3 video-classifier/train_temporal_classifier.py
```

The temporal trainer uses Apple MPS when available, then CUDA, then CPU. If `wandb` is installed, temporal training logs runs to the `action-classifier` project.

## Demo

The project demo is the temporal-window inference script. Place unseen embedding files in `data/unseen/`, then run:

```bash
python3 video-classifier/infer_temporal_window.py
```

The script always produces action predictions. If you also want the local LLM recipe summary, start Ollama before running inference, for example:

```bash
ollama serve
```

Then run the same inference command in another terminal. When Ollama is available at `http://localhost:11434`, the script lists the installed local models and prompts you to choose one for recipe generation.

By default, inference loads the ensemble from:

```text
video-classifier/outputs/models/temporal_window_ensemble
```

It writes prediction and metric CSV files to:

```text
video-classifier/outputs/predictions/temporal_window_inference
```

If Ollama is not running, inference still completes and skips the LLM recipe summary.

## More Detail

See `video-classifier/Architecture.md` for implementation details about data loading, label alignment, temporal windows, output files, inference metrics, and model checkpoint layout.
