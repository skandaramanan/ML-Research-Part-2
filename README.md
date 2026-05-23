# COMP30027 Project 2 — Coarse-to-Fine Image Classification

Two-task image classification (Task 1: 10-class coarse animals; Task 2: 10-class fine-grained birds).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name mlassignment2 --display-name "Python (MLAssignment2 .venv)"
```

In Cursor/VS Code or Jupyter, pick the kernel **Python (MLAssignment2 .venv)** for all notebooks (system Python does not include `torch`).

## Run order

1. `notebooks/00_exploration.ipynb` — sanity checks on data and metadata.
2. `notebooks/01_cnn_embeddings.ipynb` — extract and cache ResNet18 / EfficientNet-B0 embeddings to `outputs/embeddings/` (run once).
3. `notebooks/02_task1_models.ipynb` — Task 1 models, holdout + 5-fold CV, Kaggle CSV.
4. `notebooks/03_task2_models.ipynb` — Task 2 models including the required weighted soft-voting ensemble, 5-fold CV, Kaggle CSV.
5. `notebooks/04_error_analysis.ipynb` — confusion matrices, per-class accuracy, misclassified grids, class-centroid similarity, learning curves.

Before model training, generate and inspect the majority-baseline CSVs to confirm Kaggle format:

- `outputs/predictions/task1_majority_baseline_class_id.csv`
- `outputs/predictions/task2_majority_baseline_class_id.csv`

All shared code lives in `src/utils.py`. Outputs land in `outputs/` (metrics CSV, confusion matrices, Kaggle predictions, figures).

## Layout

```
task1_data/, task2_data/       provided datasets (unchanged)
src/utils.py                   shared loaders, splitting, eval, submissions, logging
notebooks/                     analysis notebooks (run in numeric order)
outputs/                       generated artefacts (gitignored embeddings)
report/                        2,500-3,000 word report PDF + source
```
