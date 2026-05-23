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

1. `notebooks/01_cnn_embeddings.ipynb` — extract and cache ResNet18 / EfficientNet-B0 embeddings to `outputs/embeddings/` (run once, or whenever the cache is missing).
2. `notebooks/02_task1_models.ipynb` — Task 1 models, holdout + 5-fold CV, confusion matrices, and Kaggle CSVs.
3. `notebooks/03_task2_models.ipynb` — Task 2 models including the required weighted soft-voting ensemble, 5-fold CV, confusion matrices, and Kaggle CSVs.
4. `notebooks/04_error_analysis.ipynb` — report-focused figures: per-class accuracy, side-by-side confusions, Task 2 misclassified grids, class-centroid similarity, and learning curves.

Canonical submission files are selected from the best validation/CV model outputs:

- `outputs/predictions/task1_submission_class_id.csv`
- `outputs/predictions/task2_submission_class_id.csv`

Majority-baseline CSVs are included only as Kaggle-format sanity checks:

- `outputs/predictions/task1_majority_baseline_class_id.csv`
- `outputs/predictions/task2_majority_baseline_class_id.csv`

All shared code lives in `src/utils.py`. Outputs land in `outputs/` (metrics CSV, confusion matrices, Kaggle predictions, figures, and local embedding caches).

## Layout

```
task1_data/, task2_data/       provided datasets (unchanged)
src/utils.py                   shared loaders, splitting, eval, submissions, logging
notebooks/                     analysis notebooks (run in numeric order)
outputs/                       generated artefacts (embedding arrays are gitignored)
report/                        report PDF + source
```

## Notes

- `outputs/embeddings/*.npy` and `outputs/embeddings/*.json` are intentionally gitignored. Re-run `notebooks/01_cnn_embeddings.ipynb` to regenerate them on a fresh checkout.
- The final report should record the Kaggle scores for both canonical submission CSVs.
