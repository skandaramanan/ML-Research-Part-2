"""Shared utilities for the COMP30027 Project 2 notebooks.

All loaders, splitters, evaluators, plotting, submission, and experiment-logging
helpers used by `notebooks/*.ipynb` live here so that notebooks stay short and
the data-handling invariants (CSV alignment, no leakage) are enforced in one
place.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42

TASK_DIRS = {
    1: REPO_ROOT / "task1_data",
    2: REPO_ROOT / "task2_data",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class TaskBundle:
    """Container for one task's loaded data. Acts like a dict for convenience."""

    X_handcrafted_train: np.ndarray
    X_handcrafted_test: np.ndarray
    y_train: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    train_meta: pd.DataFrame
    test_meta: pd.DataFrame
    class_mapping: pd.DataFrame
    feature_names: list[str]
    task_dir: Path

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self) -> list[str]:
        return [
            "X_handcrafted_train",
            "X_handcrafted_test",
            "y_train",
            "train_ids",
            "test_ids",
            "train_meta",
            "test_meta",
            "class_mapping",
            "feature_names",
            "task_dir",
        ]


def _read_feature_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path} missing image_id column")
    return df


def load_task(task: int, data_root: Path | str | None = None) -> TaskBundle:
    """Load all CSVs for `task` (1 or 2) and return aligned train/test arrays.

    The three handcrafted feature CSVs contain rows for *both* train and test
    images, keyed by ``image_id``. We merge them on ``image_id`` (strict inner
    join with alignment asserts) then split into train/test by joining on the
    respective metadata files. This guarantees feature rows align with labels
    and that test feature rows align with the Kaggle submission order.
    """
    if task not in TASK_DIRS:
        raise ValueError(f"task must be 1 or 2, got {task}")
    base = Path(data_root) if data_root is not None else TASK_DIRS[task]
    if not base.exists():
        raise FileNotFoundError(f"Task {task} data directory not found: {base}")

    color_df = _read_feature_csv(base / "color_histogram.csv")
    hog_df = _read_feature_csv(base / "hog_pca.csv")
    addl_df = _read_feature_csv(base / "additional_features.csv")

    color_df = color_df.sort_values("image_id").reset_index(drop=True)
    hog_df = hog_df.sort_values("image_id").reset_index(drop=True)
    addl_df = addl_df.sort_values("image_id").reset_index(drop=True)

    assert (color_df["image_id"].values == hog_df["image_id"].values).all(), (
        "color_histogram and hog_pca image_id ordering mismatch"
    )
    assert (color_df["image_id"].values == addl_df["image_id"].values).all(), (
        "color_histogram and additional_features image_id ordering mismatch"
    )

    color_cols = [c for c in color_df.columns if c != "image_id"]
    hog_cols = [c for c in hog_df.columns if c != "image_id"]
    addl_cols = [c for c in addl_df.columns if c != "image_id"]
    feature_names = color_cols + hog_cols + addl_cols

    merged = pd.concat(
        [
            color_df.set_index("image_id"),
            hog_df.set_index("image_id")[hog_cols],
            addl_df.set_index("image_id")[addl_cols],
        ],
        axis=1,
    )
    assert merged.shape[1] == len(feature_names), "feature concat width mismatch"

    train_meta = pd.read_csv(base / "train_metadata.csv")
    test_meta = pd.read_csv(base / "test_metadata.csv")

    train_meta = train_meta.sort_values("image_id").reset_index(drop=True)
    test_meta = test_meta.sort_values("image_id").reset_index(drop=True)

    missing_train = set(train_meta["image_id"]) - set(merged.index)
    missing_test = set(test_meta["image_id"]) - set(merged.index)
    if missing_train:
        raise AssertionError(
            f"{len(missing_train)} train image_ids missing from feature CSVs"
        )
    if missing_test:
        raise AssertionError(
            f"{len(missing_test)} test image_ids missing from feature CSVs"
        )

    X_train = merged.loc[train_meta["image_id"].values].to_numpy(dtype=np.float32)
    X_test = merged.loc[test_meta["image_id"].values].to_numpy(dtype=np.float32)
    y_train = train_meta["class_id"].to_numpy(dtype=np.int64)

    assert len(X_train) == len(y_train), "X_train/y_train length mismatch"
    assert X_train.shape[1] == X_test.shape[1] == len(feature_names)

    mapping_path = base / "class_mapping.csv"
    if mapping_path.exists():
        class_mapping = pd.read_csv(mapping_path).sort_values("class_id").reset_index(drop=True)
    else:
        class_mapping = (
            train_meta[["class_id", "class_name"]]
            .drop_duplicates()
            .sort_values("class_id")
            .reset_index(drop=True)
        )

    return TaskBundle(
        X_handcrafted_train=X_train,
        X_handcrafted_test=X_test,
        y_train=y_train,
        train_ids=train_meta["image_id"].to_numpy(),
        test_ids=test_meta["image_id"].to_numpy(),
        train_meta=train_meta,
        test_meta=test_meta,
        class_mapping=class_mapping,
        feature_names=feature_names,
        task_dir=base,
    )


def load_images(meta_df: pd.DataFrame, root: Path | str) -> Iterator:
    """Yield PIL.Image objects for each row of ``meta_df`` (lazy)."""
    from PIL import Image  # local import to keep utils import cheap

    root = Path(root)
    for path in meta_df["image_path"].tolist():
        with Image.open(root / path) as im:
            yield im.convert("RGB").copy()


EMBEDDINGS_DIR = REPO_ROOT / "outputs" / "embeddings"
VALID_BACKBONES = frozenset({"resnet18", "effnetb0"})
VALID_EMBED_SPLITS = frozenset({"train", "test"})


def embedding_cache_paths(
    task: int,
    split: str,
    backbone: str,
    embeddings_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return ``(npy_path, image_ids_json)`` for a cached embedding file."""
    if task not in TASK_DIRS:
        raise ValueError(f"task must be 1 or 2, got {task}")
    if split not in VALID_EMBED_SPLITS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if backbone not in VALID_BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(VALID_BACKBONES)}, got {backbone!r}")
    root = Path(embeddings_dir) if embeddings_dir is not None else EMBEDDINGS_DIR
    stem = f"task{task}_{split}_{backbone}"
    return root / f"{stem}.npy", root / f"{stem}_ids.json"


def save_embedding_cache(
    embeddings: np.ndarray,
    image_ids: Iterable,
    task: int,
    split: str,
    backbone: str,
    embeddings_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Save float32 embedding matrix and matching ``image_id`` order JSON."""
    import json

    image_ids = list(image_ids)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
    if len(image_ids) != emb.shape[0]:
        raise ValueError(
            f"image_ids ({len(image_ids)}) and embeddings rows ({emb.shape[0]}) mismatch"
        )

    npy_path, json_path = embedding_cache_paths(task, split, backbone, embeddings_dir)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, emb)
    with open(json_path, "w") as f:
        json.dump({"image_ids": image_ids, "shape": list(emb.shape)}, f, indent=2)
    return npy_path, json_path


def load_embedding_cache(
    task: int,
    split: str,
    backbone: str,
    embeddings_dir: Path | str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load cached embeddings and their ``image_id`` order."""
    import json

    npy_path, json_path = embedding_cache_paths(task, split, backbone, embeddings_dir)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing cache for task={task} split={split} backbone={backbone}: "
            f"expected {npy_path} and {json_path}"
        )
    emb = np.load(npy_path)
    with open(json_path) as f:
        meta = json.load(f)
    return emb, meta["image_ids"]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray | None = None,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
):
    """80/20 stratified holdout. Returns the standard sklearn tuple, plus id
    splits if ``ids`` is provided."""
    if ids is None:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    X_tr, X_va, y_tr, y_va, id_tr, id_va = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=y
    )
    return X_tr, X_va, y_tr, y_va, id_tr, id_va


def kfold(n_splits: int = 5, seed: int = DEFAULT_SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable | None = None,
) -> dict:
    """Compute accuracy, macro-F1, per-class precision/recall, confusion matrix."""
    labels_arr = None if labels is None else list(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_arr,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_arr)
    per_class = {
        k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "support": v["support"]}
        for k, v in report.items()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels_arr,
    }


def plot_confusion(
    cm: np.ndarray,
    labels: Iterable,
    out_path: Path | str,
    title: str | None = None,
    normalize: bool = True,
) -> Path:
    """Save a normalised confusion-matrix PNG."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_disp = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = ".0f"

    labels = list(labels)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(labels)), max(4, 0.6 * len(labels))))
    im = ax.imshow(cm_disp, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title or ("Normalised confusion matrix" if normalize else "Confusion matrix"),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            ax.text(
                j,
                i,
                format(cm_disp[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_disp[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------


def make_submission(
    test_ids: Iterable,
    y_pred: Iterable,
    class_mapping: pd.DataFrame,
    out_path: Path | str,
    label_column: str = "class_id",
) -> Path:
    """Write a Kaggle submission CSV.

    ``label_column`` is either ``"class_id"`` (default; matches the metadata
    convention) or ``"class_name"`` (looked up via ``class_mapping``). Confirm
    the exact column the Kaggle sample expects before final submission.
    """
    test_ids = list(test_ids)
    y_pred = list(y_pred)
    if len(test_ids) != len(y_pred):
        raise ValueError(
            f"test_ids ({len(test_ids)}) and y_pred ({len(y_pred)}) length mismatch"
        )

    if label_column == "class_id":
        labels = y_pred
    elif label_column == "class_name":
        id_to_name = dict(zip(class_mapping["class_id"], class_mapping["class_name"]))
        labels = [id_to_name[int(p)] for p in y_pred]
    else:
        raise ValueError("label_column must be 'class_id' or 'class_name'")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": test_ids, label_column: labels}).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


LOG_COLUMNS = [
    "timestamp",
    "task",
    "model",
    "feature_set",
    "hyperparams",
    "cv_mean_acc",
    "cv_std_acc",
    "cv_mean_macro_f1",
    "cv_std_macro_f1",
    "val_acc",
    "val_macro_f1",
    "train_time_s",
    "notes",
]


def log_experiment(
    row: dict,
    csv_path: Path | str = REPO_ROOT / "outputs" / "metrics" / "log.csv",
) -> Path:
    """Append a row to the experiment log CSV (creating it if needed).

    Unknown keys are kept (added as new columns on next read); missing standard
    keys become empty strings. ``timestamp`` is auto-filled if absent.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    fieldnames = list(LOG_COLUMNS)
    for k in row:
        if k not in fieldnames:
            fieldnames.append(k)

    file_exists = csv_path.exists()
    if file_exists:
        existing = pd.read_csv(csv_path, nrows=0)
        for col in existing.columns:
            if col not in fieldnames:
                fieldnames.append(col)
        existing_rows = pd.read_csv(csv_path)
        for col in fieldnames:
            if col not in existing_rows.columns:
                existing_rows[col] = ""
        new_row = {k: row.get(k, "") for k in fieldnames}
        existing_rows = pd.concat([existing_rows[fieldnames], pd.DataFrame([new_row])], ignore_index=True)
        existing_rows.to_csv(csv_path, index=False)
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return csv_path


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, numpy, and (if available) torch RNGs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
