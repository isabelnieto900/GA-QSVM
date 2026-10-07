"""Datasets and preprocessing from Section 5.1 of the paper.

Pipeline: fixed stratified train/test split -> MinMaxScaler -> PCA to
``n_features`` components (one feature per qubit, angle encoding). The scaler
and the PCA are always fitted on the training split only.
"""

from __future__ import annotations

import gzip
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DATASETS = ("digits", "fashion", "wine", "cancer")
DATASET_LABELS = {
    "digits": "Digits",
    "fashion": "Fashion",
    "wine": "Wine",
    "cancer": "Breast Cancer",
}

# Same files that tensorflow.keras.datasets.fashion_mnist downloads.
FASHION_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _load_fashion(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    folder = data_dir / "fashion_mnist"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, filename in FASHION_FILES.items():
        path = folder / filename
        if not path.exists():
            urllib.request.urlretrieve(FASHION_URL + filename, path)
        with gzip.open(path, "rb") as handle:
            header_bytes = 16 if key.endswith("images") else 8
            arrays[key] = np.frombuffer(handle.read(), dtype=np.uint8, offset=header_bytes)
    x = np.concatenate([arrays["train_images"], arrays["test_images"]]).reshape(-1, 28 * 28)
    y = np.concatenate([arrays["train_labels"], arrays["test_labels"]])
    return x.astype(float), y.astype(int)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the full dataset ``(x, y)``. Fashion is downloaded once into ``data/``."""
    if name == "digits":
        bunch = load_digits()
    elif name == "wine":
        bunch = load_wine()
    elif name == "cancer":
        bunch = load_breast_cancer()
    elif name == "fashion":
        return _load_fashion()
    else:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")
    return bunch.data, bunch.target


@dataclass(frozen=True)
class SplitSpec:
    train_size: int
    test_size: int
    seed: int
    shuffle_first: bool = False


# Fixed splits used by the original GA-QSVM code (at most 100 train / 100 test;
# Wine only has 178 samples, so its test split has 78).
PAPER_SPLITS = {
    "digits": SplitSpec(train_size=100, test_size=100, seed=55, shuffle_first=True),
    "wine": SplitSpec(train_size=100, test_size=78, seed=20),
    "cancer": SplitSpec(train_size=100, test_size=100, seed=52),
}


class Split(NamedTuple):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def scale_and_reduce(x_train, x_test, n_features: int, seed: int | None = None):
    """MinMax-scale and PCA-project, fitting both transforms on ``x_train`` only."""
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    pca = PCA(n_components=n_features, random_state=seed)
    return pca.fit_transform(x_train), pca.transform(x_test)


def paper_split(name: str, n_features: int) -> Split:
    """Fixed train/test split of the paper, reduced to ``n_features`` PCA components."""
    if name not in PAPER_SPLITS:
        raise ValueError(f"No paper split for {name!r}; expected one of {tuple(PAPER_SPLITS)}")
    spec = PAPER_SPLITS[name]
    x, y = load_dataset(name)
    if spec.shuffle_first:
        x, y = shuffle(x, y, random_state=spec.seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=spec.train_size,
        test_size=spec.test_size,
        random_state=spec.seed,
        shuffle=True,
        stratify=y,
    )
    x_train, x_test = scale_and_reduce(x_train, x_test, n_features, seed=spec.seed)
    return Split(x_train, x_test, y_train, y_test)


def explained_variance_curve(name: str) -> np.ndarray:
    """Cumulative PCA explained variance of the full MinMax-scaled dataset (Figure 3)."""
    x, _ = load_dataset(name)
    pca = PCA().fit(MinMaxScaler().fit_transform(x))
    return np.cumsum(pca.explained_variance_ratio_)


def components_for(curve: np.ndarray, threshold: float) -> int:
    """Smallest number of components whose cumulative explained variance reaches ``threshold``."""
    return int(np.searchsorted(curve, threshold) + 1)
