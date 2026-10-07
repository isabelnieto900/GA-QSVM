import numpy as np
import pytest

from ga_qsvm.data import (
    PAPER_SPLITS,
    components_for,
    explained_variance_curve,
    load_dataset,
    paper_split,
    scale_and_reduce,
)


@pytest.mark.parametrize(
    ("name", "n_train", "n_test"),
    [("digits", 100, 100), ("wine", 100, 78), ("cancer", 100, 100)],
)
@pytest.mark.parametrize("n_features", [3, 7])
def test_paper_split_sizes_and_feature_count(name, n_train, n_test, n_features):
    split = paper_split(name, n_features)

    assert split.x_train.shape == (n_train, n_features)
    assert split.x_test.shape == (n_test, n_features)
    assert len(split.y_train) == n_train
    assert len(split.y_test) == n_test


@pytest.mark.parametrize("name", list(PAPER_SPLITS))
def test_paper_split_is_deterministic_and_stratified(name):
    first = paper_split(name, 5)
    second = paper_split(name, 5)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)

    _, y = load_dataset(name)
    classes = np.unique(y)
    assert set(np.unique(first.y_train)) == set(classes)
    full_ratio = np.bincount(y) / len(y)
    train_ratio = np.bincount(first.y_train, minlength=len(classes)) / len(first.y_train)
    assert np.abs(full_ratio - train_ratio).max() < 0.02


def test_scale_and_reduce_fits_on_train_only():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 6))
    x_test = rng.normal(size=(20, 6))
    x_test_outlier = x_test.copy()
    x_test_outlier[0] = 1e6

    train_a, _ = scale_and_reduce(x_train, x_test, n_features=3)
    train_b, test_b = scale_and_reduce(x_train, x_test_outlier, n_features=3)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_allclose(train_a.mean(axis=0), 0, atol=1e-12)
    assert test_b.shape == (20, 3)


@pytest.mark.parametrize(("name", "expected"), [("digits", 30), ("wine", 10), ("cancer", 10)])
def test_figure3_components_for_95_percent_match_paper(name, expected):
    assert components_for(explained_variance_curve(name), 0.95) == expected


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("mnist")
    with pytest.raises(ValueError):
        paper_split("fashion", 5)
