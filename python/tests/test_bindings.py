"""Pytest suite for the mqces Python bindings.

Run with the in-tree build (no install needed):
    PYTHONPATH=build/python ../.venv/bin/python -m pytest python/tests
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mqces


def _gaussian_class(n: int, d: int, center: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center, scale=sigma, size=(n, d))


def test_version_is_a_string():
    assert isinstance(mqces.__version__, str)
    assert mqces.__version__.count(".") == 2


def test_perturb_zero_epsilon_is_noop():
    x = np.full((10, 3), 2.5)
    y = mqces.perturb(x, 0.0, 42)
    np.testing.assert_array_equal(y, x)


def test_perturb_deterministic():
    x = np.full((10, 3), 1.0)
    a = mqces.perturb(x, 0.05, 123)
    b = mqces.perturb(x, 0.05, 123)
    c = mqces.perturb(x, 0.05, 124)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_spatial_rank_two_points():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    u = mqces.spatial_rank(x)
    # diff norm = 5; row 0 = (-3, -4)/5 / 2 = (-0.3, -0.4)
    assert u.shape == (2, 2)
    np.testing.assert_allclose(u[0], [-0.3, -0.4], atol=1e-12)
    np.testing.assert_allclose(u[1], [ 0.3,  0.4], atol=1e-12)


def test_similarity_score_identical_samples_near_zero():
    rng = np.random.default_rng(7)
    x   = rng.normal(size=(15, 3))
    w   = np.ones(3)
    s   = mqces.similarity_score(x, x, w)
    assert s >= 0.0
    assert s < 1e-6


def test_similarity_score_symmetric():
    rng = np.random.default_rng(11)
    x   = rng.normal(size=(20, 3))
    y   = rng.normal(size=(20, 3))
    w   = np.ones(3)
    s1  = mqces.similarity_score(x, y, w)
    s2  = mqces.similarity_score(y, x, w)
    assert math.isclose(s1, s2, abs_tol=1e-7)


def test_classify_picks_correct_class_on_separated_gaussians():
    d       = 3
    classes = [
        ("A", _gaussian_class(20, d, -5.0, 0.5, 11)),
        ("B", _gaussian_class(20, d,  0.0, 0.5, 22)),
        ("C", _gaussian_class(20, d,  5.0, 0.5, 33)),
    ]
    test = _gaussian_class(20, d, 0.0, 0.5, 44)
    res  = mqces.classify(
        test=test, classes=classes, weights=np.ones(d),
        epsilon=0.0, mc_samples=5, seed=1, n_threads=1,
    )
    assert isinstance(res, mqces.ClassificationResult)
    assert res.best_class == "B"
    assert res.scores[0].class_name == "B"
    assert 0.0 <= res.misclassification_prob <= 1.0
    # Returned scores are sorted ascending.
    for prev, cur in zip(res.scores, res.scores[1:]):
        assert prev.s_xy <= cur.s_xy


def test_classify_accepts_dict_of_classes():
    d       = 2
    classes = {
        "A": _gaussian_class(15, d, -3.0, 0.5, 1),
        "B": _gaussian_class(15, d,  0.0, 0.5, 2),
        "C": _gaussian_class(15, d,  3.0, 0.5, 3),
    }
    test = _gaussian_class(15, d, -3.0, 0.5, 4)
    res  = mqces.classify(
        test=test, classes=classes, weights=np.ones(d), mc_samples=4, n_threads=1)
    assert res.best_class == "A"


def test_classify_v2_agrees_with_classic_on_easy_case():
    d       = 3
    classes = [
        ("near", _gaussian_class(20, d,  0.0, 0.5, 1)),
        ("far",  _gaussian_class(20, d, 10.0, 0.5, 2)),
    ]
    test = _gaussian_class(20, d, 0.0, 0.5, 3)
    r_c  = mqces.classify(test, classes, np.ones(d), variant="classic",
                          mc_samples=5, seed=42, n_threads=1)
    r_v2 = mqces.classify(test, classes, np.ones(d), variant="v2",
                          mc_samples=5, seed=42, n_threads=1)
    assert r_c.best_class == "near"
    assert r_v2.best_class == "near"


def test_classify_validates_weights_shape():
    d       = 3
    classes = [("k", _gaussian_class(10, d, 0.0, 1.0, 1))]
    test    = _gaussian_class(10, d, 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        mqces.classify(test, classes, np.ones(4))  # wrong length


def test_classify_validates_test_shape():
    classes = [("k", np.ones((10, 3)))]
    with pytest.raises(ValueError):
        mqces.classify(np.ones(10), classes, np.ones(3))  # test is 1-D


@pytest.mark.parametrize("variant", ["V2", "classic ", "", "paired"])
def test_classify_rejects_unknown_variant(variant):
    d       = 2
    classes = [("k", _gaussian_class(10, d, 0.0, 1.0, 1))]
    test    = _gaussian_class(10, d, 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="variant"):
        mqces.classify(test, classes, np.ones(d), variant=variant)


@pytest.mark.parametrize(
    "options",
    [{"epsilon": -0.1}, {"mc_samples": 1}, {"nota_threshold": 0.0},
     {"nota_threshold": 1.5}, {"nota_permutations": 5}],
)
def test_classify_rejects_invalid_options(options):
    d       = 2
    classes = [("k", _gaussian_class(10, d, 0.0, 1.0, 1))]
    test    = _gaussian_class(10, d, 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        mqces.classify(test, classes, np.ones(d), **options)


@pytest.mark.parametrize("variant", ["classic", "v2"])
def test_close_classes_report_substantial_uncertainty(variant):
    # Classes 0.3 standard deviations apart are genuinely hard to tell apart;
    # the default epsilon = 0 must still yield an informative probability.
    d       = 3
    classes = [
        ("A", _gaussian_class(20, d, 0.0, 1.0, 5)),
        ("B", _gaussian_class(20, d, 0.3, 1.0, 6)),
    ]
    test = _gaussian_class(20, d, 0.0, 1.0, 7)
    res  = mqces.classify(test, classes, np.ones(d), variant=variant, n_threads=1)
    assert 0.01 < res.misclassification_prob <= 0.5


def test_nota_flags_sample_from_unknown_distribution():
    d       = 3
    classes = [("k", _gaussian_class(30, d, 0.0, 1.0, 1))]
    test    = _gaussian_class(30, d, 3.0, 1.0, 2)
    res     = mqces.classify(test, classes, np.ones(d), n_threads=1)
    assert res.none_of_the_above
