"""mqces: Multivariate Quantile Comparison for Environmental Samples.

This is the Python-facing wrapper around the C++ core. The compiled
extension `_mqces_core` is imported lazily so a clean error is raised if
the package is installed without its C++ build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from . import _mqces_core as _core

__version__ = _core.__version__


@dataclass(frozen=True)
class Score:
    class_name: str
    s_xy: float


@dataclass(frozen=True)
class ClassificationResult:
    best_class: str
    scores: tuple[Score, ...]
    misclassification_prob: float
    none_of_the_above: bool


def _wrap_result(d: dict) -> ClassificationResult:
    return ClassificationResult(
        best_class=d["best_class"],
        scores=tuple(Score(class_name=n, s_xy=s) for (n, s) in d["scores"]),
        misclassification_prob=float(d["misclassification_prob"]),
        none_of_the_above=bool(d["none_of_the_above"]),
    )


def _normalize_classes(
    classes: Iterable[tuple[str, np.ndarray]] | dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    if isinstance(classes, dict):
        items = list(classes.items())
    else:
        items = list(classes)
    return [(name, np.ascontiguousarray(arr, dtype=np.float64)) for name, arr in items]


_VARIANT_DISPATCH = {
    "classic": "classify_classic",
    "v2":      "classify_v2",
    "v3":      "classify_v3",
    "v4":      "classify_v4",
}


def classify(
    test: np.ndarray,
    classes: Iterable[tuple[str, np.ndarray]] | dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    epsilon: float = 0.0,
    mc_samples: int = 10,
    seed: int = 0xC0FFEE,
    nota_threshold: float = 0.05,
    n_threads: int = 0,
    reference_size: int = 0,
    sampling_seed: int = 0xACEBEEF,
    variant: Literal["classic", "v2", "v3", "v4"] = "classic",
) -> ClassificationResult:
    """Classify ``test`` against ``classes`` using the mqces method.

    Parameters
    ----------
    test
        ``(n_specimens, n_features)`` array of measurements.
    classes
        Iterable of ``(name, specimens)`` tuples (or a name->array dict).
        Each ``specimens`` array must have the same ``n_features`` as ``test``.
    weights
        ``(n_features,)`` diagonal of the W matrix in Eq. 3.
    reference_size
        When > 0, switches spatial_rank and inverse_spatial_rank to the
        O(N·R) subsample-based approximation with ``R = reference_size``
        uniform references per cloud. ``0`` (default) uses the exact
        O(N²) kernel.
    sampling_seed
        Deterministic seed used to pick the reference subset.
    variant
        - ``"classic"`` — paper-faithful Eqs. 5-7.
        - ``"v2"``      — paired-difference t-statistic, Weiszfeld solver.
        - ``"v3"``      — paired t + Vardi-Zhang solver.
        - ``"v4"``      — paired t + VZ + Anderson acceleration
                          (production target at N = 10⁶ scale).
    """
    test_arr    = np.ascontiguousarray(test, dtype=np.float64)
    weights_arr = np.ascontiguousarray(weights, dtype=np.float64)
    if test_arr.ndim != 2:
        raise ValueError(f"test must be 2-D, got shape {test_arr.shape}")
    if weights_arr.ndim != 1 or weights_arr.shape[0] != test_arr.shape[1]:
        raise ValueError(
            f"weights must be 1-D with length {test_arr.shape[1]}, got {weights_arr.shape}")
    if variant not in _VARIANT_DISPATCH:
        raise ValueError(
            f"variant must be one of {sorted(_VARIANT_DISPATCH)}, got {variant!r}")

    fn = getattr(_core, _VARIANT_DISPATCH[variant])
    return _wrap_result(fn(
        test=test_arr,
        classes=_normalize_classes(classes),
        weights=weights_arr,
        epsilon=epsilon,
        mc_samples=mc_samples,
        seed=seed,
        nota_threshold=nota_threshold,
        n_threads=n_threads,
        reference_size=reference_size,
        sampling_seed=sampling_seed,
    ))


# Re-export primitives used by tests and downstream consumers.
similarity_score = _core.similarity_score
spatial_rank     = _core.spatial_rank
perturb          = _core.perturb

__all__ = [
    "Score",
    "ClassificationResult",
    "classify",
    "similarity_score",
    "spatial_rank",
    "perturb",
    "__version__",
]
