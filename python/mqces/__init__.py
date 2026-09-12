"""mqces: Multivariate Quantile Comparison for Environmental Samples.

This is the Python-facing wrapper around the C++ core. The compiled
extension `_mqces_core` is imported when the package is imported; if it is
missing (e.g. the package directory is on the path but the C++ build was
never run) an ImportError explaining how to build it is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

try:
    from . import _mqces_core as _core
except ImportError as exc:  # pragma: no cover - exercised only on broken installs
    raise ImportError(
        "mqces: the compiled extension module '_mqces_core' could not be imported. "
        "Install the package with `pip install .`, or configure CMake with "
        "-DMQCES_ENABLE_PYTHON=ON and put build/python on PYTHONPATH."
    ) from exc

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


def classify(
    test: np.ndarray,
    classes: Iterable[tuple[str, np.ndarray]] | dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    epsilon: float = 0.0,
    mc_samples: int = 10,
    seed: int = 0xC0FFEE,
    nota_threshold: float = 0.05,
    nota_permutations: int = 199,
    n_threads: int = 0,
    variant: Literal["classic", "v2"] = "classic",
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
    epsilon
        Measurement-error fraction (Eq. 10). 0 omits the measurement-error
        term; specimen sampling variability is always included.
    mc_samples
        Perturbation replicates (>= 2), used only when ``epsilon > 0``.
    nota_threshold
        p-value threshold in (0, 1) for the none-of-the-above test.
    nota_permutations
        Random splits for the none-of-the-above permutation test; 0 skips it.
    variant
        ``"classic"`` uses the t-statistic form of the paper's Eqs. 5-7;
        ``"v2"`` uses a paired-difference t-statistic that accounts for the
        test sample being shared by both scores. See ``mqces/classify.hpp``.

    Raises
    ------
    ValueError
        On malformed arrays, an unknown ``variant``, or invalid options.
    """
    test_arr    = np.ascontiguousarray(test, dtype=np.float64)
    weights_arr = np.ascontiguousarray(weights, dtype=np.float64)
    if test_arr.ndim != 2:
        raise ValueError(f"test must be 2-D, got shape {test_arr.shape}")
    if weights_arr.ndim != 1 or weights_arr.shape[0] != test_arr.shape[1]:
        raise ValueError(
            f"weights must be 1-D with length {test_arr.shape[1]}, got {weights_arr.shape}")

    classify_fns = {"classic": _core.classify_classic, "v2": _core.classify_v2}
    if variant not in classify_fns:
        raise ValueError(f'variant must be "classic" or "v2", got {variant!r}')
    fn = classify_fns[variant]
    return _wrap_result(fn(
        test=test_arr,
        classes=_normalize_classes(classes),
        weights=weights_arr,
        epsilon=epsilon,
        mc_samples=mc_samples,
        seed=seed,
        nota_threshold=nota_threshold,
        nota_permutations=nota_permutations,
        n_threads=n_threads,
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
