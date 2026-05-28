#!/usr/bin/env python3
"""Generate synthetic test fixtures for the mqces integration tests.

Produces three committed C++ headers under tests/data/:

  fixtures_data.hpp / fixtures_data.cpp
      A 19-class burnup-style dataset (50 specimens × 9 features per class)
      modelled loosely on the trends in Figure 1 of Weber & Dayman, plus a
      separate test sample drawn from each class.

  expected_scores.hpp / expected_scores.cpp
      Reference similarity_score matrix S[test_k][class_j] computed by an
      independent pure-NumPy reimplementation of Eqs. 1-3. The C++
      integration test compares its own scores to these values.

Both are deterministic functions of a fixed seed.

Usage:
    .venv/bin/python tests/data/generate_fixtures.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_CLASSES   = 19           # time steps in the paper
N_SPECIMENS = 50           # per class (paper used 990, scaled down for tests)
N_FEATURES  = 9            # 234U, 235U, 236U, 238Pu, 240Pu, 241Pu, 242Pu, 133Cs, 134Cs
SEED        = 0xC0FFEE

# Feature weights from Eq. 4 of the paper (diag W = (10^4, 10^2, ...)).
FEATURE_WEIGHTS = np.array(
    [1e4, 1e2, 1e4, 1e4, 1e1, 1e2, 1e4, 1e0, 1e2], dtype=np.float64)

# Per-class scale of within-class specimen variability (relative).
WITHIN_CLASS_NOISE = 0.05

# Solver settings — must match similarity_score's defaults in C++ for the
# reference scores to be comparable.
INV_RANK_TOL  = 1e-5
INV_RANK_ITRS = 1000


# ---------------------------------------------------------------------------
# Synthetic class centers
# ---------------------------------------------------------------------------

def class_center(k: int) -> np.ndarray:
    """Mean isotopic-ratio vector for class k (k = 0 .. N_CLASSES-1).

    The trends are modeled by simple monotonic curves chosen so neighboring
    classes (k, k+1) are similar but distinguishable, and remote classes
    (k, k+10) are clearly different.
    """
    t = k / (N_CLASSES - 1)                                   # 0 .. 1
    return np.array([
        2e-4 + 1.5e-4 * t,                                    # 234U/238U
        7e-3 + 2e-3 * t,                                      # 235U/238U
        4e-5 + 1.4e-4 * t,                                    # 236U/238U  (Fig. 1 trend)
        1e-4 + 8e-5 * t,                                      # 238Pu/239Pu
        4e-1 + 1e-1 * t,                                      # 240Pu/239Pu
        2e-2 + 1.5e-2 * t,                                    # 241Pu/239Pu
        3e-4 + 4e-4 * t,                                      # 242Pu/239Pu
        1.0 - 0.3 * t,                                        # 133Cs/137Cs
        6e-1 - 4e-1 * t,                                      # 134Cs/137Cs
    ], dtype=np.float64)


def make_class(k: int, rng: np.random.Generator) -> np.ndarray:
    """N_SPECIMENS × N_FEATURES sample for class k."""
    center = class_center(k)
    # Multiplicative within-class noise so ratios stay positive.
    noise = rng.normal(loc=0.0, scale=WITHIN_CLASS_NOISE,
                       size=(N_SPECIMENS, N_FEATURES))
    return center * (1.0 + noise)


# ---------------------------------------------------------------------------
# Pure-NumPy reimplementation of Eqs. 1-3 (reference)
# ---------------------------------------------------------------------------

def spatial_rank(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Eq. 1: U_j = (1/N) Σ_{i≠j} (x_j − x_i) / ||x_j − x_i||."""
    N, _ = x.shape
    u = np.zeros_like(x)
    if N <= 1:
        return u
    for j in range(N):
        diffs = x[j] - x                       # (N, d)
        norms = np.linalg.norm(diffs, axis=1)  # (N,)
        mask  = (np.arange(N) != j) & (norms > eps)
        u[j]  = diffs[mask].T @ (1.0 / norms[mask]) / N
    return u


def inverse_spatial_rank_row(u_j: np.ndarray, y: np.ndarray,
                             tol: float, max_iters: int) -> np.ndarray:
    """Weiszfeld iteration matching detail/nonlinear_solve.hpp."""
    M = y.shape[0]
    x = y.mean(axis=0) + u_j
    for _ in range(max_iters):
        diffs = x - y
        norms = np.linalg.norm(diffs, axis=1)
        ok    = norms > 1e-12
        if not ok.any():
            break
        inv_d = 1.0 / norms[ok]
        numerator = M * u_j + (y[ok].T @ inv_d)
        x_next    = numerator / inv_d.sum()
        step      = np.linalg.norm(x_next - x)
        x = x_next
        if step < tol:
            break
    return x


def inverse_spatial_rank(u: np.ndarray, y: np.ndarray,
                         tol: float = INV_RANK_TOL,
                         max_iters: int = INV_RANK_ITRS) -> np.ndarray:
    return np.array([inverse_spatial_rank_row(u_j, y, tol, max_iters) for u_j in u])


def similarity_score(x: np.ndarray, y: np.ndarray,
                     weights: np.ndarray) -> float:
    """Eq. 3."""
    x_tilde = inverse_spatial_rank(spatial_rank(x), y)
    y_tilde = inverse_spatial_rank(spatial_rank(y), x)
    dx = x - x_tilde
    dy = y - y_tilde
    return float((dx ** 2 @ weights).sum() + (dy ** 2 @ weights).sum())


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "tests" / "data"


def _fmt_double_array(name: str, arr: np.ndarray, shape_comment: str) -> str:
    flat = arr.reshape(-1)
    body = ", ".join(f"{x:.17g}" for x in flat)
    wrapped = textwrap.fill(body, width=92, subsequent_indent="    ")
    return f"const double {name}[{flat.size}] = {{\n    {wrapped}\n}};  // {shape_comment}\n"


def write_fixtures_header(class_data: np.ndarray, test_data: np.ndarray) -> None:
    header = f"""// AUTO-GENERATED by tests/data/generate_fixtures.py. Do not edit by hand.
//
// Synthetic 19-class burnup-style fixture for mqces integration tests.

#pragma once

namespace mqces::test_fixtures {{

inline constexpr int N_CLASSES   = {N_CLASSES};
inline constexpr int N_SPECIMENS = {N_SPECIMENS};
inline constexpr int N_FEATURES  = {N_FEATURES};

extern const char*  class_names[N_CLASSES];
extern const double feature_weights[N_FEATURES];
extern const double class_data[N_CLASSES * N_SPECIMENS * N_FEATURES];
extern const double test_data [N_CLASSES * N_SPECIMENS * N_FEATURES];

}}  // namespace mqces::test_fixtures
"""
    (DATA_DIR / "fixtures_data.hpp").write_text(header)

    names = ", ".join(f'"t={k}"' for k in range(N_CLASSES))
    weights_str = ", ".join(f"{w:.17g}" for w in FEATURE_WEIGHTS)
    body = f"""// AUTO-GENERATED by tests/data/generate_fixtures.py. Do not edit by hand.

#include "fixtures_data.hpp"

namespace mqces::test_fixtures {{

const char* class_names[N_CLASSES] = {{ {names} }};

const double feature_weights[N_FEATURES] = {{ {weights_str} }};

{_fmt_double_array("class_data", class_data, "(N_CLASSES, N_SPECIMENS, N_FEATURES)")}
{_fmt_double_array("test_data",  test_data,  "(N_CLASSES, N_SPECIMENS, N_FEATURES)")}

}}  // namespace mqces::test_fixtures
"""
    (DATA_DIR / "fixtures_data.cpp").write_text(body)


def write_expected_scores(matrix: np.ndarray) -> None:
    header = """// AUTO-GENERATED by tests/data/generate_fixtures.py. Do not edit by hand.
//
// Reference similarity_score(test_k, class_j) computed by a pure-NumPy
// reimplementation of Eqs. 1-3 with the same solver tolerances as
// similarity_score's C++ defaults. The C++ integration test compares its
// own results to these values to cross-validate the implementation.

#pragma once

#include "fixtures_data.hpp"

namespace mqces::test_fixtures {

// Row-major (test_k × class_j).
extern const double expected_scores[N_CLASSES * N_CLASSES];

}  // namespace mqces::test_fixtures
"""
    (DATA_DIR / "expected_scores.hpp").write_text(header)

    body = f"""// AUTO-GENERATED by tests/data/generate_fixtures.py. Do not edit by hand.

#include "expected_scores.hpp"

namespace mqces::test_fixtures {{

{_fmt_double_array("expected_scores", matrix, "row-major (test_k, class_j)")}

}}  // namespace mqces::test_fixtures
"""
    (DATA_DIR / "expected_scores.cpp").write_text(body)


def main() -> None:
    rng = np.random.default_rng(SEED)
    class_data = np.stack([make_class(k, rng) for k in range(N_CLASSES)])
    # Test samples come from an independent RNG stream so the per-class test
    # sample is distinct from (but drawn from the same distribution as) the
    # known-class specimens.
    rng_test = np.random.default_rng(SEED ^ 0xDEADBEEF)
    test_data = np.stack([make_class(k, rng_test) for k in range(N_CLASSES)])

    write_fixtures_header(class_data, test_data)
    print(f"Wrote {DATA_DIR/'fixtures_data.hpp'} + .cpp")

    # Reference scores: full (19 × 19) S[test_k][class_j] matrix. Expensive
    # in pure Python — ~100s — but only runs at fixture-regen time.
    print(f"Computing reference {N_CLASSES} × {N_CLASSES} score matrix...")
    matrix = np.zeros((N_CLASSES, N_CLASSES))
    for k in range(N_CLASSES):
        for j in range(N_CLASSES):
            matrix[k, j] = similarity_score(test_data[k], class_data[j], FEATURE_WEIGHTS)
        print(f"  test class {k:2d} done")
    write_expected_scores(matrix)
    print(f"Wrote {DATA_DIR/'expected_scores.hpp'} + .cpp")


if __name__ == "__main__":
    main()
