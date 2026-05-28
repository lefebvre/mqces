#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

#include <cstddef>

namespace mqces {

// Spatial rank of each row of `x` relative to the cloud of all other rows
// (Eq. 1 of Weber & Dayman). For an (N × d) input the result is also (N × d):
// row j is U_j = (1/N) Σ_{i≠j} (x_j − x_i) / ||x_j − x_i||.
//
// Tolerance on the denominator handles coincident points; pairs at distance
// less than `eps` are skipped (treated as zero-contribution rather than NaN).
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
spatial_rank(const Sample& x, double eps = 1e-12);

// SamplingConfig-aware overload. When `sampling.reference_size` is 0 or
// >= N, dispatches to the exact path above; otherwise uses the
// O(N·R)-cost subsample approximation:
//
//   U_j ≈ (1/R) Σ_{i ∈ Refs, i ≠ j} (x_j − x_i) / ||x_j − x_i||
//
// where Refs is an R-sized uniform subset of {0..N-1} selected
// deterministically from `sampling.seed`. The estimator is unbiased; its
// per-coordinate variance scales as O(1/R).
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
spatial_rank(const Sample& x, const SamplingConfig& sampling, double eps = 1e-12);

// Inverse spatial rank: given target rank vectors `u` (one per row), find
// the points `x_tilde` in the cloud `y` that would produce those ranks
// when measured against `y`. Solves Eq. 2 via the configured inner solver
// (see detail/inverse_solvers.hpp).
//
// Throws std::runtime_error if any row fails to converge inside the
// configured iteration cap.
struct InverseRankResult {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_tilde;
    std::size_t                                                            max_iters_used = 0;
    double                                                                 max_residual = 0.0;
};

// Overload accepting an explicit SolverConfig (preferred new surface).
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    const SolverConfig&                                                           solver);

// Full surface: caller controls both inner solver and the reference subset
// used to approximate the per-row Weiszfeld sum. When
// `sampling.reference_size` is 0 or >= y.rows(), the solver sees all of y
// (current behavior); otherwise it sees an R-sized subsample deterministic
// in `sampling.seed`.
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    const SolverConfig&                                                           solver,
    const SamplingConfig&                                                         sampling);

// Legacy overload: existing callers pass tol/max_iters directly. Wraps
// the SolverConfig form with kind=Weiszfeld for source compatibility.
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    double                                                                        tol = 1e-9,
    std::size_t                                                                   max_iters = 200);

}  // namespace mqces
