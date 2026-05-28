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

// Inverse spatial rank: given target rank vectors `u` (one per row), find
// the points `x_tilde` in the cloud `y` that would produce those ranks
// when measured against `y`. Solves Eq. 2 via a damped fixed-point
// iteration (Anderson acceleration; see detail/nonlinear_solve.hpp).
//
// Throws std::runtime_error if any row fails to converge inside `max_iters`.
struct InverseRankResult {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_tilde;
    std::size_t                                                            max_iters_used = 0;
    double                                                                 max_residual = 0.0;
};

InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    double                                                                        tol = 1e-9,
    std::size_t                                                                   max_iters = 200);

}  // namespace mqces
