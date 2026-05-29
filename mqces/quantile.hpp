#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

#include <cstddef>

namespace mqces {

/**
 * @brief Spatial rank of each row of `x` relative to the rest of the cloud (Eq. 1).
 *
 * For an (N × d) input the result is also (N × d): row j is
 * @f[
 *   U_j = \frac{1}{N} \sum_{i \ne j} \frac{x_j - x_i}{\|x_j - x_i\|}.
 * @f]
 * Pairs at distance less than `eps` are skipped (treated as
 * zero-contribution rather than NaN).
 *
 * @param x    Sample cloud (N × d).
 * @param eps  Coincident-point tolerance on the denominator.
 * @return     Spatial rank matrix (N × d).
 */
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
spatial_rank(const Sample& x, double eps = 1e-12);

/**
 * @brief Spatial rank with reference-subset sampling.
 *
 * When `sampling.reference_size` is 0 or `>= N`, dispatches to the exact
 * path above. Otherwise uses the O(N·R)-cost subsample approximation
 * @f[
 *   U_j \approx \frac{1}{R} \sum_{i \in \text{Refs},\, i \ne j} \frac{x_j - x_i}{\|x_j - x_i\|},
 * @f]
 * where `Refs` is an R-sized uniform subset of {0..N-1} selected
 * deterministically from `sampling.seed`. The estimator is unbiased; its
 * per-coordinate variance scales as O(1/R).
 *
 * @param x         Sample cloud (N × d).
 * @param sampling  Reference-subset configuration.
 * @param eps       Coincident-point tolerance.
 * @return          Spatial rank matrix (N × d).
 */
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
spatial_rank(const Sample& x, const SamplingConfig& sampling, double eps = 1e-12);

/**
 * @brief Result bundle for `inverse_spatial_rank`.
 */
struct InverseRankResult {
    /** @brief Solved reconstructions x_tilde (rows aligned with `u`). */
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x_tilde;
    std::size_t max_iters_used = 0;   ///< Max iterations consumed by any single row.
    double      max_residual   = 0.0; ///< Max fixed-point residual at exit.
    std::size_t aa_fallbacks   = 0;   ///< Aggregate AA → VZ fallbacks (zero unless `VardiZhangAA`).
    std::size_t aa_restarts    = 0;   ///< Aggregate AA history restarts.
};

/**
 * @brief Inverse spatial rank: solve Eq. 2 for the reconstruction `x_tilde`.
 *
 * Given target rank vectors `u` (one per row), find the points
 * `x_tilde` in the cloud `y` whose spatial rank against `y` equals
 * `u`. The inner iteration is selected by `solver.kind`.
 *
 * @param u       Target rank vectors (one per row).
 * @param y       Reference cloud.
 * @param solver  Inner-solver configuration.
 * @return        Reconstruction and convergence diagnostics.
 * @throws std::runtime_error if any row fails to converge inside
 *         `solver.max_iters`.
 */
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    const SolverConfig&                                                           solver);

/**
 * @brief Inverse spatial rank with caller-controlled sampling.
 *
 * When `sampling.reference_size` is 0 or `>= y.rows()`, the solver sees
 * all of `y` (current behavior). Otherwise it sees an R-sized subsample
 * deterministic in `sampling.seed`.
 *
 * @param u         Target rank vectors.
 * @param y         Reference cloud.
 * @param solver    Inner-solver configuration.
 * @param sampling  Reference-subset configuration.
 * @return          Reconstruction and convergence diagnostics.
 */
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    const SolverConfig&                                                           solver,
    const SamplingConfig&                                                         sampling);

/**
 * @brief Legacy overload: pass `tol` / `max_iters` directly.
 *
 * Wraps the `SolverConfig` form with `kind = Weiszfeld` for
 * source-compatibility with pre-v3 callers.
 */
InverseRankResult inverse_spatial_rank(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& u,
    const Sample&                                                                 y,
    double                                                                        tol = 1e-9,
    std::size_t                                                                   max_iters = 200);

}  // namespace mqces
