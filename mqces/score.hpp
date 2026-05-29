#pragma once

#include <mqces/types.hpp>

namespace mqces {

/**
 * @brief Similarity score S_XY between two samples (Eq. 3 of Weber & Dayman).
 *
 * Computes
 * @f[
 *   S_{XY} = \sum_i (\tilde{x}_i - x_i)^\top W (\tilde{x}_i - x_i)
 *          + \sum_i (\tilde{y}_i - y_i)^\top W (\tilde{y}_i - y_i),
 * @f]
 * where @f$\tilde{x}@f$ are the inverse-spatial-rank reconstructions of
 * @f$x@f$'s ranks against @f$y@f$ (and vice versa for @f$\tilde{y}@f$),
 * and @f$W = \mathrm{diag}(\text{weights})@f$. Lower is better; 0 means
 * the two samples are indistinguishable under the rank metric.
 *
 * This overload uses an internal `SolverConfig` tuned for the score's
 * accumulated-error budget (loose tolerance, generous iteration cap).
 *
 * @param x        First sample (N_x × d).
 * @param y        Second sample (N_y × d). Must match `x.cols()`.
 * @param weights  Diagonal of W; length must equal `x.cols()`.
 * @return         Non-negative similarity score.
 * @throws std::invalid_argument if dimensions mismatch.
 */
double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights);

/**
 * @brief Similarity score with a caller-controlled inner solver.
 *
 * Same equation as the single-argument overload; the caller picks the
 * `SolverConfig` used by `inverse_spatial_rank`. Used by `mqces::v3`
 * (Vardi-Zhang) and `mqces::v4` (VardiZhang + Anderson).
 *
 * @param x        First sample (N_x × d).
 * @param y        Second sample (N_y × d).
 * @param weights  Diagonal of W.
 * @param solver   Inner-solver configuration.
 * @return         Non-negative similarity score.
 * @throws std::invalid_argument if dimensions mismatch.
 */
double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights,
                        const SolverConfig& solver);

/**
 * @brief Similarity score with caller-controlled solver and sampling.
 *
 * Full surface: caller controls both the inner solver and the
 * reference-subset sampling used by `spatial_rank` and
 * `inverse_spatial_rank`. The same `sampling` is passed to both
 * directions of the reconstruction so the R-subset of each cloud is
 * consistent across them.
 *
 * @param x         First sample (N_x × d).
 * @param y         Second sample (N_y × d).
 * @param weights   Diagonal of W.
 * @param solver    Inner-solver configuration.
 * @param sampling  Reference-subset configuration; `reference_size == 0`
 *                  dispatches to the exact O(N²) path.
 * @return          Non-negative similarity score.
 * @throws std::invalid_argument if dimensions mismatch.
 */
double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights,
                        const SolverConfig& solver, const SamplingConfig& sampling);

}  // namespace mqces
