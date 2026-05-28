#pragma once

#include <mqces/types.hpp>

namespace mqces {

// Similarity score S_XY (Eq. 3 of Weber & Dayman):
//
//   S_XY = Σ_i (x̃_i − x_i)ᵀ W (x̃_i − x_i)
//        + Σ_i (ỹ_i − y_i)ᵀ W (ỹ_i − y_i)
//
// where x̃ are the inverse-spatial-rank reconstructions of x's ranks against
// y (and vice versa for ỹ), and W = diag(weights). Lower is better; 0 means
// the two samples are indistinguishable under the rank metric.
//
// The single-argument overload uses an internal SolverConfig tuned for the
// score's accumulated-error budget (loose tol, generous iter cap). The
// two-argument overload lets the caller override the inner solver — used
// by mqces::v3 (VardiZhang) and mqces::v4 (VardiZhangAA) once they land.
//
// Throws std::invalid_argument if `weights.size() != x.cols()` or the two
// samples have different feature counts.
double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights);

double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights,
                        const SolverConfig& solver);

}  // namespace mqces
