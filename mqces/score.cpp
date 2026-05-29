#include <mqces/score.hpp>

#include <mqces/quantile.hpp>

#include <Eigen/Core>

#include <stdexcept>
#include <string>

namespace mqces {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Weighted squared distance Σ_i (a_i − b_i)ᵀ diag(w) (a_i − b_i) over all rows.
double weighted_sq_distance(
    const RowMajorMatrix& a, const RowMajorMatrix& b, const FeatureWeights& w)
{
    RowMajorMatrix diff = a - b;
    return (diff.array().square().matrix() * w).sum();
}

// Default SolverConfig used by the single-argument similarity_score
// overload. Plain Weiszfeld plateaus around 1e-6 to 1e-7 when iterates
// approach a y_i; the loose tol matches the precision of the downstream
// sum-of-weighted-squared-distances score and avoids spurious failures.
// Callers wanting a tighter solution (1e-9) should pass an explicit
// SolverConfig with kind=VardiZhang or VardiZhangAA.
SolverConfig score_default_solver()
{
    SolverConfig c;
    c.kind      = SolverKind::Weiszfeld;
    c.tol       = 1e-5;
    c.max_iters = 1000;
    return c;
}

}  // namespace

double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights,
                        const SolverConfig& solver, const SamplingConfig& sampling)
{
    if (x.rows() == 0 || y.rows() == 0) {
        throw std::invalid_argument(
            "similarity_score: both samples must have at least one row (got x.rows()="
            + std::to_string(x.rows()) + ", y.rows()=" + std::to_string(y.rows()) + ")");
    }
    if (x.cols() == 0) {
        throw std::invalid_argument(
            "similarity_score: samples must have at least one feature column");
    }
    if (x.cols() != y.cols()) {
        throw std::invalid_argument(
            "similarity_score: x.cols() (" + std::to_string(x.cols())
            + ") must equal y.cols() (" + std::to_string(y.cols()) + ")");
    }
    if (weights.size() != x.cols()) {
        throw std::invalid_argument(
            "similarity_score: weights.size() (" + std::to_string(weights.size())
            + ") must equal x.cols() (" + std::to_string(x.cols()) + ")");
    }

    auto u_x = spatial_rank(x, sampling);
    auto u_y = spatial_rank(y, sampling);

    auto x_tilde = inverse_spatial_rank(u_x, y, solver, sampling).x_tilde;
    auto y_tilde = inverse_spatial_rank(u_y, x, solver, sampling).x_tilde;

    return weighted_sq_distance(x_tilde, x, weights)
         + weighted_sq_distance(y_tilde, y, weights);
}

double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights,
                        const SolverConfig& solver)
{
    return similarity_score(x, y, weights, solver, SamplingConfig{});
}

double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights)
{
    return similarity_score(x, y, weights, score_default_solver(), SamplingConfig{});
}

}  // namespace mqces
