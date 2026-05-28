#include <mqces/score.hpp>

#include <mqces/quantile.hpp>

#include <Eigen/Core>

#include <stdexcept>
#include <string>

namespace mqces {

namespace {

// Weighted squared distance Σ_i (a_i − b_i)ᵀ diag(w) (a_i − b_i) over all rows.
double weighted_sq_distance(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& a,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& b,
    const FeatureWeights&                                                         w)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> diff = a - b;
    return (diff.array().square().matrix() * w).sum();
}

}  // namespace

double similarity_score(const Sample& x, const Sample& y, const FeatureWeights& weights)
{
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

    auto u_x = spatial_rank(x);
    auto u_y = spatial_rank(y);

    // Looser tolerance / higher iteration cap than the inverse_spatial_rank
    // defaults: when x and y are drawn from similar distributions, Weiszfeld
    // plateaus near coincident points at residuals around 1e-6 to 1e-7 that
    // do not shrink with more iterations. Per-coordinate error at that scale
    // is well below the noise level of the score itself (a sum of weighted
    // squared distances over many specimens). A proper fix is to swap the
    // iteration for Vardi-Zhang or to add stagnation detection; that's a
    // separate piece of work — see TODO in detail/nonlinear_solve.hpp.
    constexpr double      kScoreInvRankTol      = 1e-5;
    constexpr std::size_t kScoreInvRankMaxIters = 1000;
    auto x_tilde
        = inverse_spatial_rank(u_x, y, kScoreInvRankTol, kScoreInvRankMaxIters).x_tilde;
    auto y_tilde
        = inverse_spatial_rank(u_y, x, kScoreInvRankTol, kScoreInvRankMaxIters).x_tilde;

    return weighted_sq_distance(x_tilde, x, weights)
         + weighted_sq_distance(y_tilde, y, weights);
}

}  // namespace mqces
