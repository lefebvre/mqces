#include <mqces/classify.hpp>
#include <mqces/nota.hpp>

#include <mqces/detail/classify_internal.hpp>
#include <mqces/detail/stats.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <span>

// v2 differences vs classic:
//
//   * The t-statistic is computed on paired per-replicate differences
//       d_i = S(T, X_{runner-up,i}) − S(T, X_{best,i})
//     using the same MC seeds (so the two scores share their noise
//     realization). This is both more powerful and methodologically
//     correct: in classic, X̄_j and X̄_k are computed against the same
//     test sample and are NOT independent, so the (s_j² + s_k²)/N
//     denominator overstates the standard error of the mean difference.
//   * Reported probability is Pr(T_paired ≤ 0) — the probability that on
//     a fresh replicate the runner-up would have scored at least as low
//     as the chosen best. This preserves the sign of the difference
//     (classic's 1 - cdf(t) inverts the geometric meaning when the
//     reported "best" actually has a higher mean than the runner-up
//     in the test sample, which classic can never observe — but it
//     does matter for downstream NOTA reasoning).

namespace mqces::v2 {

ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options)
{
    detail::validate_classify_inputs(test, known, options);

    const auto K = static_cast<Eigen::Index>(known.size());
    const auto N = static_cast<Eigen::Index>(options.uncertainty.mc_samples);

    const Eigen::MatrixXd scores = detail::collect_replicate_scores(test, known, options);

    Eigen::VectorXd means(K);
    for (Eigen::Index k = 0; k < K; ++k) {
        means(k) = detail::row_mean(scores, k);
    }

    Eigen::Index best_idx = 0;
    means.minCoeff(&best_idx);

    std::vector<std::pair<double, Eigen::Index>> ranking;
    ranking.reserve(static_cast<std::size_t>(K));
    for (Eigen::Index k = 0; k < K; ++k) {
        ranking.emplace_back(means(k), k);
    }
    std::sort(ranking.begin(), ranking.end());

    ClassificationResult result;
    result.scores.reserve(static_cast<std::size_t>(K));
    for (const auto& [m, k] : ranking) {
        result.scores.push_back({known[static_cast<std::size_t>(k)].name, m});
    }
    result.best_class = known[static_cast<std::size_t>(best_idx)].name;

    if (K >= 2) {
        const Eigen::Index runner_up = ranking[1].second;
        // Paired differences: d_i = score_{runner_up,i} − score_{best,i}.
        Eigen::ArrayXd d = scores.row(runner_up).array() - scores.row(best_idx).array();
        const double   mean_d = d.mean();
        double         var_d  = 0.0;
        if (N > 1) {
            var_d = (d - mean_d).square().sum() / static_cast<double>(N - 1);
        }
        const double se = std::sqrt(var_d / static_cast<double>(N));
        if (se > 0.0 && std::isfinite(se)) {
            const double t  = mean_d / se;
            const double df = static_cast<double>(N - 1);
            // Pr(T_paired ≤ 0): probability the runner-up would have scored
            // at least as low as best. With mean_d > 0 (runner-up worse), this
            // is the left tail; with mean_d < 0 (impossible by construction
            // since best has the lowest mean), it would exceed 0.5.
            result.misclassification_prob = detail::students_t_cdf(-t, df);
        } else {
            result.misclassification_prob = (mean_d == 0.0) ? 0.5 : 0.0;
        }
    }

    Eigen::VectorXd best_row = scores.row(best_idx).transpose();
    result.none_of_the_above = is_none_of_the_above(
        test, known[static_cast<std::size_t>(best_idx)], best_row, options);

    return result;
}

}  // namespace mqces::v2
