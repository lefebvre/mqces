#include <mqces/classify.hpp>
#include <mqces/nota.hpp>

#include <mqces/detail/classify_internal.hpp>
#include <mqces/detail/stats.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <span>

namespace mqces::classic {

ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options)
{
    detail::validate_classify_inputs(test, known, options);

    const auto K = static_cast<Eigen::Index>(known.size());
    const auto N = static_cast<Eigen::Index>(options.uncertainty.mc_samples);

    const Eigen::MatrixXd scores = detail::collect_replicate_scores(test, known, options);

    // Per-class mean and variance.
    Eigen::VectorXd means(K);
    Eigen::VectorXd vars(K);
    for (Eigen::Index k = 0; k < K; ++k) {
        means(k) = detail::row_mean(scores, k);
        vars(k)  = detail::row_variance(scores, k);
    }

    // Best class = argmin(mean).
    Eigen::Index best_idx = 0;
    means.minCoeff(&best_idx);

    // Build sorted score list.
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

    // Eqs. 5-7: t-statistic against the runner-up. With sorted scores the
    // runner-up is ranking[1] (always present because validate enforces
    // known.size() >= 1; for K == 1 we set probability to 0 — no alternative
    // to misclassify into).
    if (K >= 2) {
        const Eigen::Index runner_up = ranking[1].second;
        const double       diff      = means(runner_up) - means(best_idx);
        const double       denom     = std::sqrt(
            (vars(runner_up) + vars(best_idx)) / static_cast<double>(N));
        if (denom > 0.0 && std::isfinite(denom)) {
            const double t = diff / denom;
            const double df = static_cast<double>(N - 1);
            // Paper's Eq. 7: 1 - Pr(T < t). Equivalently Pr(T >= t).
            result.misclassification_prob = 1.0 - detail::students_t_cdf(t, df);
        } else {
            // Zero spread — the two means agree exactly. Treat as 50/50.
            result.misclassification_prob = (diff == 0.0) ? 0.5 : 0.0;
        }
    }

    // NOTA check uses the same MC scores plus a within-class reference.
    Eigen::VectorXd best_row = scores.row(best_idx).transpose();
    result.none_of_the_above = is_none_of_the_above(
        test, known[static_cast<std::size_t>(best_idx)], best_row, options);

    return result;
}

}  // namespace mqces::classic
