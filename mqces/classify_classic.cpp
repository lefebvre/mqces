#include <mqces/classify.hpp>
#include <mqces/nota.hpp>

#include <mqces/detail/classify_internal.hpp>
#include <mqces/detail/stats.hpp>

#include <Eigen/Core>

#include <cmath>
#include <span>

namespace mqces::classic {

ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options)
{
    detail::validate_classify_inputs(test, known, options);

    auto        ranking = detail::rank_classes(test, known, options);
    auto&       result  = ranking.result;
    const auto  best    = ranking.best;
    const auto  runner  = ranking.runner_up;

    // Eqs. 5-7: t-statistic against the runner-up, treating the two scores as
    // independent. With K == 1 there is no alternative to misclassify into
    // and the probability stays 0.
    if (known.size() >= 2) {
        const auto rep_best   = detail::collect_replicates(test, known[best], best, options);
        const auto rep_runner = detail::collect_replicates(test, known[runner], runner, options);

        const auto score_variance = [](const detail::Replicates& rep) {
            return detail::jackknife_variance(rep.drop_test)
                 + detail::jackknife_variance(rep.drop_class)
                 + detail::sample_variance(rep.perturbed);
        };

        const double diff = ranking.observed(static_cast<Eigen::Index>(runner))
                          - ranking.observed(static_cast<Eigen::Index>(best));
        const double denom = std::sqrt(score_variance(rep_runner) + score_variance(rep_best));
        if (denom > 0.0 && std::isfinite(denom)) {
            const double t  = diff / denom;
            const double df = detail::comparison_df(test, known[best], known[runner]);
            // Paper's Eq. 7: 1 - Pr(T < t). Equivalently Pr(T >= t).
            result.misclassification_prob = 1.0 - detail::students_t_cdf(t, df);
        } else {
            // Zero spread — only possible when every specimen is identical.
            result.misclassification_prob = (diff == 0.0) ? 0.5 : 0.0;
        }
    }

    result.none_of_the_above = is_none_of_the_above(
        test, known[best], ranking.observed(static_cast<Eigen::Index>(best)), options);
    return result;
}

}  // namespace mqces::classic
