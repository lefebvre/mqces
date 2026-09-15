#include <mqces/classify.hpp>
#include <mqces/detail/classify_internal.hpp>
#include <mqces/detail/stats.hpp>
#include <mqces/nota.hpp>

#include <Eigen/Core>

#include <cmath>
#include <span>

// v2 differences vs classic:
//
//   * The t-statistic is computed on the paired difference
//       d = S(T, X_runner-up) − S(T, X_best).
//     Both scores share the test sample, so they are NOT independent: every
//     test-row jackknife replicate and every perturbation replicate moves
//     both at once. v2 takes the variance of d replicate by replicate, which
//     accounts for the covariance between the two scores; classic's
//     s_j² + s_k² assumes it is zero. The covariance can have either sign —
//     leaving out a test row near one class can raise the other class's
//     score — so neither variant is uniformly more confident than the other.
//   * Reported probability is Pr(T ≤ −t) — the probability that the
//     runner-up would have scored at least as low as the chosen best. This
//     preserves the sign of the difference (classic's 1 - cdf(t) inverts
//     the geometric meaning when the reported "best" actually has a higher
//     score than the runner-up, which classic can never observe — but it
//     does matter for downstream NOTA reasoning).

namespace mqces::v2 {

ClassificationResult classify(const Sample& test,
                              std::span<const Class> known,
                              const ClassifierOptions& options) {
  detail::validate_classify_inputs(test, known, options);

  auto ranking = detail::rank_classes(test, known, options);
  auto& result = ranking.result;
  const auto best = ranking.best;
  const auto runner = ranking.runner_up;

  if (known.size() >= 2) {
    const auto rep_best = detail::collect_replicates(test, known[best], best, options);
    const auto rep_runner = detail::collect_replicates(test, known[runner], runner, options);

    // Class rows enter only their own score, so those terms add; test rows
    // and perturbations are shared, so their replicates are differenced.
    const double var_d = detail::jackknife_variance(rep_runner.drop_test - rep_best.drop_test) +
                         detail::jackknife_variance(rep_runner.drop_class) +
                         detail::jackknife_variance(rep_best.drop_class) +
                         detail::sample_variance(rep_runner.perturbed - rep_best.perturbed);

    const double mean_d = ranking.observed(static_cast<Eigen::Index>(runner)) -
                          ranking.observed(static_cast<Eigen::Index>(best));
    const double se = std::sqrt(var_d);
    if (se > 0.0 && std::isfinite(se)) {
      const double t = mean_d / se;
      const double df = detail::comparison_df(test, known[best], known[runner]);
      // Pr(T ≤ −t): with mean_d > 0 (runner-up worse) this is the left
      // tail; mean_d < 0 is impossible since best has the lowest score.
      result.misclassification_prob = detail::students_t_cdf(-t, df);
    } else {
      result.misclassification_prob = (mean_d == 0.0) ? 0.5 : 0.0;
    }
  }

  result.none_of_the_above = is_none_of_the_above(
    test, known[best], ranking.observed(static_cast<Eigen::Index>(best)), options);
  return result;
}

}  // namespace mqces::v2
