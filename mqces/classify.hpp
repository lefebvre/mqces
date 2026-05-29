#pragma once

#include <mqces/types.hpp>

#include <span>

namespace mqces {

/**
 * @brief Paper-faithful classifier (Weber & Dayman, Eqs. 5-7).
 *
 * Uses the independent-samples t-statistic of Eq. 6 for the
 * misclassification probability:
 * @f[
 *   t = \frac{\bar{X}_j - \bar{X}_k}{\sqrt{(s_j^2 + s_k^2)/N}},\quad
 *   df = N - 1,
 * @f]
 * reported as @f$1 - \Pr(T < t)@f$. The inner solver is plain Weiszfeld
 * with the loose tolerance the paper uses.
 *
 * @param test     Test sample (n_specimens × n_features).
 * @param known    Known classes; each must share `n_features` with `test`.
 * @param options  Classifier options; `options.weights` length must equal
 *                 the per-class feature count.
 * @return         Best class, sorted per-class scores, and
 *                 misclassification probability vs the runner-up.
 */
namespace classic {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace classic

/**
 * @brief Corrected-statistics classifier (paired-difference t).
 *
 * Same inputs and result shape as `classic`. Differences:
 *
 * - **t-statistic**: paired-difference form with sign preserved:
 *   @f$d_i = S(T, X_{j,i}) - S(T, X_{k,i})@f$,
 *   @f$t = \mathrm{mean}(d) / (\mathrm{sd}(d) / \sqrt{N})@f$,
 *   reported as @f$\Pr(T < 0)@f$ — "probability the runner-up was
 *   actually better on a random replicate".
 * - **Inner solver**: still plain Weiszfeld (loose).
 *
 * @param test     Test sample.
 * @param known    Known classes.
 * @param options  Classifier options.
 * @return         Classification result.
 */
namespace v2 {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace v2

/**
 * @brief Paired-difference t with Vardi-Zhang inner solver.
 *
 * Identical statistics to `v2`. The inner solver is Vardi-Zhang
 * (Vardi & Zhang 2000), which closes Weiszfeld's residual-plateau
 * pathology with a subgradient correction at the vertices @f$y_i@f$.
 * Result: scores reproducible to `~1e-9` regardless of whether iterates
 * hover near a vertex.
 *
 * Caller-provided `options.solver` is honored, defaulting to a tight VZ
 * config when unset. MC, NOTA, and sampling behavior match v2.
 *
 * @param test     Test sample.
 * @param known    Known classes.
 * @param options  Classifier options.
 * @return         Classification result.
 */
namespace v3 {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace v3

/**
 * @brief Paired-difference t with Vardi-Zhang + Anderson acceleration.
 *
 * Same statistics and subgradient correction as `v3`, plus Type-II
 * Anderson acceleration (AA) over a rolling window of past iterates.
 * AA converts VZ's linear convergence to superlinear when the problem
 * is amenable; the safeguarded fallback (Toth-Kelley 2015) reverts to
 * plain VZ whenever an accelerated step inflates the residual, so
 * divergence is impossible.
 *
 * This is the production-target variant at @f$N = 10^6@f$ scale.
 *
 * @param test     Test sample.
 * @param known    Known classes.
 * @param options  Classifier options.
 * @return         Classification result.
 */
namespace v4 {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace v4

}  // namespace mqces
