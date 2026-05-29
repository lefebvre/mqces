#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

namespace mqces {

/**
 * @brief None-of-the-above decision (Weber & Dayman, page 8).
 *
 * Given the chosen best class and the N test-vs-best Monte-Carlo scores
 * already computed by the classifier, build a within-class reference
 * set of N scores (best.specimens split into halves, each half
 * perturbed, `similarity_score` between the halves) and compare the two
 * distributions via a two-sample t-test on their means.
 *
 * Returns `true` (i.e. *reject the classification*) when the test
 * sample's scores are statistically *larger* than the within-class
 * scores at the configured `options.nota_threshold` p-value.
 *
 * Returns `false` (*accept the classification*) when:
 * - `best.specimens` has fewer than 4 rows (cannot split meaningfully),
 * - either distribution has zero spread (no statistical signal),
 * - the test-vs-best mean is below the within-class mean (better match
 *   than the class is to itself — definitely not NOTA).
 *
 * @param test                  Test sample.
 * @param best                  The winning class.
 * @param test_vs_best_scores   The per-MC similarity scores between
 *                              `test` and `best.specimens` already
 *                              computed by the classifier.
 * @param options               Carries `nota_threshold`, `uncertainty`,
 *                              and other shared knobs.
 * @return                      `true` if the test sample is NOTA.
 */
bool is_none_of_the_above(
    const Sample&             test,
    const Class&              best,
    const Eigen::VectorXd&    test_vs_best_scores,
    const ClassifierOptions&  options);

}  // namespace mqces
