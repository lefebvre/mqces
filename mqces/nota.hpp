#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

namespace mqces {

// None-of-the-above decision (Weber & Dayman, page 8).
//
// Given the chosen best class and the N test-vs-best Monte-Carlo scores
// already computed by the classifier, build a within-class reference set
// of N scores (best.specimens split into halves, each half perturbed,
// similarity_score between the halves) and compare the two distributions
// via a two-sample t-test on their means. If the test sample's scores are
// statistically *larger* than the within-class scores at the configured
// `nota_threshold` p-value, the test sample does not look like a typical
// draw from the chosen class — return true.
//
// Returns false (i.e. "accept the classification") when:
//   - best.specimens has fewer than 4 rows (cannot split meaningfully),
//   - the two distributions have zero spread (no statistical signal),
//   - the test-vs-best mean is below the within-class mean (better match
//     than the class is to itself — definitely not NOTA).
bool is_none_of_the_above(
    const Sample&             test,
    const Class&              best,
    const Eigen::VectorXd&    test_vs_best_scores,
    const ClassifierOptions&  options);

}  // namespace mqces
