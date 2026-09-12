#pragma once

#include <mqces/types.hpp>

#include <span>

namespace mqces {

// Classifies a test sample against a set of known classes. Returns the
// per-class similarity score on the observed data (lower is better), the
// chosen class (lowest score), a misclassification probability against the
// runner-up, and the none-of-the-above decision (see nota.hpp).
//
// Both `classic` and `v2` consume the same inputs, select the same class and
// produce the same result shape; they differ only in how the
// misclassification probability is computed. In both, a score's variance
// combines a delete-one jackknife over the specimens of both samples with,
// when epsilon > 0, the Monte-Carlo variance under measurement error
// (details in detail/classify_internal.hpp).
//
//   classic — the t-statistic form of Weber & Dayman Eqs. 5-7:
//             t = (S_k − S_j) / √(s_j² + s_k²), reported as 1 − Pr(T < t).
//             Departs from the paper in where s² comes from: the paper takes
//             it from Monte-Carlo replicates alone and divides by √N, which
//             captures measurement noise but not specimen sampling
//             variability and is overconfident by orders of magnitude.
//
//   v2      — paired-difference t with sign preserved:
//             d = S_k − S_j, t = d / sd(d), where sd(d) accounts for the test
//             sample being shared by both scores. Reported as Pr(T < −t):
//             the probability that the runner-up is actually the better match.
//
// Both use df = (smallest of the three sample sizes) − 1.
//
// `options.weights` must have length == classes' feature count. Every
// sample needs at least 2 specimens. Throws std::invalid_argument on invalid
// inputs or options, and std::runtime_error if a score's inverse-rank solve
// fails to converge.

namespace classic {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace classic

namespace v2 {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace v2

}  // namespace mqces
