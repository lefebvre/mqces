#pragma once

#include <mqces/types.hpp>

#include <span>

namespace mqces {

// Classifies a test sample against a set of known classes. Returns the
// per-class average similarity score (lower is better), the chosen class,
// and a misclassification probability against the runner-up.
//
// Both `classic` and `v2` consume the same inputs and produce the same
// result shape; they differ only in how the misclassification probability
// is computed:
//
//   classic — Eqs. 5-7 of Weber & Dayman verbatim:
//             t = (X̄_j − X̄_k) / √((s_j² + s_k²) / N), df = N − 1,
//             reported as 1 − Pr(T < t).
//
//   v2      — paired-difference t with sign preserved:
//             d_i = S(T, X_{j,i}) − S(T, X_{k,i}),  t = mean(d) / (sd(d)/√N),
//             df = N − 1, reported as Pr(T < 0) so the value is
//             unambiguously "probability that the runner-up was actually
//             better on a random replicate."
//
// `options.weights` must have length == classes' feature count.

namespace classic {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace classic

namespace v2 {
ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options);
}  // namespace v2

}  // namespace mqces
