#pragma once

#include <mqces/types.hpp>

namespace mqces {

// None-of-the-above decision (Weber & Dayman, page 8).
//
// Null hypothesis: the test specimens and the chosen class's specimens are
// draws from one distribution. Under that null the rows of the two samples
// are exchangeable, so the observed score S(test, best) is compared against
// the scores of `options.nota_permutations` random re-splits of the pooled
// rows into groups of the original sizes. The p-value is
//
//     p = (1 + #{permuted scores >= observed}) / (1 + nota_permutations)
//
// which is exact at any sample size — no variance estimate or distributional
// assumption is involved. Returns true (does not belong) when
// p < options.nota_threshold, false otherwise, and false when
// options.nota_permutations == 0.
//
// `observed_score` must be similarity_score(test, best.specimens,
// options.weights). Measurement-error perturbation is not applied: it is
// already present in the observed specimens, and perturbing only the
// permuted splits would bias the test toward acceptance.
bool is_none_of_the_above(const Sample& test,
                          const Class& best,
                          double observed_score,
                          const ClassifierOptions& options);

}  // namespace mqces
