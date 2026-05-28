#pragma once

#include <mqces/types.hpp>

#include <cstdint>

namespace mqces {

// Multiplicative Gaussian perturbation (Eq. 10 of Weber & Dayman):
//     X = X^0 * (1 + ε N(0, 1))
// applied element-wise. Returns a fresh Sample. `epsilon <= 0` is a no-op
// and returns a copy of `x0` unchanged.
//
// `seed` deterministically selects the RNG stream; pass distinct seeds for
// different Monte-Carlo replicates to obtain independent perturbations.
Sample perturb(const Sample& x0, double epsilon, std::uint64_t seed);

}  // namespace mqces
