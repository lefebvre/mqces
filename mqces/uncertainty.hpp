#pragma once

#include <mqces/types.hpp>

#include <cstdint>

namespace mqces {

/**
 * @brief Multiplicative Gaussian perturbation (Eq. 10 of Weber & Dayman).
 *
 * Applies
 * @f[
 *   X = X^0 \cdot (1 + \varepsilon\, \mathcal{N}(0, 1))
 * @f]
 * element-wise. Returns a fresh `Sample`. `epsilon <= 0` is a no-op
 * and returns a copy of `x0` unchanged.
 *
 * @param x0       Baseline sample (no measurement noise applied).
 * @param epsilon  Relative measurement error fraction.
 * @param seed     RNG stream selector; distinct seeds → independent
 *                 perturbations.
 * @return         Perturbed copy of `x0`.
 */
Sample perturb(const Sample& x0, double epsilon, std::uint64_t seed);

}  // namespace mqces
