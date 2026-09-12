#pragma once

// Gaussian sample generators shared by the unit tests.
//
// Draws go through mqces::detail::standard_normal rather than
// std::normal_distribution, whose algorithm the standard leaves unspecified:
// a test built on these sees the same data on every standard library, so a
// pass/fail outcome that depends on the particular draw is portable.

#include <mqces/detail/rng.hpp>
#include <mqces/types.hpp>

#include <cstdint>
#include <random>
#include <string>

namespace mqces::test_support {

// n specimens in d features, each coordinate drawn from N(center, sigma²).
inline Sample gaussian_sample(int n, int d, double center, double sigma, std::mt19937_64& gen) {
  Sample s(n, d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      s(i, j) = center + sigma * detail::standard_normal(gen);
    }
  }
  return s;
}

inline Sample gaussian_sample(int n, int d, double center, double sigma, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  return gaussian_sample(n, d, center, sigma, gen);
}

inline Class make_class(
  const std::string& name, int n, int d, double center, double sigma, std::uint64_t seed) {
  return Class{name, gaussian_sample(n, d, center, sigma, seed)};
}

}  // namespace mqces::test_support
