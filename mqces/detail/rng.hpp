#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <utility>

// Random-number helpers.
//
// std::mt19937_64's output sequence is fixed by the standard, but the
// algorithms behind std::normal_distribution, std::uniform_int_distribution
// and std::shuffle are not: libstdc++, libc++ and MSVC produce different
// values from the same engine state. Everything here is written in terms of
// raw engine output so a given seed yields the same draws on every standard
// library (up to libm rounding in log/cos).

namespace mqces::detail {

// splitmix64 mixer — used to derive uncorrelated 64-bit seeds from an
// integer index.
constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Derive a per-stream seed from a base seed and a stream index. Two
// different stream indices give uncorrelated mt19937_64 sequences,
// which is what we want for the Monte-Carlo replicate loop.
inline std::mt19937_64 make_stream(std::uint64_t base_seed, std::uint64_t stream_index) {
  return std::mt19937_64(splitmix64(base_seed ^ splitmix64(stream_index)));
}

// Uniform double in the open interval (0, 1): the top 53 bits of one engine
// output, offset by half a step so neither endpoint is reachable.
inline double uniform_open01(std::mt19937_64& rng) {
  return (static_cast<double>(rng() >> 11) + 0.5) * 0x1.0p-53;
}

// Standard normal draw via Box-Muller. The second variate is discarded so
// the function stays stateless.
inline double standard_normal(std::mt19937_64& rng) {
  const double u1 = uniform_open01(rng);
  const double u2 = uniform_open01(rng);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
}

// Uniform integer in [0, n) without modulo bias. `n` must be > 0.
inline std::uint64_t uniform_index(std::mt19937_64& rng, std::uint64_t n) {
  // 2^64 mod n: discarding draws below it leaves a multiple of n values.
  const std::uint64_t threshold = (0 - n) % n;
  std::uint64_t x = rng();
  while (x < threshold) {
    x = rng();
  }
  return x % n;
}

// Fisher-Yates shuffle of [first, first + n).
template <class It>
void shuffle(It first, std::size_t n, std::mt19937_64& rng) {
  for (std::size_t i = n; i > 1; --i) {
    const auto j = static_cast<std::size_t>(uniform_index(rng, i));
    using std::swap;
    swap(first[i - 1], first[j]);
  }
}

}  // namespace mqces::detail
