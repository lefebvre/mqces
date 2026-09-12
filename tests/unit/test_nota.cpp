#include "../support/random_samples.hpp"

#include <mqces/classify.hpp>
#include <mqces/nota.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::is_none_of_the_above;
using mqces::Sample;
using mqces::test_support::gaussian_sample;
using mqces::test_support::make_class;

namespace {

ClassifierOptions default_options(int d) {
  ClassifierOptions opts;
  opts.weights = FeatureWeights::Ones(d);
  opts.uncertainty.seed = 9999;
  opts.nota_threshold = 0.05;
  opts.nota_permutations = 99;
  return opts;
}

// Fraction of `trials` single-class classifications that report NOTA when the
// test sample is drawn from N(shift, 1) and the class from N(0, 1).
double nota_rate(double shift, int trials, std::uint64_t seed) {
  constexpr int n = 20;
  constexpr int d = 3;
  std::mt19937_64 gen(seed);
  auto opts = default_options(d);
  int fired = 0;
  for (int t = 0; t < trials; ++t) {
    std::vector<Class> known{{"k", gaussian_sample(n, d, 0.0, 1.0, gen)}};
    Sample test = gaussian_sample(n, d, shift, 1.0, gen);
    opts.uncertainty.seed = seed + static_cast<std::uint64_t>(t);
    if (mqces::classic::classify(test, std::span<const Class>{known}, opts).none_of_the_above) {
      ++fired;
    }
  }
  return static_cast<double>(fired) / trials;
}

}  // namespace

// nota_permutations == 0 disables the test regardless of the data.
TEST(NOTA, DisabledWithZeroPermutations) {
  Class c = make_class("k", 20, 2, 0.0, 0.3, 1);
  Sample test = gaussian_sample(20, 2, 50.0, 1.0, 2);
  auto opts = default_options(2);
  opts.nota_permutations = 0;
  const double observed = mqces::similarity_score(test, c.specimens, opts.weights);
  EXPECT_FALSE(is_none_of_the_above(test, c, observed, opts));
}

// A test sample drawn from exactly the class distribution is accepted at the
// default threshold.
TEST(NOTA, AcceptsGenuineMember) {
  constexpr int d = 3;
  std::vector<Class> known{make_class("k", 30, d, 0.0, 1.0, 11)};
  Sample test = gaussian_sample(30, d, 0.0, 1.0, 12);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, default_options(d));
  EXPECT_FALSE(res.none_of_the_above);
}

// Calibration: over many genuine members the rejection rate matches the
// nominal threshold. The permutation p-value is exact, so the expected rate
// is 5%; the bound allows ~3 binomial standard deviations over 100 trials.
TEST(NOTA, FalsePositiveRateMatchesThreshold) {
  EXPECT_LE(nota_rate(0.0, 100, 0xA11CE), 0.12);
}

// Power: a one-standard-deviation shift in every feature is detected almost
// always at n = 20.
TEST(NOTA, DetectsShiftedDistribution) {
  EXPECT_GE(nota_rate(1.0, 50, 0xB0B), 0.9);
}

// When the test is drawn from a wildly different distribution than the only
// "matched" class, NOTA should fire.
TEST(NOTA, RejectsClearOutlier) {
  constexpr int d = 3;
  std::vector<Class> known{make_class("k", 30, d, 0.0, 0.3, 21)};
  Sample test = gaussian_sample(30, d, 50.0, 5.0, 22);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, default_options(d));
  EXPECT_TRUE(res.none_of_the_above);
}

TEST(NOTA, ThrowsOnFeatureMismatch) {
  Class c = make_class("k", 10, 3, 0.0, 1.0, 1);
  Sample test = gaussian_sample(10, 2, 0.0, 1.0, 2);
  EXPECT_THROW(is_none_of_the_above(test, c, 1.0, default_options(2)), std::invalid_argument);
}
