// Calibration of the misclassification probability.
//
// Each trial draws fresh specimens for two classes A ~ N(0, 1) and
// B ~ N(shift, 1) and a test sample from A, then classifies. Over many trials
// the mean reported probability should track the observed rate at which B is
// wrongly chosen. The jackknife standard error is somewhat conservative, so
// the reported probability may exceed the observed rate but must never be
// far below it — that direction is the overconfidence these tests guard
// against.

#include "../support/random_samples.hpp"

#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

using mqces::Class;
using mqces::ClassificationResult;
using mqces::ClassifierOptions;
using mqces::Sample;
using mqces::test_support::gaussian_sample;

namespace {

using ClassifyFn = std::function<ClassificationResult(
  const Sample&, std::span<const Class>, const ClassifierOptions&)>;

struct Calibration {
  double error_rate = 0.0;     // fraction of trials choosing B
  double mean_reported = 0.0;  // mean misclassification_prob
};

Calibration run_trials(const ClassifyFn& classify, double shift, int trials, std::uint64_t seed) {
  constexpr int n = 20;
  constexpr int d = 3;

  ClassifierOptions opts;
  opts.weights = Eigen::VectorXd::Ones(d);
  opts.nota_permutations = 0;

  std::mt19937_64 gen(seed);
  Calibration c;
  for (int t = 0; t < trials; ++t) {
    std::vector<Class> known{
      {"A", gaussian_sample(n, d, 0.0, 1.0, gen)},
      {"B", gaussian_sample(n, d, shift, 1.0, gen)},
    };
    Sample test = gaussian_sample(n, d, 0.0, 1.0, gen);
    const auto r = classify(test, std::span<const Class>{known}, opts);
    c.error_rate += r.best_class == "B" ? 1.0 : 0.0;
    c.mean_reported += r.misclassification_prob;
  }
  c.error_rate /= trials;
  c.mean_reported /= trials;
  return c;
}

struct Variant {
  std::string name;
  ClassifyFn classify;
};

class Calibrated : public ::testing::TestWithParam<Variant> {};

}  // namespace

// Classes 0.3 standard deviations apart are confused roughly a third of the
// time. The reported probability must reflect that; before it included
// specimen sampling variability it came out around 1e-10.
TEST_P(Calibrated, CloseClasses) {
  const auto c = run_trials(GetParam().classify, 0.3, 100, 0xCA11B);
  EXPECT_GT(c.error_rate, 0.15);  // the scenario is genuinely ambiguous
  EXPECT_NEAR(c.mean_reported, c.error_rate, 0.1);
}

// Moderately separated classes: rare errors, small but honest probabilities.
TEST_P(Calibrated, SeparatedClasses) {
  const auto c = run_trials(GetParam().classify, 1.0, 100, 0x5E9A);
  EXPECT_LT(c.error_rate, 0.05);
  EXPECT_LT(c.mean_reported, 0.06);
  EXPECT_GE(c.mean_reported, c.error_rate - 0.02);
}

INSTANTIATE_TEST_SUITE_P(Variants,
                         Calibrated,
                         ::testing::Values(Variant{"classic", &mqces::classic::classify},
                                           Variant{"v2", &mqces::v2::classify}),
                         [](const ::testing::TestParamInfo<Variant>& info) {
                           return info.param.name;
                         });
