#include "../support/random_samples.hpp"

#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

using mqces::Class;
using mqces::ClassificationResult;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::Sample;
using mqces::test_support::gaussian_sample;
using mqces::test_support::make_class;

namespace {

ClassifierOptions default_options(int d) {
  ClassifierOptions opts;
  opts.weights = FeatureWeights::Ones(d);
  opts.uncertainty.epsilon = 0.0;
  opts.uncertainty.mc_samples = 5;
  opts.uncertainty.seed = 1234;
  opts.n_threads = 1;  // deterministic order in tests
  return opts;
}

void expect_invalid(const ClassifierOptions& opts) {
  Sample test = gaussian_sample(10, 3, 0.0, 1.0, 5);
  std::vector<Class> known{make_class("k", 10, 3, 0.0, 1.0, 1)};
  EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, opts),
               std::invalid_argument);
}

}  // namespace

TEST(ClassifyClassic, ThrowsOnEmptyKnownList) {
  Sample test = Sample::Random(10, 3);
  auto opts = default_options(3);
  EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{}, opts),
               std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnWeightDimensionMismatch) {
  auto opts = default_options(3);
  opts.weights = FeatureWeights::Ones(4);
  expect_invalid(opts);
}

TEST(ClassifyClassic, ThrowsOnFeatureMismatchAcrossClass) {
  Sample test = Sample::Random(10, 3);
  std::vector<Class> known{make_class("ok", 10, 3, 0.0, 1.0, 1),
                           make_class("bad", 10, 4, 0.0, 1.0, 2)};
  auto opts = default_options(3);
  EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, opts),
               std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnSingleSpecimenClass) {
  Sample test = gaussian_sample(10, 3, 0.0, 1.0, 5);
  std::vector<Class> known{make_class("k", 1, 3, 0.0, 1.0, 1)};
  EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, default_options(3)),
               std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnTooFewMcSamples) {
  auto opts = default_options(3);
  opts.uncertainty.mc_samples = 0;
  expect_invalid(opts);
  opts.uncertainty.mc_samples = 1;
  expect_invalid(opts);
}

TEST(ClassifyClassic, ThrowsOnInvalidEpsilon) {
  auto opts = default_options(3);
  opts.uncertainty.epsilon = -0.01;
  expect_invalid(opts);
  opts.uncertainty.epsilon = std::numeric_limits<double>::quiet_NaN();
  expect_invalid(opts);
}

TEST(ClassifyClassic, ThrowsOnNotaThresholdOutsideUnitInterval) {
  auto opts = default_options(3);
  for (double bad : {0.0, 1.0, -0.5, 2.0, std::numeric_limits<double>::quiet_NaN()}) {
    opts.nota_threshold = bad;
    expect_invalid(opts);
  }
}

// With 9 permutations the smallest p-value is 0.1, which can never fall
// below 0.05: reject the configuration rather than silently never firing.
TEST(ClassifyClassic, ThrowsWhenNotaCanNeverFire) {
  auto opts = default_options(3);
  opts.nota_threshold = 0.05;
  opts.nota_permutations = 9;
  expect_invalid(opts);
}

// Boundary of the check above. NOTA fires only when p < nota_threshold, so
// with 19 permutations the smallest p-value, 1/20, equals the threshold and
// can never fire: rejected. With 20 permutations the smallest p-value, 1/21,
// is below it: accepted, and NOTA does fire on a clear outlier.
TEST(ClassifyClassic, NotaPermutationBoundary) {
  auto opts = default_options(3);
  opts.nota_threshold = 0.05;
  opts.nota_permutations = 19;
  expect_invalid(opts);

  opts.nota_permutations = 20;
  std::vector<Class> known{make_class("k", 20, 3, 0.0, 0.3, 21)};
  Sample outlier = gaussian_sample(20, 3, 50.0, 5.0, 22);
  ClassificationResult res;
  EXPECT_NO_THROW(res = mqces::classic::classify(outlier, std::span<const Class>{known}, opts));
  EXPECT_TRUE(res.none_of_the_above);
}

// With three well-separated Gaussians, classification of a test drawn from
// one of them should pick its own class.
TEST(ClassifyClassic, PicksCorrectClassOnSeparatedGaussians) {
  constexpr int d = 3;
  std::vector<Class> known{
    make_class("A", 20, d, -5.0, 0.5, 11),
    make_class("B", 20, d, 0.0, 0.5, 22),
    make_class("C", 20, d, 5.0, 0.5, 33),
  };
  Sample test = gaussian_sample(20, d, 0.0, 0.5, 44);
  auto opts = default_options(d);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, opts);
  EXPECT_EQ(res.best_class, "B");
  ASSERT_EQ(res.scores.size(), 3u);
  EXPECT_EQ(res.scores.front().class_name, "B");
  EXPECT_LE(res.scores.front().s_xy, res.scores.back().s_xy);
  EXPECT_LT(res.misclassification_prob, 1e-3);
}

// Result list is sorted ascending by score (best first).
TEST(ClassifyClassic, ScoresAreSortedAscending) {
  constexpr int d = 2;
  std::vector<Class> known{
    make_class("A", 15, d, -3.0, 0.5, 1),
    make_class("B", 15, d, 0.0, 0.5, 2),
    make_class("C", 15, d, 3.0, 0.5, 3),
  };
  Sample test = gaussian_sample(15, d, -3.0, 0.5, 4);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, default_options(d));
  for (std::size_t i = 1; i < res.scores.size(); ++i) {
    EXPECT_LE(res.scores[i - 1].s_xy, res.scores[i].s_xy);
  }
}

// The misclassification probability must not depend on epsilon == 0 being a
// special case: sampling variability alone yields an informative value.
TEST(ClassifyClassic, ZeroEpsilonStillReportsUncertainty) {
  constexpr int d = 3;
  std::vector<Class> known{
    make_class("A", 20, d, 0.0, 1.0, 5),
    make_class("B", 20, d, 0.3, 1.0, 6),
  };
  Sample test = gaussian_sample(20, d, 0.0, 1.0, 7);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, default_options(d));
  EXPECT_GT(res.misclassification_prob, 0.05);
  EXPECT_LE(res.misclassification_prob, 0.5);
}

// Measurement error only adds variance: a larger epsilon cannot make the
// classifier more confident.
TEST(ClassifyClassic, MeasurementErrorWidensUncertainty) {
  constexpr int d = 3;
  std::vector<Class> known{
    make_class("A", 20, d, 10.0, 1.0, 5),
    make_class("B", 20, d, 10.6, 1.0, 6),
  };
  Sample test = gaussian_sample(20, d, 10.0, 1.0, 7);
  auto opts = default_options(d);
  opts.uncertainty.mc_samples = 20;
  const auto p0 =
    mqces::classic::classify(test, std::span<const Class>{known}, opts).misclassification_prob;
  opts.uncertainty.epsilon = 0.2;
  const auto p1 =
    mqces::classic::classify(test, std::span<const Class>{known}, opts).misclassification_prob;
  EXPECT_GT(p1, p0);
}

// With K = 1 there is no alternative to misclassify into; report 0.
TEST(ClassifyClassic, SingleClassReportsZeroMisclassProb) {
  constexpr int d = 2;
  std::vector<Class> known{make_class("only", 15, d, 0.0, 1.0, 8)};
  Sample test = gaussian_sample(15, d, 0.0, 1.0, 9);
  auto res = mqces::classic::classify(test, std::span<const Class>{known}, default_options(d));
  EXPECT_EQ(res.best_class, "only");
  EXPECT_EQ(res.misclassification_prob, 0.0);
}

// Results do not depend on the thread count.
TEST(ClassifyClassic, ThreadCountDoesNotChangeResult) {
  constexpr int d = 3;
  std::vector<Class> known{
    make_class("A", 20, d, 0.0, 1.0, 5),
    make_class("B", 20, d, 0.5, 1.0, 6),
  };
  Sample test = gaussian_sample(20, d, 0.0, 1.0, 7);
  auto opts = default_options(d);
  opts.uncertainty.epsilon = 0.05;
  const auto r1 = mqces::classic::classify(test, std::span<const Class>{known}, opts);
  opts.n_threads = 4;
  const auto r4 = mqces::classic::classify(test, std::span<const Class>{known}, opts);
  EXPECT_EQ(r1.misclassification_prob, r4.misclassification_prob);
  EXPECT_EQ(r1.none_of_the_above, r4.none_of_the_above);
  ASSERT_EQ(r1.scores.size(), r4.scores.size());
  for (std::size_t i = 0; i < r1.scores.size(); ++i) {
    EXPECT_EQ(r1.scores[i].s_xy, r4.scores[i].s_xy);
  }
}
