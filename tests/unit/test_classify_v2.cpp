#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include "../support/random_samples.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <span>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::Sample;
using mqces::test_support::gaussian_sample;
using mqces::test_support::make_class;

namespace {

ClassifierOptions default_options(int d)
{
    ClassifierOptions opts;
    opts.weights                 = FeatureWeights::Ones(d);
    opts.uncertainty.epsilon     = 0.0;
    opts.uncertainty.mc_samples  = 5;
    opts.uncertainty.seed        = 4321;
    opts.n_threads               = 1;
    return opts;
}

}  // namespace

TEST(ClassifyV2, PicksCorrectClassOnSeparatedGaussians)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("near", 20, d,  0.0, 0.5, 1),
        make_class("far",  20, d, 10.0, 0.5, 2),
    };
    Sample test = gaussian_sample(20, d, 0.0, 0.5, 3);
    auto   res  = mqces::v2::classify(test, std::span<const Class>{known}, default_options(d));
    EXPECT_EQ(res.best_class, "near");
    EXPECT_LT(res.misclassification_prob, 1e-3);
}

// Close classes: sampling variability alone keeps the probability well away
// from zero even with epsilon == 0.
TEST(ClassifyV2, ZeroEpsilonStillReportsUncertainty)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, 0.0, 1.0, 5),
        make_class("B", 20, d, 0.3, 1.0, 6),
    };
    Sample test = gaussian_sample(20, d, 0.0, 1.0, 7);
    auto   res  = mqces::v2::classify(test, std::span<const Class>{known}, default_options(d));
    EXPECT_GT(res.misclassification_prob, 0.05);
    EXPECT_LE(res.misclassification_prob, 0.5);
}

// Two classes with identical specimens tie exactly: every paired replicate
// difference is zero, so the probability is exactly one half.
TEST(ClassifyV2, IdenticalClassesAreACoinFlip)
{
    constexpr int d = 2;
    Class              a = make_class("A", 15, d, 0.0, 1.0, 5);
    std::vector<Class> known{a, Class{"A-copy", a.specimens}};
    Sample             test = gaussian_sample(15, d, 0.0, 1.0, 7);
    auto               res  = mqces::v2::classify(test, std::span<const Class>{known},
                                                  default_options(d));
    EXPECT_EQ(res.misclassification_prob, 0.5);
}

TEST(ClassifyV2, SingleClassReportsZeroMisclassProb)
{
    constexpr int d = 2;
    std::vector<Class> known{make_class("only", 15, d, 0.0, 1.0, 8)};
    Sample             test = gaussian_sample(15, d, 0.0, 1.0, 9);
    auto               res  = mqces::v2::classify(
        test, std::span<const Class>{known}, default_options(d));
    EXPECT_EQ(res.best_class, "only");
    EXPECT_EQ(res.misclassification_prob, 0.0);
}
