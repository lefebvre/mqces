#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::Sample;

namespace {

// Build a class of n specimens in d features, sampled from N(center, sigma).
Class make_class(const std::string& name, int n, int d, double center,
                 double sigma, std::uint64_t seed)
{
    std::mt19937_64            gen(seed);
    std::normal_distribution<> dist(center, sigma);
    Class                      c;
    c.name      = name;
    c.specimens = Sample(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            c.specimens(i, j) = dist(gen);
        }
    }
    return c;
}

ClassifierOptions default_options(int d)
{
    ClassifierOptions opts;
    opts.weights                 = FeatureWeights::Ones(d);
    opts.uncertainty.epsilon     = 0.0;
    opts.uncertainty.mc_samples  = 5;
    opts.uncertainty.seed        = 1234;
    opts.n_threads               = 1;  // deterministic order in tests
    return opts;
}

}  // namespace

TEST(ClassifyClassic, ThrowsOnEmptyKnownList)
{
    Sample test = Sample::Random(10, 3);
    auto   opts = default_options(3);
    EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{}, opts),
                 std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnWeightDimensionMismatch)
{
    Sample             test = Sample::Random(10, 3);
    std::vector<Class> known{make_class("k", 10, 3, 0.0, 1.0, 1)};
    auto               opts        = default_options(3);
    opts.weights                   = FeatureWeights::Ones(4);
    EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, opts),
                 std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnFeatureMismatchAcrossClass)
{
    Sample             test = Sample::Random(10, 3);
    std::vector<Class> known{make_class("ok", 10, 3, 0.0, 1.0, 1),
                             make_class("bad", 10, 4, 0.0, 1.0, 2)};
    auto               opts = default_options(3);
    EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, opts),
                 std::invalid_argument);
}

TEST(ClassifyClassic, ThrowsOnZeroMcSamples)
{
    Sample             test = Sample::Random(10, 3);
    std::vector<Class> known{make_class("k", 10, 3, 0.0, 1.0, 1)};
    auto               opts                = default_options(3);
    opts.uncertainty.mc_samples            = 0;
    EXPECT_THROW(mqces::classic::classify(test, std::span<const Class>{known}, opts),
                 std::invalid_argument);
}

// With three well-separated Gaussians, classification of a test drawn from
// one of them should pick its own class.
TEST(ClassifyClassic, PicksCorrectClassOnSeparatedGaussians)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, -5.0, 0.5, 11),
        make_class("B", 20, d,  0.0, 0.5, 22),
        make_class("C", 20, d,  5.0, 0.5, 33),
    };
    Sample test = make_class("test", 20, d, 0.0, 0.5, 44).specimens;
    auto   opts = default_options(d);
    auto   res  = mqces::classic::classify(test, std::span<const Class>{known}, opts);
    EXPECT_EQ(res.best_class, "B");
    ASSERT_EQ(res.scores.size(), 3u);
    EXPECT_EQ(res.scores.front().class_name, "B");
    EXPECT_LE(res.scores.front().s_xy, res.scores.back().s_xy);
}

// Result list is sorted ascending by score (best first).
TEST(ClassifyClassic, ScoresAreSortedAscending)
{
    constexpr int d = 2;
    std::vector<Class> known{
        make_class("A", 15, d, -3.0, 0.5, 1),
        make_class("B", 15, d,  0.0, 0.5, 2),
        make_class("C", 15, d,  3.0, 0.5, 3),
    };
    Sample test = make_class("test", 15, d, -3.0, 0.5, 4).specimens;
    auto   res  = mqces::classic::classify(test, std::span<const Class>{known},
                                        default_options(d));
    for (std::size_t i = 1; i < res.scores.size(); ++i) {
        EXPECT_LE(res.scores[i - 1].s_xy, res.scores[i].s_xy);
    }
}

// Misclassification probability is a number in [0, 1].
TEST(ClassifyClassic, MisclassProbInUnitInterval)
{
    constexpr int d = 2;
    std::vector<Class> known{
        make_class("A", 15, d,  0.0, 0.5, 5),
        make_class("B", 15, d,  0.6, 0.5, 6),
    };
    Sample test = make_class("test", 15, d, 0.0, 0.5, 7).specimens;
    auto   res  = mqces::classic::classify(test, std::span<const Class>{known},
                                        default_options(d));
    EXPECT_GE(res.misclassification_prob, 0.0);
    EXPECT_LE(res.misclassification_prob, 1.0);
}

// With K = 1 there is no alternative to misclassify into; report 0.
TEST(ClassifyClassic, SingleClassReportsZeroMisclassProb)
{
    constexpr int d = 2;
    std::vector<Class> known{make_class("only", 15, d, 0.0, 1.0, 8)};
    Sample             test = make_class("test", 15, d, 0.0, 1.0, 9).specimens;
    auto               res  = mqces::classic::classify(
        test, std::span<const Class>{known}, default_options(d));
    EXPECT_EQ(res.best_class, "only");
    EXPECT_EQ(res.misclassification_prob, 0.0);
}
