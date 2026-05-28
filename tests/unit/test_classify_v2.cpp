#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>
#include <span>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::Sample;

namespace {

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
    opts.uncertainty.seed        = 4321;
    opts.n_threads               = 1;
    return opts;
}

}  // namespace

// v2 should agree with classic on best-class selection — the mean ordering
// is identical (both consume the same MC score matrix).
TEST(ClassifyV2, BestClassAgreesWithClassic)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, -5.0, 0.5, 11),
        make_class("B", 20, d,  0.0, 0.5, 22),
        make_class("C", 20, d,  5.0, 0.5, 33),
    };
    Sample test = make_class("test", 20, d, 0.0, 0.5, 44).specimens;
    auto   opts = default_options(d);
    auto   r_classic = mqces::classic::classify(test, std::span<const Class>{known}, opts);
    auto   r_v2      = mqces::v2::classify(     test, std::span<const Class>{known}, opts);
    EXPECT_EQ(r_classic.best_class, r_v2.best_class);
}

// Misclass prob ∈ [0, 1] in v2 too.
TEST(ClassifyV2, MisclassProbInUnitInterval)
{
    constexpr int d = 2;
    std::vector<Class> known{
        make_class("A", 15, d,  0.0, 0.5, 5),
        make_class("B", 15, d,  0.6, 0.5, 6),
    };
    Sample test = make_class("test", 15, d, 0.0, 0.5, 7).specimens;
    auto   res  = mqces::v2::classify(test, std::span<const Class>{known}, default_options(d));
    EXPECT_GE(res.misclassification_prob, 0.0);
    EXPECT_LE(res.misclassification_prob, 1.0);
}

// Paired-difference t-statistic should generally be less noisy than the
// independent-samples form when the two scores are strongly correlated
// (same underlying test sample shared between both replicates). So v2's
// misclass prob should be ≤ classic's for an "obvious" winner.
TEST(ClassifyV2, IsAtLeastAsConfidentAsClassicOnEasyCase)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("near", 20, d,  0.0, 0.5, 1),
        make_class("far",  20, d, 10.0, 0.5, 2),
    };
    // Test clearly matches "near".
    Sample test = make_class("t", 20, d, 0.0, 0.5, 3).specimens;
    auto   opts = default_options(d);
    auto   r_classic = mqces::classic::classify(test, std::span<const Class>{known}, opts);
    auto   r_v2      = mqces::v2::classify(     test, std::span<const Class>{known}, opts);
    EXPECT_EQ(r_v2.best_class, "near");
    // Loose check: v2 should report at most ~ as much misclass probability.
    EXPECT_LE(r_v2.misclassification_prob, r_classic.misclassification_prob + 0.01);
}

TEST(ClassifyV2, SingleClassReportsZeroMisclassProb)
{
    constexpr int d = 2;
    std::vector<Class> known{make_class("only", 15, d, 0.0, 1.0, 8)};
    Sample             test = make_class("test", 15, d, 0.0, 1.0, 9).specimens;
    auto               res  = mqces::v2::classify(
        test, std::span<const Class>{known}, default_options(d));
    EXPECT_EQ(res.best_class, "only");
    EXPECT_EQ(res.misclassification_prob, 0.0);
}
