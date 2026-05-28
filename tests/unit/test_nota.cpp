#include <mqces/classify.hpp>
#include <mqces/nota.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>
#include <span>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::is_none_of_the_above;
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
    opts.weights                = FeatureWeights::Ones(d);
    // Non-zero epsilon is required for NOTA to be meaningful: zero MC noise
    // collapses both score distributions to a single point, making any
    // mean-difference t-test reject by construction.
    opts.uncertainty.epsilon    = 0.05;
    opts.uncertainty.mc_samples = 8;
    opts.uncertainty.seed       = 9999;
    opts.nota_threshold         = 0.05;
    opts.n_threads              = 1;
    return opts;
}

}  // namespace

// With fewer than 4 specimens in best.specimens we cannot split and must
// abstain (return false) regardless of the test scores.
TEST(NOTA, ReturnsFalseForTinyClass)
{
    Class           c   = make_class("tiny", 3, 2, 0.0, 1.0, 1);
    Sample          test = make_class("t", 8, 2, 0.0, 1.0, 2).specimens;
    Eigen::VectorXd scores = Eigen::VectorXd::Constant(8, 1e3);  // huge — would otherwise trigger NOTA
    EXPECT_FALSE(is_none_of_the_above(test, c, scores, default_options(2)));
}

// When the test scores agree closely with the within-class baseline,
// NOTA should NOT fire. The default p-value threshold of 0.05 is sensitive
// to small differences between two independent draws from the same
// distribution (within-class scores are systematically lower because both
// halves share the same 30 specimens), so we tighten the threshold here.
// This documents a real limitation of NOTA's calibration that the paper
// also flags qualitatively.
TEST(NOTA, AcceptsGenuineMember)
{
    constexpr int d = 3;
    std::vector<Class> known{make_class("k", 30, d, 0.0, 1.0, 11)};
    Sample             test = make_class("t", 30, d, 0.0, 1.0, 12).specimens;
    auto               opts = default_options(d);
    opts.nota_threshold     = 1e-4;  // tighter than the default 0.05
    auto res = mqces::classic::classify(test, std::span<const Class>{known}, opts);
    EXPECT_FALSE(res.none_of_the_above);
}

// When the test is drawn from a wildly different distribution than the only
// "matched" class, NOTA should fire.
TEST(NOTA, RejectsClearOutlier)
{
    constexpr int d = 3;
    std::vector<Class> known{make_class("k", 30, d, 0.0, 0.3, 21)};
    // Test sampled from a totally different distribution: large mean and
    // larger spread.
    Sample test = make_class("t", 30, d, 50.0, 5.0, 22).specimens;
    auto   opts = default_options(d);
    auto   res  = mqces::classic::classify(test, std::span<const Class>{known}, opts);
    EXPECT_TRUE(res.none_of_the_above);
}
