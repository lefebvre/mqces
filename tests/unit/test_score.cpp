#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include <mqces/detail/rng.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>
#include <stdexcept>

using mqces::FeatureWeights;
using mqces::Sample;
using mqces::similarity_score;

namespace {

Sample gaussian_sample(int n, int d, std::uint64_t seed)
{
    std::mt19937_64            gen(seed);
    const auto n01 = [](std::mt19937_64& g) { return mqces::detail::standard_normal(g); };
    Sample                     s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = n01(gen);
        }
    }
    return s;
}

}  // namespace

// Mismatched feature counts must throw before we touch the data.
TEST(SimilarityScore, ThrowsOnFeatureDimMismatch)
{
    Sample         x = gaussian_sample(10, 3, 1);
    Sample         y = gaussian_sample(10, 4, 2);
    FeatureWeights w = FeatureWeights::Ones(3);
    EXPECT_THROW(similarity_score(x, y, w), std::invalid_argument);
}

// Weight length must match feature count.
TEST(SimilarityScore, ThrowsOnWeightLengthMismatch)
{
    Sample         x = gaussian_sample(10, 3, 3);
    Sample         y = gaussian_sample(10, 3, 4);
    FeatureWeights w = FeatureWeights::Ones(5);  // wrong size
    EXPECT_THROW(similarity_score(x, y, w), std::invalid_argument);
}

// Comparing a sample with itself yields a near-zero score: the inverse-rank
// recovery is exact (up to solver tolerance), so the per-row squared
// distances all vanish.
TEST(SimilarityScore, IdenticalSamplesYieldNearZero)
{
    Sample         x = gaussian_sample(15, 3, 7);
    FeatureWeights w = FeatureWeights::Ones(3);
    double         s = similarity_score(x, x, w);
    EXPECT_GE(s, 0.0);
    EXPECT_LT(s, 1e-10);
}

// Score is symmetric: S(x, y) == S(y, x) by construction (Eq. 3 is the sum
// of two symmetric reconstruction terms).
TEST(SimilarityScore, IsSymmetric)
{
    Sample         x  = gaussian_sample(20, 3, 11);
    Sample         y  = gaussian_sample(20, 3, 13);
    FeatureWeights w  = FeatureWeights::Ones(3);
    double         s1 = similarity_score(x, y, w);
    double         s2 = similarity_score(y, x, w);
    EXPECT_NEAR(s1, s2, 1e-9);
}

// Two distinct samples drawn from non-overlapping Gaussians should produce
// a strictly positive score.
TEST(SimilarityScore, DistinctSamplesYieldPositiveScore)
{
    Sample x = gaussian_sample(20, 3, 17);
    Sample y = gaussian_sample(20, 3, 19);
    y.array() += 5.0;  // shift y far from x
    FeatureWeights w = FeatureWeights::Ones(3);
    double         s = similarity_score(x, y, w);
    EXPECT_GT(s, 1e-3);
}

// Doubling all weights doubles the score (linearity in the diagonal of W).
TEST(SimilarityScore, ScalesLinearlyWithWeights)
{
    Sample         x  = gaussian_sample(15, 3, 21);
    Sample         y  = gaussian_sample(15, 3, 22);
    FeatureWeights w  = FeatureWeights::Ones(3);
    double         s1 = similarity_score(x, y, w);
    double         s2 = similarity_score(x, y, (2.0 * w).eval());
    EXPECT_NEAR(s2, 2.0 * s1, 1e-9);
}
