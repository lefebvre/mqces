#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>

using mqces::Sample;
using mqces::SamplingConfig;
using mqces::spatial_rank;

namespace {

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Sample gaussian_sample(int n, int d, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> dist(0.0, 1.0);
    Sample                     s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = dist(rng);
        }
    }
    return s;
}

double frobenius(const RowMajor& a, const RowMajor& b)
{
    return (a - b).norm();
}

}  // namespace

// reference_size == 0 should dispatch to the exact path and return
// bit-identical output (modulo float reassociation) to the no-sampling
// overload.
TEST(SpatialRankApprox, ZeroSizeIsExactPath)
{
    const auto x = gaussian_sample(80, 9, 0x123);
    const auto u_exact = spatial_rank(x);
    SamplingConfig cfg;
    cfg.reference_size = 0;
    const auto u_via_cfg = spatial_rank(x, cfg);
    EXPECT_LE((u_exact - u_via_cfg).array().abs().maxCoeff(), 1e-15);
}

// reference_size >= N also dispatches to the exact path (no real
// subsampling possible).
TEST(SpatialRankApprox, OversizeDispatchesToExact)
{
    const auto x = gaussian_sample(40, 9, 0x456);
    SamplingConfig cfg;
    cfg.reference_size = 1000;  // >> N
    const auto u_via_cfg = spatial_rank(x, cfg);
    const auto u_exact   = spatial_rank(x);
    EXPECT_LE((u_exact - u_via_cfg).array().abs().maxCoeff(), 1e-15);
}

// Determinism: same seed reproduces the same output regardless of when /
// how many times the function is called.
TEST(SpatialRankApprox, IsDeterministicForFixedSeed)
{
    const auto x = gaussian_sample(200, 9, 0xABC);
    SamplingConfig cfg;
    cfg.reference_size = 30;
    cfg.seed           = 42;
    const auto u1 = spatial_rank(x, cfg);
    const auto u2 = spatial_rank(x, cfg);
    EXPECT_LE((u1 - u2).array().abs().maxCoeff(), 0.0);
}

// Different seeds produce different output.
TEST(SpatialRankApprox, DifferentSeedsDiffer)
{
    const auto x = gaussian_sample(200, 9, 0xCAFE);
    SamplingConfig cfg_a;
    cfg_a.reference_size = 30;
    cfg_a.seed           = 1;
    SamplingConfig cfg_b = cfg_a;
    cfg_b.seed           = 2;
    const auto u_a = spatial_rank(x, cfg_a);
    const auto u_b = spatial_rank(x, cfg_b);
    EXPECT_GT((u_a - u_b).norm(), 1e-6);
}

// Bias / variance check: the approximate estimator's mean (over many seeds)
// should approach the exact spatial rank, and the per-coordinate error
// should shrink as O(1/√R).
TEST(SpatialRankApprox, MeanOverSeedsApproachesExact)
{
    const auto x       = gaussian_sample(150, 6, 0xBEEF);
    const auto u_exact = spatial_rank(x);

    SamplingConfig cfg;
    cfg.reference_size = 60;  // 40% sample

    constexpr int trials = 64;
    RowMajor      mean_u = RowMajor::Zero(u_exact.rows(), u_exact.cols());
    for (int t = 0; t < trials; ++t) {
        cfg.seed = static_cast<std::uint64_t>(0xA17EF00DULL + t);
        mean_u  += spatial_rank(x, cfg);
    }
    mean_u /= static_cast<double>(trials);

    const double err = (mean_u - u_exact).array().abs().maxCoeff();
    EXPECT_LT(err, 0.05) << "mean over seeds drifted too far from exact";
}

// Variance comparison: larger R should give a tighter estimate. We expect
// the per-coordinate error at R = 4·R0 to be roughly half that at R0
// (variance ~ 1/R → std ~ 1/√R), with plenty of slack for a single trial.
TEST(SpatialRankApprox, ErrorShrinksWithLargerR)
{
    const auto x       = gaussian_sample(400, 9, 0x5EED);
    const auto u_exact = spatial_rank(x);

    SamplingConfig cfg_small;
    cfg_small.reference_size = 25;
    cfg_small.seed           = 1;
    SamplingConfig cfg_large = cfg_small;
    cfg_large.reference_size = 100;  // 4× R

    const double err_small = frobenius(spatial_rank(x, cfg_small), u_exact);
    const double err_large = frobenius(spatial_rank(x, cfg_large), u_exact);

    EXPECT_LT(err_large, err_small)
        << "expected the 100-point estimator to be tighter than the 25-point one";
}

// Output of the approximate path is finite even with coincident query/ref
// rows (the i == j skipping kicks in).
TEST(SpatialRankApprox, FiniteWithDuplicateRows)
{
    Sample x(10, 3);
    x.setRandom();
    x.row(2) = x.row(7);  // exact duplicate
    SamplingConfig cfg;
    cfg.reference_size = 4;
    cfg.seed           = 0xC0DE;
    const auto u = spatial_rank(x, cfg);
    EXPECT_TRUE(u.allFinite());
}
