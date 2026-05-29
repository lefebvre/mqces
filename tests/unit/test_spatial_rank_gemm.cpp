#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>

using mqces::Sample;
using mqces::spatial_rank;

namespace {

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Plain O(N²) reference implementation. Distinct from the GEMM path so a
// bug in either is unlikely to disguise itself; we use this to validate
// the tiled-GEMM accumulation order.
RowMajor naive_spatial_rank(const Sample& x, double eps = 1e-12)
{
    const Eigen::Index N = x.rows();
    const Eigen::Index d = x.cols();
    RowMajor           u = RowMajor::Zero(N, d);
    if (N <= 1) {
        return u;
    }
    for (Eigen::Index j = 0; j < N; ++j) {
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
        for (Eigen::Index i = 0; i < N; ++i) {
            if (i == j) {
                continue;
            }
            Eigen::VectorXd diff = x.row(j).transpose() - x.row(i).transpose();
            double          norm = diff.norm();
            if (norm < eps) {
                continue;
            }
            acc += diff / norm;
        }
        u.row(j) = acc.transpose() / static_cast<double>(N);
    }
    return u;
}

Sample gaussian_sample(int n, int d, double sigma, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> dist(0.0, sigma);
    Sample                     s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = dist(rng);
        }
    }
    return s;
}

double max_abs_diff(const RowMajor& a, const RowMajor& b)
{
    return (a - b).array().abs().maxCoeff();
}

}  // namespace

// Small N (below the tile size): exercises only the single diagonal tile.
TEST(SpatialRankGemm, AgreesWithNaiveAtSmallN)
{
    const auto x = gaussian_sample(50, 9, 1.0, 0xABCDEF);
    const auto u_gemm = spatial_rank(x);
    const auto u_naive = naive_spatial_rank(x);
    // Reassociation tolerance: ~N · ulp(1) ≈ 1e-13 at N=50.
    EXPECT_LE(max_abs_diff(u_gemm, u_naive), 1e-13);
}

// N straddles the kTileRows boundary so we exercise diagonal + off-diagonal
// tiles together.
TEST(SpatialRankGemm, AgreesWithNaiveAcrossTileBoundary)
{
    const auto x = gaussian_sample(300, 9, 1.0, 0x9CA770);
    const auto u_gemm = spatial_rank(x);
    const auto u_naive = naive_spatial_rank(x);
    EXPECT_LE(max_abs_diff(u_gemm, u_naive), 1e-12);
}

// Multiple tiles, larger feature dim — closer to the production target.
TEST(SpatialRankGemm, AgreesWithNaiveAtLargerNAndHigherD)
{
    const auto x = gaussian_sample(600, 25, 1.0, 0x5EED5);
    const auto u_gemm = spatial_rank(x);
    const auto u_naive = naive_spatial_rank(x);
    EXPECT_LE(max_abs_diff(u_gemm, u_naive), 1e-12);
}

// Coincident-point handling must match the naive code, which skips pairs
// with distance < eps. Construct a sample with deliberate duplicates.
TEST(SpatialRankGemm, CoincidentPointsHandledIdentically)
{
    Sample x(8, 3);
    x << 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0,  // duplicate of row 0
         1.0, 0.0, 0.0,
         1.0, 0.0, 0.0,  // duplicate of row 2
         0.0, 1.0, 0.0,
         0.0, 1.0, 0.0,  // duplicate of row 4
         2.0, 2.0, 2.0,
         3.0, 3.0, 3.0;
    const auto u_gemm = spatial_rank(x);
    const auto u_naive = naive_spatial_rank(x);
    EXPECT_TRUE(u_gemm.allFinite());
    EXPECT_LE(max_abs_diff(u_gemm, u_naive), 1e-13);
}

// Near-coincident pairs are where the GEMM fast formula (s_i + s_j − 2·G)
// can cancel catastrophically; the cancellation guard in quantile.cpp
// should fall back to a direct subtraction and still match the naive
// answer to ~1e-12.
TEST(SpatialRankGemm, NearCoincidentPairsAreNumericallyStable)
{
    Sample x(6, 4);
    x.setRandom();
    // Make rows 2 and 3 differ only in the last bit of the last column —
    // they're at distance ~1e-15 which falls below eps=1e-12 and should
    // be skipped.
    x.row(3) = x.row(2);
    x(3, 3)  = std::nextafter(x(2, 3), x(2, 3) + 1.0);
    const auto u_gemm = spatial_rank(x);
    EXPECT_TRUE(u_gemm.allFinite());
    const auto u_naive = naive_spatial_rank(x);
    EXPECT_LE(max_abs_diff(u_gemm, u_naive), 1e-12);
}

// Edge cases that the existing test_spatial_rank suite already covers,
// re-checked here to lock the GEMM path's behavior on degenerate inputs.
TEST(SpatialRankGemm, EmptyAndSingleton)
{
    Sample empty(0, 3);
    auto   u_empty = spatial_rank(empty);
    EXPECT_EQ(u_empty.rows(), 0);
    EXPECT_EQ(u_empty.cols(), 3);

    Sample single(1, 3);
    single << 1.0, 2.0, 3.0;
    auto u_single = spatial_rank(single);
    ASSERT_EQ(u_single.rows(), 1);
    EXPECT_NEAR(u_single.row(0).norm(), 0.0, 1e-15);
}
