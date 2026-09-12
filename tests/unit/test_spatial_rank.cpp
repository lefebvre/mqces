#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using mqces::Sample;
using mqces::spatial_rank;

namespace {

constexpr double kTightTol = 1e-12;
constexpr double kLooseTol = 1e-9;

}  // namespace

// N == 0 and N == 1: spatial rank is defined as zero (no other points to
// average over). The function should not divide by zero or read past the
// end of the input.
TEST(SpatialRank, EmptyAndSingleton) {
  Sample empty(0, 3);
  auto u_empty = spatial_rank(empty);
  EXPECT_EQ(u_empty.rows(), 0);
  EXPECT_EQ(u_empty.cols(), 3);

  Sample single(1, 3);
  single << 1.0, 2.0, 3.0;
  auto u_single = spatial_rank(single);
  ASSERT_EQ(u_single.rows(), 1);
  EXPECT_NEAR(u_single.row(0).norm(), 0.0, kTightTol);
}

// Two distinct points: each rank vector is the unit vector from the other
// point, scaled by 1/N = 1/2. The two rank vectors are antiparallel.
TEST(SpatialRank, TwoPoints) {
  Sample x(2, 2);
  x << 0.0, 0.0, 3.0, 4.0;  // ||diff|| = 5
  auto u = spatial_rank(x);
  ASSERT_EQ(u.rows(), 2);
  // row 0: (x_0 - x_1)/5 / 2 = (-3,-4)/5 / 2 = (-0.3, -0.4)
  EXPECT_NEAR(u(0, 0), -0.3, kTightTol);
  EXPECT_NEAR(u(0, 1), -0.4, kTightTol);
  EXPECT_NEAR(u(1, 0), 0.3, kTightTol);
  EXPECT_NEAR(u(1, 1), 0.4, kTightTol);
  EXPECT_NEAR((u.row(0) + u.row(1)).norm(), 0.0, kTightTol);
}

// Symmetric cloud around the origin: the centroid row's rank should be
// (close to) zero by symmetry. Use a regular hexagon plus its centroid.
TEST(SpatialRank, SymmetricCloudHasZeroCentroidRank) {
  constexpr int N = 7;  // 6 hex vertices + centroid
  Sample x(N, 2);
  x.row(0).setZero();
  for (int k = 0; k < 6; ++k) {
    double theta = 2.0 * std::numbers::pi * static_cast<double>(k) / 6.0;
    x(k + 1, 0) = std::cos(theta);
    x(k + 1, 1) = std::sin(theta);
  }
  auto u = spatial_rank(x);
  EXPECT_NEAR(u.row(0).norm(), 0.0, kLooseTol) << "centroid rank should vanish by symmetry";
}

// Collinear points along the +x axis: every rank vector lies on that line,
// with sign determined by the point's position relative to the others.
TEST(SpatialRank, CollinearPointsLieOnAxis) {
  constexpr int N = 5;
  Sample x(N, 3);
  for (int i = 0; i < N; ++i) {
    x(i, 0) = static_cast<double>(i);
    x(i, 1) = 0.0;
    x(i, 2) = 0.0;
  }
  auto u = spatial_rank(x);
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(u(i, 1), 0.0, kTightTol) << "row " << i << " y";
    EXPECT_NEAR(u(i, 2), 0.0, kTightTol) << "row " << i << " z";
  }
  // Leftmost point ranks to (−1, 0, 0) × 4/N: all 4 others are to its right.
  EXPECT_NEAR(u(0, 0), -static_cast<double>(N - 1) / static_cast<double>(N), kTightTol);
  EXPECT_NEAR(u(N - 1, 0), static_cast<double>(N - 1) / static_cast<double>(N), kTightTol);
  // Middle point sees a balanced left/right split → zero rank.
  EXPECT_NEAR(u(N / 2, 0), 0.0, kTightTol);
}

// Coincident points (identical rows) shouldn't blow up: contributions with
// d_ij < eps are skipped.
TEST(SpatialRank, CoincidentPointsAreSkipped) {
  Sample x(3, 2);
  x << 1.0, 1.0, 1.0, 1.0,  // coincident with row 0
    5.0, 5.0;
  auto u = spatial_rank(x);
  EXPECT_TRUE(u.allFinite());
}

// Permuting the rows of the input permutes the rows of the output identically
// — the operator is row-equivariant.
TEST(SpatialRank, RowPermutationEquivariance) {
  Sample x(4, 2);
  x << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
  auto u_orig = spatial_rank(x);

  Sample x_perm(4, 2);
  x_perm.row(0) = x.row(2);
  x_perm.row(1) = x.row(0);
  x_perm.row(2) = x.row(3);
  x_perm.row(3) = x.row(1);
  auto u_perm = spatial_rank(x_perm);

  EXPECT_NEAR((u_perm.row(0) - u_orig.row(2)).norm(), 0.0, kTightTol);
  EXPECT_NEAR((u_perm.row(1) - u_orig.row(0)).norm(), 0.0, kTightTol);
  EXPECT_NEAR((u_perm.row(2) - u_orig.row(3)).norm(), 0.0, kTightTol);
  EXPECT_NEAR((u_perm.row(3) - u_orig.row(1)).norm(), 0.0, kTightTol);
}
