// Anderson-accelerated inverse spatial rank: same solutions as plain
// Weiszfeld, far fewer iterations where Weiszfeld is slow, and no dependence
// on the data's scale.

#include "../support/random_samples.hpp"

#include <mqces/detail/nonlinear_solve.hpp>
#include <mqces/detail/rng.hpp>
#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

using mqces::inverse_spatial_rank;
using mqces::InverseRankSolver;
using mqces::Sample;
using mqces::SolverConfig;
using mqces::spatial_rank;
using mqces::test_support::gaussian_sample;

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const SolverConfig kWeiszfeld{};
const SolverConfig kAnderson{InverseRankSolver::Anderson, 5};

// Signed spatial rank of point x against cloud y (Eq. 2's right-hand side).
Eigen::VectorXd rank_against(const Eigen::VectorXd& x, const Sample& y) {
  Eigen::VectorXd acc = Eigen::VectorXd::Zero(y.cols());
  for (Eigen::Index i = 0; i < y.rows(); ++i) {
    const Eigen::VectorXd diff = x - y.row(i).transpose();
    const double norm = diff.norm();
    if (norm > 1e-15) {
      acc += diff / norm;
    }
  }
  return acc / static_cast<double>(y.rows());
}

struct Problem {
  RowMajorMatrix u;  // target ranks, one per row
  Sample y;          // cloud to invert against
};

// A random split of a tight cluster pooled with a broad one `separation`
// away, as the none-of-the-above permutation test produces for an outlier.
// Rows whose solution lies between the two clusters are where plain
// Weiszfeld needs thousands of iterations.
Problem cluster_gap(double separation, double scale, std::uint64_t split_seed) {
  constexpr int n = 30;
  constexpr int d = 3;
  Sample pooled(2 * n, d);
  pooled.topRows(n) = gaussian_sample(n, d, separation, 5.0, 22) * scale;
  pooled.bottomRows(n) = gaussian_sample(n, d, 0.0, 0.3, 21) * scale;

  auto rng = mqces::detail::make_stream(split_seed, 7);
  std::vector<Eigen::Index> perm(2 * n);
  std::iota(perm.begin(), perm.end(), Eigen::Index{0});
  mqces::detail::shuffle(perm.begin(), perm.size(), rng);

  Sample x(n, d);
  Problem p;
  p.y.resize(n, d);
  for (Eigen::Index j = 0; j < n; ++j) {
    x.row(j) = pooled.row(perm[static_cast<std::size_t>(j)]);
    p.y.row(j) = pooled.row(perm[static_cast<std::size_t>(n + j)]);
  }
  p.u = spatial_rank(x);
  return p;
}

}  // namespace

// The recovered points reproduce the target ranks.
TEST(InverseRankAnderson, RoundTripOnGaussianCloud) {
  Sample y = gaussian_sample(30, 4, 0.0, 1.0, 42);
  Sample queries = gaussian_sample(5, 4, 0.0, 1.0, 43);
  RowMajorMatrix u(queries.rows(), queries.cols());
  for (Eigen::Index k = 0; k < queries.rows(); ++k) {
    u.row(k) = rank_against(queries.row(k).transpose(), y).transpose();
  }

  const auto inv = inverse_spatial_rank(u, y, 1e-12, 500, kAnderson);
  for (Eigen::Index k = 0; k < u.rows(); ++k) {
    const Eigen::VectorXd recomputed = rank_against(inv.x_tilde.row(k).transpose(), y);
    EXPECT_NEAR((recomputed - u.row(k).transpose()).norm(), 0.0, 1e-7) << "row " << k;
  }
}

// Both solvers converge to the same points on a typical problem.
TEST(InverseRankAnderson, MatchesWeiszfeldSolution) {
  Sample x = gaussian_sample(50, 9, 0.0, 1.0, 100);
  Sample y = gaussian_sample(50, 9, 0.3, 1.0, 200);
  const RowMajorMatrix u = spatial_rank(x);

  const auto weiszfeld = inverse_spatial_rank(u, y, 1e-10, 100000, kWeiszfeld);
  const auto anderson = inverse_spatial_rank(u, y, 1e-10, 100000, kAnderson);
  EXPECT_LT((anderson.x_tilde - weiszfeld.x_tilde).rowwise().norm().maxCoeff(), 1e-6);
  EXPECT_LT(anderson.max_iters_used, weiszfeld.max_iters_used);
}

// Between well-separated clusters plain Weiszfeld crawls; Anderson reaches
// the same tolerance in a small, bounded number of iterations and agrees with
// Weiszfeld's solution to within that slow convergence's own error.
TEST(InverseRankAnderson, ConvergesQuicklyBetweenClusters) {
  std::size_t worst_weiszfeld = 0;
  std::size_t worst_anderson = 0;
  for (std::uint64_t split = 0; split < 10; ++split) {
    const Problem p = cluster_gap(500.0, 1.0, split);
    const auto weiszfeld = inverse_spatial_rank(p.u, p.y, 1e-5, 200000, kWeiszfeld);
    const auto anderson = inverse_spatial_rank(p.u, p.y, 1e-5, 200000, kAnderson);
    worst_weiszfeld = std::max(worst_weiszfeld, weiszfeld.max_iters_used);
    worst_anderson = std::max(worst_anderson, anderson.max_iters_used);
    EXPECT_LT((anderson.x_tilde - weiszfeld.x_tilde).rowwise().norm().maxCoeff(), 1e-2)
      << "split " << split;
  }
  EXPECT_GT(worst_weiszfeld, 1000u) << "problem no longer exercises slow Weiszfeld convergence";
  EXPECT_LT(worst_anderson, 150u);
}

// Scaling the data by c scales Anderson's iterates by c, so a tolerance
// relative to the scale takes the same work and gives the same solution.
// Rounding can move the stopping point by one iteration, which changes the
// result by about one tolerance-sized step, hence the tight tolerance here.
TEST(InverseRankAnderson, IsScaleInvariant) {
  constexpr double kRelTol = 1e-9;
  const Problem base = cluster_gap(500.0, 1.0, 3);
  const auto reference = inverse_spatial_rank(base.u, base.y, kRelTol, 200000, kAnderson);
  for (const double scale : {1e-3, 1e3}) {
    const Problem scaled = cluster_gap(500.0, scale, 3);
    const auto result =
      inverse_spatial_rank(scaled.u, scaled.y, kRelTol * scale, 200000, kAnderson);
    const auto iteration_gap = static_cast<long long>(result.max_iters_used) -
                               static_cast<long long>(reference.max_iters_used);
    EXPECT_LE(std::llabs(iteration_gap), 1) << "scale " << scale;
    EXPECT_LT((result.x_tilde / scale - reference.x_tilde).rowwise().norm().maxCoeff(), 1e-7)
      << "scale " << scale;
  }
}

// When the solution coincides with a data point the objective has a kink
// there; Anderson must still converge onto that point at any scale.
TEST(InverseRankAnderson, ConvergesToSolutionOnDataPoint) {
  constexpr Eigen::Index k = 7;
  for (const double scale : {1e-3, 1.0, 1e3}) {
    const Sample y = gaussian_sample(30, 3, 0.0, 1.0, 11) * scale;
    // The rank of y_k against the other points, nudged by less than one unit
    // vector's worth: the subgradient condition still holds at y_k, so y_k is
    // the exact solution.
    Eigen::VectorXd u = Eigen::VectorXd::Zero(3);
    for (Eigen::Index j = 0; j < y.rows(); ++j) {
      if (j != k) {
        const Eigen::VectorXd e = y.row(k).transpose() - y.row(j).transpose();
        u += e / e.norm();
      }
    }
    u += Eigen::Vector3d(0.3, -0.4, 0.0);
    u /= static_cast<double>(y.rows());

    const auto result = inverse_spatial_rank(u.transpose(), y, 1e-9 * scale, 20000, kAnderson);
    EXPECT_LT((result.x_tilde.row(0) - y.row(k)).norm() / scale, 1e-6) << "scale " << scale;
  }
}

// Rows of a matrix, for small hand-written samples.
namespace {

Sample rows(std::initializer_list<std::initializer_list<double>> values) {
  Sample s(static_cast<Eigen::Index>(values.size()),
           static_cast<Eigen::Index>(values.begin()->size()));
  Eigen::Index i = 0;
  for (const auto& row : values) {
    Eigen::Index j = 0;
    for (const double v : row) {
      s(i, j++) = v;
    }
    ++i;
  }
  return s;
}

}  // namespace

// Collinearity is judged against what floating-point distances can resolve:
// a transverse offset below sqrt(epsilon) of the cloud's length is invisible.
TEST(InverseRankAnderson, DetectsCollinearClouds) {
  using mqces::detail::cloud_is_collinear;
  EXPECT_TRUE(cloud_is_collinear(rows({{1.0, 2.0}})));
  EXPECT_TRUE(cloud_is_collinear(rows({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}})));
  EXPECT_TRUE(cloud_is_collinear(rows({{0.0}, {1.0}, {5.0}, {2.0}})));
  EXPECT_TRUE(cloud_is_collinear(rows({{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}, {-2.0, -4.0, -6.0}})));
  EXPECT_TRUE(cloud_is_collinear(rows({{0.0, 0.0}, {1.0, 1e-10}, {2.0, 0.0}, {3.0, -1e-10}})));
  EXPECT_FALSE(cloud_is_collinear(rows({{0.0, 0.0}, {1.0, 1e-6}, {2.0, 0.0}, {3.0, -1e-6}})));
  EXPECT_FALSE(cloud_is_collinear(rows({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}})));
}

// A one-feature cloud's inverse rank can be a whole interval, so the two
// solvers could select different points and change the winning class. Both
// must therefore take the Weiszfeld path and agree exactly.
TEST(InverseRankAnderson, CollinearCloudUsesWeiszfeld) {
  const Sample test = rows({{0.0}, {1.0}, {2.0}, {3.0}});
  const Sample cls = rows({{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}});
  const RowMajorMatrix u = spatial_rank(test);
  const auto weiszfeld = inverse_spatial_rank(u, cls, 1e-5, 50000, kWeiszfeld);
  const auto anderson = inverse_spatial_rank(u, cls, 1e-5, 50000, kAnderson);
  EXPECT_EQ(anderson.x_tilde, weiszfeld.x_tilde);

  // The same holds for points on a slanted line in three dimensions.
  Sample line(6, 3);
  for (Eigen::Index i = 0; i < line.rows(); ++i) {
    const auto t = static_cast<double>(i * i) - 4.0;
    line.row(i) << 1.0 + t, 2.0 - 2.0 * t, 0.5 * t;
  }
  const RowMajorMatrix u_line = spatial_rank(line.topRows(4));
  EXPECT_EQ(inverse_spatial_rank(u_line, line, 1e-5, 50000, kAnderson).x_tilde,
            inverse_spatial_rank(u_line, line, 1e-5, 50000, kWeiszfeld).x_tilde);
}

// A cloud with zero spread would put Anderson's start on every point at once,
// where the Weiszfeld map is undefined. Called directly, the Anderson routine
// hands such a cloud to plain Weiszfeld, which converges onto the point.
TEST(InverseRankAnderson, ZeroSpreadCloudConverges) {
  const Sample y = rows({{3.0, -1.0}, {3.0, -1.0}});
  const Eigen::Vector2d u(0.2, -0.1);
  const auto result = mqces::detail::solve_inverse_rank_row_anderson(u, y, 1e-9, 1000, 5);
  ASSERT_TRUE(result.converged);
  EXPECT_LT((result.x - y.row(0).transpose()).norm(), 1e-6);
}
