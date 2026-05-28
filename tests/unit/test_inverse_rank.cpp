#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <stdexcept>

using mqces::inverse_spatial_rank;
using mqces::Sample;
using mqces::spatial_rank;

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr double kRoundTripTol = 1e-7;

}  // namespace

// Round-trip property: if we compute ranks of x against y and then ask
// inverse_spatial_rank to recover points whose ranks (against y) match,
// the recovered points should reproduce the input ranks. We can verify
// directly that spatial_rank-style evaluation of x_tilde against y returns
// the original u vectors.
TEST(InverseRank, RoundTripOnGaussianCloud)
{
    std::mt19937_64                gen(42);
    std::normal_distribution<>     n01(0.0, 1.0);

    constexpr int M = 30;  // size of cloud y
    constexpr int d = 4;
    Sample        y(M, d);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = n01(gen);
        }
    }

    // Pick a handful of "query" points and compute their spatial-rank-style
    // images against y, then invert.
    constexpr int N = 5;
    RowMajorMatrix u(N, d);
    std::vector<Eigen::VectorXd> truth_points;
    truth_points.reserve(N);

    auto signed_rank_against_y = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
        for (int i = 0; i < M; ++i) {
            Eigen::VectorXd diff = x - y.row(i).transpose();
            double          nrm  = diff.norm();
            if (nrm > 1e-15) {
                acc += diff / nrm;
            }
        }
        return (acc / static_cast<double>(M)).eval();
    };

    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd q(d);
        for (int j = 0; j < d; ++j) {
            q(j) = n01(gen);
        }
        truth_points.push_back(q);
        u.row(k) = signed_rank_against_y(q).transpose();
    }

    auto inv = inverse_spatial_rank(u, y, 1e-12, 500);
    ASSERT_EQ(inv.x_tilde.rows(), N);
    ASSERT_EQ(inv.x_tilde.cols(), d);

    // The inverse is well-defined: re-evaluate ranks of x_tilde against y
    // and compare to the input u.
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd recomputed = signed_rank_against_y(inv.x_tilde.row(k).transpose());
        Eigen::VectorXd target     = u.row(k).transpose();
        EXPECT_NEAR((recomputed - target).norm(), 0.0, kRoundTripTol)
            << "row " << k << " round-trip failure";
    }
}

// Mismatched feature dims must throw rather than read past the end.
TEST(InverseRank, ThrowsOnDimensionMismatch)
{
    RowMajorMatrix u(2, 3);
    u.setZero();
    Sample y(5, 4);
    y.setRandom();
    EXPECT_THROW(inverse_spatial_rank(u, y), std::invalid_argument);
}

// Non-convergence within max_iters surfaces as a runtime_error rather than
// returning silently-wrong data. Use max_iters=1 on a non-trivial problem to
// force the failure.
TEST(InverseRank, ThrowsOnNonConvergence)
{
    std::mt19937_64               gen(7);
    std::normal_distribution<>    n01(0.0, 1.0);
    constexpr int M = 20, d = 3, N = 2;

    Sample y(M, d);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = n01(gen);
        }
    }
    RowMajorMatrix u(N, d);
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < d; ++j) {
            u(k, j) = 0.5 * n01(gen);
        }
    }
    // tol absurdly tight, max_iters = 1 → guaranteed failure.
    EXPECT_THROW(inverse_spatial_rank(u, y, 1e-30, 1), std::runtime_error);
}

// Result struct should report at least 1 iteration of work on a converged
// non-trivial problem, and a finite residual. The input `u` must be a valid
// rank vector (norm < 1 by construction), so compute it from real query
// points rather than sampling arbitrary directions.
TEST(InverseRank, ReportsIterationCount)
{
    std::mt19937_64            gen(123);
    std::normal_distribution<> n01(0.0, 1.0);
    constexpr int              M = 20, d = 3, N = 4;

    Sample y(M, d);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = n01(gen);
        }
    }

    auto signed_rank_against_y = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
        for (int i = 0; i < M; ++i) {
            Eigen::VectorXd diff = x - y.row(i).transpose();
            double          nrm  = diff.norm();
            if (nrm > 1e-15) {
                acc += diff / nrm;
            }
        }
        return (acc / static_cast<double>(M)).eval();
    };

    RowMajorMatrix u(N, d);
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd q(d);
        for (int j = 0; j < d; ++j) {
            q(j) = n01(gen);
        }
        u.row(k) = signed_rank_against_y(q).transpose();
    }

    auto inv = inverse_spatial_rank(u, y, 1e-10, 500);
    EXPECT_GE(inv.max_iters_used, 1u);
    EXPECT_TRUE(std::isfinite(inv.max_residual));
    EXPECT_LE(inv.max_residual, 1e-10);
}
