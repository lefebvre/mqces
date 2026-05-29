#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <stdexcept>

using mqces::inverse_spatial_rank;
using mqces::Sample;
using mqces::SolverConfig;
using mqces::SolverKind;

namespace {

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Sample gaussian(int n, int d, std::uint64_t seed)
{
    std::mt19937_64            gen(seed);
    std::normal_distribution<> n01(0.0, 1.0);
    Sample                     s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = n01(gen);
        }
    }
    return s;
}

// Rank-vector-against-cloud helper for building valid u inputs.
Eigen::VectorXd rank_against(const Eigen::VectorXd& x, const Sample& y, double eps = 1e-15)
{
    const auto      M = y.rows();
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(x.size());
    for (Eigen::Index i = 0; i < M; ++i) {
        Eigen::VectorXd diff = x - y.row(i).transpose();
        double          nrm  = diff.norm();
        if (nrm > eps) {
            acc += diff / nrm;
        }
    }
    return acc / static_cast<double>(M);
}

}  // namespace

// On a well-conditioned random input, VZ should reach the same answer as
// plain Weiszfeld (to within solver tolerance). VZ should never make
// things *worse* on benign data.
TEST(InverseVZ, MatchesWeiszfeldOnBenignInput)
{
    std::mt19937_64               gen(0xC0DE);
    std::normal_distribution<>    n01(0.0, 1.0);
    constexpr int                 M = 30, d = 4, N = 4;
    Sample                        y = gaussian(M, d, 1);

    RowMajor u(N, d);
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd q(d);
        for (int j = 0; j < d; ++j) {
            q(j) = n01(gen);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_w;
    sf_w.kind      = SolverKind::Weiszfeld;
    sf_w.tol       = 1e-10;
    sf_w.max_iters = 500;

    SolverConfig sf_vz = sf_w;
    sf_vz.kind         = SolverKind::VardiZhang;

    auto r_w  = inverse_spatial_rank(u, y, sf_w);
    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    // Both converge to the same x_tilde to within their per-row tol.
    EXPECT_LE((r_w.x_tilde - r_vz.x_tilde).array().abs().maxCoeff(), 1e-8);
}

// Vertex configuration recovery: construct an input where the inverse-rank
// solution sits exactly at one of the y_i points. VZ should converge to
// the configured tight tol and recover the vertex.
//
// We do NOT also assert that plain Weiszfeld fails here: whether Weiszfeld
// plateaus on a specific seed depends on the iterate's path through the
// cloud, which depends on the initial guess geometry. The plateau is
// observable in aggregate (across many seeds and on the realistic
// classifier inputs in test_solver_score_parity / classify_v3) but isn't
// reliable as a per-case test invariant.
TEST(InverseVZ, ConvergesTightOnVertexCase)
{
    constexpr int d = 3, M = 10;
    Sample        y = gaussian(M, d, 0xBADBEEF);

    // Target u is the rank of y.row(3) against the cloud — solution is
    // exactly y.row(3), the classical vertex case.
    RowMajor u(1, d);
    u.row(0) = rank_against(y.row(3).transpose(), y).transpose();

    SolverConfig sf_vz;
    sf_vz.kind          = SolverKind::VardiZhang;
    sf_vz.tol           = 1e-10;
    sf_vz.max_iters     = 200;
    sf_vz.vz_vertex_eps = 1e-4;

    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    EXPECT_LE(r_vz.max_residual, 1e-10);
    const auto& recovered = r_vz.x_tilde.row(0).transpose();
    EXPECT_LE((recovered - y.row(3).transpose()).norm(), 1e-3);
}

// Determinism: identical inputs produce identical outputs across runs.
TEST(InverseVZ, IsDeterministic)
{
    Sample y = gaussian(20, 4, 7);
    RowMajor u(3, 4);
    for (int k = 0; k < 3; ++k) {
        Eigen::VectorXd q(4);
        std::mt19937_64 g(42 + k);
        std::normal_distribution<> n01(0.0, 1.0);
        for (int j = 0; j < 4; ++j) {
            q(j) = n01(g);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_vz;
    sf_vz.kind          = SolverKind::VardiZhang;
    sf_vz.tol           = 1e-10;
    sf_vz.max_iters     = 500;
    sf_vz.vz_vertex_eps = 1e-6;

    auto r1 = inverse_spatial_rank(u, y, sf_vz);
    auto r2 = inverse_spatial_rank(u, y, sf_vz);
    EXPECT_LE((r1.x_tilde - r2.x_tilde).array().abs().maxCoeff(), 0.0);
}

// Round-trip: VZ recovers x_tilde such that rank(x_tilde, y) == u given.
TEST(InverseVZ, RoundTripMatchesInputRanks)
{
    std::mt19937_64 gen(0xABCD);
    std::normal_distribution<> n01(0.0, 1.0);
    constexpr int   d = 3, M = 25, N = 5;
    Sample          y = gaussian(M, d, 11);

    RowMajor u(N, d);
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd q(d);
        for (int j = 0; j < d; ++j) {
            q(j) = n01(gen);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_vz;
    sf_vz.kind      = SolverKind::VardiZhang;
    sf_vz.tol       = 1e-10;
    sf_vz.max_iters = 500;

    auto r = inverse_spatial_rank(u, y, sf_vz);
    for (Eigen::Index k = 0; k < N; ++k) {
        Eigen::VectorXd recomputed = rank_against(r.x_tilde.row(k).transpose(), y);
        Eigen::VectorXd target     = u.row(k).transpose();
        EXPECT_LE((recomputed - target).norm(), 1e-7);
    }
}
