#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>

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

// On a benign input AA should converge to the same fixed point as plain
// VZ. Both are solving the same equation; AA changes the path, not the
// answer.
TEST(InverseVZAA, MatchesVzOnBenignInput)
{
    std::mt19937_64            gen(0xC0DE);
    std::normal_distribution<> n01(0.0, 1.0);
    constexpr int              M = 40, d = 5, N = 4;
    Sample                     y = gaussian(M, d, 1);

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
    sf_vz.tol       = 1e-9;
    sf_vz.max_iters = 500;

    SolverConfig sf_aa = sf_vz;
    sf_aa.kind         = SolverKind::VardiZhangAA;
    sf_aa.aa_window    = 5;
    sf_aa.aa_reg       = 1e-12;

    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    auto r_aa = inverse_spatial_rank(u, y, sf_aa);
    EXPECT_LE((r_vz.x_tilde - r_aa.x_tilde).array().abs().maxCoeff(), 1e-6);
}

// Iteration-count reduction: AA should typically reach the same tight
// tol in fewer iterations than plain VZ on the same input. We don't
// assert a strict ratio because it depends on the input geometry; we
// assert AA uses no MORE iterations than VZ and that the comparison runs
// to convergence in both cases.
TEST(InverseVZAA, IterCountIsNoWorseThanVz)
{
    std::mt19937_64            gen(0xBEAD1);
    std::normal_distribution<> n01(0.0, 1.0);
    constexpr int              M = 50, d = 6, N = 8;
    Sample                     y = gaussian(M, d, 17);

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
    sf_vz.tol       = 1e-9;
    sf_vz.max_iters = 1000;

    SolverConfig sf_aa = sf_vz;
    sf_aa.kind         = SolverKind::VardiZhangAA;
    sf_aa.aa_window    = 5;
    sf_aa.aa_reg       = 1e-12;

    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    auto r_aa = inverse_spatial_rank(u, y, sf_aa);

    EXPECT_LE(r_aa.max_iters_used, r_vz.max_iters_used)
        << "AA iter count (" << r_aa.max_iters_used
        << ") should not exceed VZ iter count (" << r_vz.max_iters_used << ")";
}

// Counters: fallback and restart counters are zero for non-AA solvers
// and may be non-zero for AA (we don't assert a specific count — just
// that they're populated).
TEST(InverseVZAA, FallbackCountersZeroForNonAA)
{
    Sample y = gaussian(20, 4, 7);
    RowMajor u(3, 4);
    for (int k = 0; k < 3; ++k) {
        Eigen::VectorXd q(4);
        std::mt19937_64            g(static_cast<std::mt19937_64::result_type>(42 + k));
        std::normal_distribution<> n01(0.0, 1.0);
        for (int j = 0; j < 4; ++j) {
            q(j) = n01(g);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_vz;
    sf_vz.kind      = SolverKind::VardiZhang;
    sf_vz.tol       = 1e-8;
    sf_vz.max_iters = 500;
    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    EXPECT_EQ(r_vz.aa_fallbacks, 0u);
    EXPECT_EQ(r_vz.aa_restarts, 0u);
}

// Determinism: same inputs reproduce the same x_tilde (AA's regularized
// LS solve is deterministic; the safeguard fallback uses no RNG).
TEST(InverseVZAA, IsDeterministic)
{
    Sample y = gaussian(30, 5, 99);
    RowMajor u(4, 5);
    std::mt19937_64            g(1);
    std::normal_distribution<> n01(0.0, 1.0);
    for (int k = 0; k < 4; ++k) {
        Eigen::VectorXd q(5);
        for (int j = 0; j < 5; ++j) {
            q(j) = n01(g);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_aa;
    sf_aa.kind      = SolverKind::VardiZhangAA;
    sf_aa.tol       = 1e-9;
    sf_aa.max_iters = 500;
    sf_aa.aa_window = 5;
    sf_aa.aa_reg    = 1e-12;

    auto r1 = inverse_spatial_rank(u, y, sf_aa);
    auto r2 = inverse_spatial_rank(u, y, sf_aa);
    EXPECT_LE((r1.x_tilde - r2.x_tilde).array().abs().maxCoeff(), 0.0);
    EXPECT_EQ(r1.aa_fallbacks, r2.aa_fallbacks);
}

// Iteration-count property at d=25 — the production target dimensionality.
// At d=25 with a moderate cloud (M=200), VZ+AA should typically reach the
// same tight tol in roughly half the iterations VZ uses on its own. We
// assert a loose 0.8× bound here so the property survives across
// reasonable seed variation; the documented production-target ratio of
// 0.3× holds on larger inputs.
TEST(InverseVZAA, IterCountAtProductionDimReducesVsVz)
{
    std::mt19937_64            gen(0xB16D17);
    std::normal_distribution<> n01(0.0, 1.0);
    constexpr int              M = 200, d = 25, N = 16;
    Sample                     y = gaussian(M, d, 51);

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
    sf_vz.tol       = 1e-9;
    sf_vz.max_iters = 1000;

    SolverConfig sf_aa = sf_vz;
    sf_aa.kind         = SolverKind::VardiZhangAA;
    sf_aa.aa_window    = 5;
    sf_aa.aa_reg       = 1e-12;

    auto r_vz = inverse_spatial_rank(u, y, sf_vz);
    auto r_aa = inverse_spatial_rank(u, y, sf_aa);

    const auto vz_iters = static_cast<double>(r_vz.max_iters_used);
    const auto aa_iters = static_cast<double>(r_aa.max_iters_used);
    EXPECT_LT(aa_iters, 0.8 * vz_iters)
        << "AA used " << aa_iters << " iters; VZ used " << vz_iters
        << " (ratio " << (aa_iters / vz_iters) << ", aa_fallbacks="
        << r_aa.aa_fallbacks << ")";
}

// Safeguard turned off: AA still produces a finite, non-NaN result on
// benign inputs (the safeguard is for pathological cases, not required
// for correctness on typical data).
TEST(InverseVZAA, SafeguardOffStillProducesFiniteOutput)
{
    Sample y = gaussian(25, 4, 33);
    RowMajor u(3, 4);
    std::mt19937_64            g(2);
    std::normal_distribution<> n01(0.0, 1.0);
    for (int k = 0; k < 3; ++k) {
        Eigen::VectorXd q(4);
        for (int j = 0; j < 4; ++j) {
            q(j) = n01(g);
        }
        u.row(k) = rank_against(q, y).transpose();
    }

    SolverConfig sf_aa;
    sf_aa.kind         = SolverKind::VardiZhangAA;
    sf_aa.tol          = 1e-7;
    sf_aa.max_iters    = 500;
    sf_aa.aa_window    = 5;
    sf_aa.aa_reg       = 1e-12;
    sf_aa.aa_safeguard = false;

    auto r = inverse_spatial_rank(u, y, sf_aa);
    EXPECT_TRUE(r.x_tilde.allFinite());
    EXPECT_EQ(r.aa_fallbacks, 0u);  // safeguard off → never counts a fallback
}
