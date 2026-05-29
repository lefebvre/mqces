#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <mqces/detail/kdtree.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <set>

using mqces::Sample;
using mqces::SamplingConfig;
using mqces::spatial_rank;

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

}  // namespace

// Tree construction over a small cloud and confirmation that every input
// row index is reachable through the leaves (a permutation-completeness
// invariant — no point is silently dropped during the build).
TEST(KdTree, EveryPointIsReachableThroughLeaves)
{
    constexpr int N = 200;
    constexpr int d = 4;
    Sample        x = gaussian(N, d, 0x100);
    mqces::detail::KdTree tree(x, /*leaf_size=*/16);

    // Traverse with opening_theta = 0 so EVERY subtree gets opened to a
    // leaf — no centroid approximation kicks in. Every point should be
    // visited via exact_fn exactly once.
    std::set<Eigen::Index> visited;
    Eigen::VectorXd        query = Eigen::VectorXd::Zero(d);
    tree.traverse(
        query, 0.0,
        [&](Eigen::Index idx) { visited.insert(idx); },
        [&](const Eigen::VectorXd&, std::size_t) {
            FAIL() << "approx_fn must not be called with opening_theta = 0";
        });

    EXPECT_EQ(visited.size(), static_cast<std::size_t>(N));
    for (Eigen::Index i = 0; i < N; ++i) {
        EXPECT_NE(visited.find(i), visited.end()) << "index " << i << " missing";
    }
}

// Approximation kicks in when opening_theta > 0: at least one subtree
// is summarized via centroid + count for a query that's far from the
// cloud's bulk.
TEST(KdTree, OpeningCriterionSummarizesFarSubtrees)
{
    constexpr int N = 200;
    constexpr int d = 3;
    Sample        x = gaussian(N, d, 0x101);
    mqces::detail::KdTree tree(x, 16);

    Eigen::VectorXd query = Eigen::VectorXd::Constant(d, 100.0);  // far away

    int exact_calls  = 0;
    int approx_calls = 0;
    tree.traverse(
        query, /*opening_theta=*/0.5,
        [&](Eigen::Index) { ++exact_calls; },
        [&](const Eigen::VectorXd&, std::size_t) { ++approx_calls; });

    EXPECT_GT(approx_calls, 0)
        << "far query should summarize at least one subtree";
}

// Edge: empty and singleton clouds.
TEST(KdTree, HandlesEmptyAndSingleton)
{
    Sample empty(0, 3);
    mqces::detail::KdTree t_empty(empty, 8);
    EXPECT_EQ(t_empty.size(), 0u);
    EXPECT_TRUE(t_empty.empty());

    Sample single(1, 3);
    single << 1.0, 2.0, 3.0;
    mqces::detail::KdTree t_single(single, 8);
    EXPECT_EQ(t_single.size(), 1u);
    EXPECT_FALSE(t_single.empty());

    int hits = 0;
    Eigen::VectorXd q(3);
    q << 0.0, 0.0, 0.0;
    t_single.traverse(
        q, 0.5, [&](Eigen::Index) { ++hits; }, [&](const Eigen::VectorXd&, std::size_t) {});
    EXPECT_EQ(hits, 1);
}

// spatial_rank with KdTreeLocalExact strategy at opening_theta == 0
// should match the exact path bit-for-bit (well, modulo reassociation):
// theta = 0 means no subtree is ever approximated, so we're computing
// the exact sum, just via tree traversal instead of the GEMM kernel.
TEST(KdTree, SpatialRankWithThetaZeroMatchesExact)
{
    constexpr int N = 80;
    constexpr int d = 4;
    Sample        x = gaussian(N, d, 0x200);

    SamplingConfig kd_cfg;
    kd_cfg.strategy         = SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.0;  // never approximate
    kd_cfg.kd_leaf_size     = 16;

    const auto u_exact = spatial_rank(x);
    const auto u_kd    = spatial_rank(x, kd_cfg);
    EXPECT_LE((u_exact - u_kd).array().abs().maxCoeff(), 1e-12);
}

// As opening_theta grows, the approximation error grows — but it should
// remain bounded. At theta = 0.5 on a small cloud, the per-coordinate
// error should be well below the typical magnitude of U.
TEST(KdTree, SpatialRankApproximationErrorIsBounded)
{
    constexpr int N = 150;
    constexpr int d = 4;
    Sample        x = gaussian(N, d, 0x201);

    SamplingConfig kd_cfg;
    kd_cfg.strategy         = SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.5;
    kd_cfg.kd_leaf_size     = 16;

    const auto u_exact = spatial_rank(x);
    const auto u_kd    = spatial_rank(x, kd_cfg);

    const double max_err = (u_exact - u_kd).array().abs().maxCoeff();
    EXPECT_LT(max_err, 0.5) << "k-d tree max coordinate error " << max_err;
}

// Tighter opening_theta → tighter approximation: error should shrink
// (modulo small per-seed noise) as we open more subtrees exactly.
TEST(KdTree, TighterThetaReducesApproximationError)
{
    constexpr int N = 150;
    constexpr int d = 4;
    Sample        x = gaussian(N, d, 0x202);

    SamplingConfig cfg_tight;
    cfg_tight.strategy         = SamplingConfig::Strategy::KdTreeLocalExact;
    cfg_tight.kd_opening_theta = 0.1;
    cfg_tight.kd_leaf_size     = 16;

    SamplingConfig cfg_loose       = cfg_tight;
    cfg_loose.kd_opening_theta     = 0.8;

    const auto u_exact = spatial_rank(x);
    const auto u_tight = spatial_rank(x, cfg_tight);
    const auto u_loose = spatial_rank(x, cfg_loose);

    const double err_tight = (u_exact - u_tight).norm();
    const double err_loose = (u_exact - u_loose).norm();
    EXPECT_LE(err_tight, err_loose)
        << "tighter theta should not be worse: tight=" << err_tight
        << ", loose=" << err_loose;
}

// Output is finite even with duplicate rows in the cloud (build doesn't
// infinite-recurse on equal-value medians).
TEST(KdTree, FiniteOnDuplicateRows)
{
    Sample x(10, 3);
    x.setRandom();
    x.row(2) = x.row(7);  // duplicate
    x.row(4) = x.row(7);  // triplicate
    SamplingConfig kd_cfg;
    kd_cfg.strategy         = SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.5;
    kd_cfg.kd_leaf_size     = 4;
    const auto u = spatial_rank(x, kd_cfg);
    EXPECT_TRUE(u.allFinite());
}
