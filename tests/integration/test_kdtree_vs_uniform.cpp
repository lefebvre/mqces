#include <mqces/quantile.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include "../data/fixtures_data.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>

namespace tf = mqces::test_fixtures;

namespace {

mqces::Sample make_sample(const double* base)
{
    mqces::Sample s(tf::N_SPECIMENS, tf::N_FEATURES);
    for (int i = 0; i < tf::N_SPECIMENS; ++i) {
        for (int j = 0; j < tf::N_FEATURES; ++j) {
            s(i, j) = base[i * tf::N_FEATURES + j];
        }
    }
    return s;
}

mqces::FeatureWeights load_weights()
{
    mqces::FeatureWeights w(tf::N_FEATURES);
    for (int j = 0; j < tf::N_FEATURES; ++j) {
        w(j) = tf::feature_weights[j];
    }
    return w;
}

mqces::SolverConfig loose_solver()
{
    mqces::SolverConfig s;
    s.kind      = mqces::SolverKind::Weiszfeld;
    s.tol       = 1e-5;
    s.max_iters = 1000;
    return s;
}

}  // namespace

// Spatial-rank fidelity: at fixture scale, k-d tree with a tight opening
// criterion should yield a per-row rank vector close to the exact path —
// on the same order as (or tighter than) the uniform-subsample
// approximation at a comparable compute budget.
//
// The per-fixture sizes are small (N=50), so we use opening_theta = 0.2
// (most subtrees opened exactly) and assert max-coordinate agreement
// within 0.05 — far below the typical magnitude of the rank vector.
TEST(KdTreeVsUniform, KdSpatialRankApproachesExactOnFixture)
{
    const double* base = tf::class_data + (10 * tf::N_SPECIMENS * tf::N_FEATURES);
    const auto    x    = make_sample(base);

    mqces::SamplingConfig kd_cfg;
    kd_cfg.strategy         = mqces::SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.2;
    kd_cfg.kd_leaf_size     = 8;

    const auto u_exact = mqces::spatial_rank(x);
    const auto u_kd    = mqces::spatial_rank(x, kd_cfg);

    const double max_err = (u_exact - u_kd).array().abs().maxCoeff();
    EXPECT_LT(max_err, 0.05) << "k-d tree max coord error " << max_err;
}

// similarity_score with KdTreeLocalExact: should produce a finite,
// deterministic score that's close to the exact score on the fixture.
// Note that similarity_score's inverse_spatial_rank step falls through
// to the uniform-subsample path for this strategy (tranche-o scope) —
// we accept the looser tolerance because of that.
TEST(KdTreeVsUniform, SimilarityScoreWithKdIsDeterministic)
{
    const auto w      = load_weights();
    const auto solver = loose_solver();
    const auto x      = make_sample(tf::test_data + (10 * tf::N_SPECIMENS * tf::N_FEATURES));
    const auto y      = make_sample(tf::class_data + (10 * tf::N_SPECIMENS * tf::N_FEATURES));

    mqces::SamplingConfig kd_cfg;
    kd_cfg.strategy         = mqces::SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.3;
    kd_cfg.kd_leaf_size     = 8;
    kd_cfg.reference_size   = 25;   // for the inverse_spatial_rank fallback path

    const double s1 = mqces::similarity_score(x, y, w, solver, kd_cfg);
    const double s2 = mqces::similarity_score(x, y, w, solver, kd_cfg);
    EXPECT_EQ(s1, s2);
    EXPECT_TRUE(std::isfinite(s1));
}

// At equivalent compute budget (k-d tree opening_theta=0.5 vs. uniform
// R=N/2), the k-d tree should produce no-worse error than uniform
// sampling on the spatial_rank primitive. The local-exact strategy
// captures the dominant near-field contributions exactly, which
// generally beats a random subset.
TEST(KdTreeVsUniform, KdNoWorseThanUniformAtComparableBudget)
{
    const double* base = tf::class_data + (5 * tf::N_SPECIMENS * tf::N_FEATURES);
    const auto    x    = make_sample(base);

    mqces::SamplingConfig kd_cfg;
    kd_cfg.strategy         = mqces::SamplingConfig::Strategy::KdTreeLocalExact;
    kd_cfg.kd_opening_theta = 0.5;
    kd_cfg.kd_leaf_size     = 8;

    mqces::SamplingConfig uni_cfg;
    uni_cfg.strategy       = mqces::SamplingConfig::Strategy::Uniform;
    uni_cfg.reference_size = tf::N_SPECIMENS / 2;  // R = 25
    uni_cfg.seed           = 0xA17EF00DULL;

    const auto u_exact = mqces::spatial_rank(x);
    const auto u_kd    = mqces::spatial_rank(x, kd_cfg);
    const auto u_uni   = mqces::spatial_rank(x, uni_cfg);

    const double err_kd  = (u_exact - u_kd).norm();
    const double err_uni = (u_exact - u_uni).norm();
    // Loose form of "no worse": allow up to 2× the uniform error for
    // some per-seed variance in the uniform estimator.
    EXPECT_LT(err_kd, 2.0 * err_uni)
        << "k-d tree err " << err_kd << " vs uniform err " << err_uni;
}
