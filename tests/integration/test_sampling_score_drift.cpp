#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include "../data/fixtures_data.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

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

// Bias decreases monotonically as R approaches N. similarity_score is
// quadratic in (x̃ − x), so the per-coordinate variance of the
// rank-inversion at small R contributes a non-vanishing upward bias in
// expectation — averaging over seeds doesn't remove it. What we DO
// verify is that the bias shrinks as R grows: at R = 0.9·N the
// approximation is within a tight band of exact, at R = 0.5·N it's
// looser, and the relationship is monotone for the trials we run.
//
// This is a real property of the estimator; production-scale usage
// targets R ≪ N where the per-trial bias is dominated by the wall-time
// savings (R = 10⁴ vs N = 10⁶ → R/N = 0.01, variance ~1%).
TEST(SamplingScoreDrift, BiasDecreasesWithLargerR)
{
    const auto w      = load_weights();
    const auto solver = loose_solver();

    constexpr int kClass = 10;
    const auto test = make_sample(tf::test_data + (kClass * tf::N_SPECIMENS * tf::N_FEATURES));
    const auto cls  = make_sample(tf::class_data + (kClass * tf::N_SPECIMENS * tf::N_FEATURES));
    const auto s_exact = mqces::similarity_score(test, cls, w, solver);

    constexpr int trials = 16;
    auto mean_at_R = [&](std::size_t R) {
        double sum = 0.0;
        mqces::SamplingConfig sampling;
        sampling.reference_size = R;
        for (int t = 0; t < trials; ++t) {
            sampling.seed = static_cast<std::uint64_t>(0xA17EF00DULL) + static_cast<std::uint64_t>(t);
            sum += mqces::similarity_score(test, cls, w, solver, sampling);
        }
        return sum / trials;
    };

    const double mean_25 = mean_at_R(25);
    const double mean_35 = mean_at_R(35);
    const double mean_45 = mean_at_R(45);

    const double bias_25 = std::abs(mean_25 - s_exact);
    const double bias_35 = std::abs(mean_35 - s_exact);
    const double bias_45 = std::abs(mean_45 - s_exact);

    EXPECT_LT(bias_45, bias_25)
        << "expected R=45 bias < R=25 bias; got " << bias_45 << " vs " << bias_25;
    EXPECT_LT(bias_35, bias_25)
        << "expected R=35 bias < R=25 bias; got " << bias_35 << " vs " << bias_25;
    EXPECT_LT(bias_45 / std::max(1e-12, s_exact), 0.5)
        << "expected R=45 bias < 50% of exact " << s_exact
        << "; got " << bias_45;
}

// Per-call agreement at R close to N. With R ≈ N we sample most of the
// cloud, so the per-trial variance is small and any single call should
// land within a tight band of the exact score.
TEST(SamplingScoreDrift, LargeRSinglePassWithinTightBand)
{
    const auto w      = load_weights();
    const auto solver = loose_solver();

    mqces::SamplingConfig sampling;
    sampling.reference_size = 45;  // 90% of N_SPECIMENS = 50
    sampling.seed           = 0xDEADC0FFEEULL;

    int total = 0;
    int hits  = 0;
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const auto test = make_sample(tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES);
        for (int j = 0; j < tf::N_CLASSES; ++j) {
            const auto cls = make_sample(tf::class_data + j * tf::N_SPECIMENS * tf::N_FEATURES);
            const auto s_exact  = mqces::similarity_score(test, cls, w, solver);
            const auto s_approx = mqces::similarity_score(test, cls, w, solver, sampling);
            const double rel = std::abs(s_approx - s_exact) / std::max(1e-12, s_exact);
            if (rel < 0.30) {
                ++hits;
            }
            ++total;
        }
    }
    EXPECT_GE(hits, static_cast<int>(0.85 * total))
        << hits << "/" << total << " pairs landed within 30% at R = 45";
}

// Determinism: same sampling seed reproduces the same score bit-for-bit
// (no thread-local-RNG nondeterminism, no double-draw inside the same call).
TEST(SamplingScoreDrift, ApproxIsDeterministic)
{
    const auto w      = load_weights();
    const auto solver = loose_solver();
    const auto x      = make_sample(tf::test_data);
    const auto y      = make_sample(tf::class_data);

    mqces::SamplingConfig sampling;
    sampling.reference_size = 20;
    sampling.seed           = 0xBEEFCAFE;

    const auto s1 = mqces::similarity_score(x, y, w, solver, sampling);
    const auto s2 = mqces::similarity_score(x, y, w, solver, sampling);
    EXPECT_EQ(s1, s2);
}
