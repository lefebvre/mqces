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

}  // namespace

// Cross-solver parity: similarity_score with VardiZhang and similarity_score
// with plain Weiszfeld should agree to within the looser Weiszfeld tolerance
// on every (test_k, class_j) pair of the 19-class fixture. VZ may converge
// tighter, but both should be within a small absolute / relative band of
// the same fixed point.
//
// (When tranche l adds AA, VZ vs VZ+AA agreement gets a tighter parity
// test — they're converging to the same fixed point at the same tol.)
TEST(SolverScoreParity, VzAgreesWithWeiszfeldOnFixture)
{
    const auto w = load_weights();

    mqces::SolverConfig solver_w;
    solver_w.kind      = mqces::SolverKind::Weiszfeld;
    solver_w.tol       = 1e-5;
    solver_w.max_iters = 1000;

    mqces::SolverConfig solver_vz;
    solver_vz.kind      = mqces::SolverKind::VardiZhang;
    solver_vz.tol       = 1e-7;  // VZ is happy to converge tighter
    solver_vz.max_iters = 500;

    int    compared    = 0;
    double max_abs_dif = 0.0;
    double max_rel_dif = 0.0;
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* test_base = tf::test_data + (k * tf::N_SPECIMENS * tf::N_FEATURES);
        const auto    test      = make_sample(test_base);
        for (int j = 0; j < tf::N_CLASSES; ++j) {
            const double* class_base
                = tf::class_data + (j * tf::N_SPECIMENS * tf::N_FEATURES);
            const auto cls = make_sample(class_base);

            const double s_w  = mqces::similarity_score(test, cls, w, solver_w);
            const double s_vz = mqces::similarity_score(test, cls, w, solver_vz);

            const double abs_dif = std::abs(s_w - s_vz);
            const double rel_dif = abs_dif / std::max(1e-12, std::abs(s_w));
            max_abs_dif = std::max(max_abs_dif, abs_dif);
            max_rel_dif = std::max(max_rel_dif, rel_dif);
            ++compared;
        }
    }

    EXPECT_EQ(compared, tf::N_CLASSES * tf::N_CLASSES);
    // Loose band: the Weiszfeld tolerance (1e-5) is the limiting factor.
    // Per-pair score is a sum of ~50 weighted squared distances; per-point
    // 1e-5 error accumulates to roughly 1e-3 in the score. Allow a
    // little headroom.
    EXPECT_LT(max_abs_dif, 1e-2)
        << "max absolute score diff " << max_abs_dif << " across solvers";
    EXPECT_LT(max_rel_dif, 0.5)
        << "max relative score diff " << max_rel_dif << " across solvers";
}

// VZ should converge tightly enough to be reproducible at 1e-9 — the
// plateau-recovery property is what's being exercised here.
TEST(SolverScoreParity, VzScoreIsReproducible)
{
    const auto w = load_weights();
    mqces::SolverConfig solver_vz;
    solver_vz.kind      = mqces::SolverKind::VardiZhang;
    solver_vz.tol       = 1e-9;
    solver_vz.max_iters = 500;

    const auto test = make_sample(tf::test_data + (10 * tf::N_SPECIMENS * tf::N_FEATURES));
    const auto cls  = make_sample(tf::class_data + (10 * tf::N_SPECIMENS * tf::N_FEATURES));

    const double s1 = mqces::similarity_score(test, cls, w, solver_vz);
    const double s2 = mqces::similarity_score(test, cls, w, solver_vz);
    EXPECT_EQ(s1, s2);
}
