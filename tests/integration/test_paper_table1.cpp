#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include "../data/expected_scores.hpp"
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

}  // namespace

// The full S(test_k, class_j) matrix produced by similarity_score in C++
// should match the pure-NumPy reference in expected_scores within a
// reasonable tolerance.
//
// Both implementations use Weiszfeld with tol = 1e-5, so per-coordinate
// numerical agreement is around 1e-5, and the final score (a sum of d × N
// weighted squared distances) accumulates that error multiplied by the
// largest weight (~1e4). The score values themselves range across many
// orders of magnitude; we use a generous *relative* tolerance plus a small
// absolute floor.
TEST(PaperTable1, ScoreMatrixMatchesReference)
{
    const auto w = load_weights();
    int        compared    = 0;
    int        rel_failures = 0;
    int        abs_failures = 0;

    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* test_base = tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES;
        const auto    test      = make_sample(test_base);

        for (int j = 0; j < tf::N_CLASSES; ++j) {
            const double* class_base
                = tf::class_data + j * tf::N_SPECIMENS * tf::N_FEATURES;
            const auto cls      = make_sample(class_base);
            const auto cpp_score = mqces::similarity_score(test, cls, w);
            const auto ref_score = tf::expected_scores[k * tf::N_CLASSES + j];

            const double abs_diff = std::abs(cpp_score - ref_score);
            const double rel_diff = abs_diff / std::max(1e-30, std::abs(ref_score));

            // Pass either an absolute floor (handles near-zero references)
            // or a relative tolerance.
            const bool ok = (abs_diff < 1e-3) || (rel_diff < 5e-3);
            if (!ok) {
                if (rel_diff >= 5e-3) {
                    ++rel_failures;
                }
                if (abs_diff >= 1e-3) {
                    ++abs_failures;
                }
                EXPECT_TRUE(ok)
                    << "S(test=" << k << ", class=" << j << "): "
                    << "cpp=" << cpp_score << ", ref=" << ref_score
                    << ", abs=" << abs_diff << ", rel=" << rel_diff;
            }
            ++compared;
        }
    }

    EXPECT_EQ(compared, tf::N_CLASSES * tf::N_CLASSES);
    EXPECT_EQ(rel_failures, 0);
    EXPECT_EQ(abs_failures, 0);
}

// Sanity check on the reference data itself: diagonal entries (test from
// class k vs class k specimens) should be small (the test sample is drawn
// from the same distribution), and far-off-diagonal entries should be
// large.
TEST(PaperTable1, ReferenceScoresAreOrdered)
{
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double s_kk = tf::expected_scores[k * tf::N_CLASSES + k];
        // Far-off-diagonal: pick a class on the opposite end of the range.
        const int j_far = (k < tf::N_CLASSES / 2)
                              ? (tf::N_CLASSES - 1)
                              : 0;
        const double s_kfar = tf::expected_scores[k * tf::N_CLASSES + j_far];
        EXPECT_LT(s_kk, s_kfar)
            << "Reference: S(test_" << k << ", class_" << k << ") should be < "
            << "S(test_" << k << ", class_" << j_far << ")";
    }
}
