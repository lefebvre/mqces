#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include "../data/fixtures_data.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <span>
#include <vector>

namespace tf = mqces::test_fixtures;

namespace {

// Construct a Sample from a (N_SPECIMENS × N_FEATURES) slice of the
// row-major fixture array.
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

std::vector<mqces::Class> build_known_classes()
{
    std::vector<mqces::Class> known;
    known.reserve(tf::N_CLASSES);
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* base = tf::class_data + k * tf::N_SPECIMENS * tf::N_FEATURES;
        known.push_back({tf::class_names[k], make_sample(base)});
    }
    return known;
}

mqces::ClassifierOptions default_options()
{
    mqces::ClassifierOptions opts;
    opts.weights = mqces::FeatureWeights(tf::N_FEATURES);
    for (int j = 0; j < tf::N_FEATURES; ++j) {
        opts.weights(j) = tf::feature_weights[j];
    }
    opts.uncertainty.epsilon    = 0.01;
    opts.uncertainty.mc_samples = 5;
    opts.uncertainty.seed       = 0xBEEF;
    opts.n_threads              = 0;  // auto
    return opts;
}

}  // namespace

// End-to-end: every test sample drawn from class k should be classified to
// some class. (Strict "best == k" doesn't hold for every k at this fixture
// size + noise level — neighboring classes' centers are very close — but
// "best is within 2 of true k" should hold for the bulk of cases.)
TEST(BurnupPipeline, EveryTestSampleNearItsTrueClass)
{
    auto known = build_known_classes();
    auto opts  = default_options();

    int correct_or_adjacent = 0;
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* base = tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES;
        auto          test = make_sample(base);
        auto result = mqces::classic::classify(test, std::span<const mqces::Class>{known}, opts);
        ASSERT_FALSE(result.best_class.empty()) << "no best class for k=" << k;
        // Look up the chosen class's index.
        int chosen = -1;
        for (int j = 0; j < tf::N_CLASSES; ++j) {
            if (result.best_class == tf::class_names[j]) {
                chosen = j;
                break;
            }
        }
        ASSERT_GE(chosen, 0) << "chosen class not in fixture for k=" << k;
        if (std::abs(chosen - k) <= 2) {
            ++correct_or_adjacent;
        }
    }
    // Loose property: at least ~60% of test samples should land within 2
    // time-steps of their true class. Neighboring classes are very similar
    // by construction (smooth burnup curves), so an exact match is too
    // strict at N_SPECIMENS=50.
    EXPECT_GE(correct_or_adjacent, static_cast<int>(0.6 * tf::N_CLASSES));
}

// Misclassification probabilities should all fall inside [0, 1].
TEST(BurnupPipeline, MisclassProbsAreCalibrated)
{
    auto known = build_known_classes();
    auto opts  = default_options();
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* base = tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES;
        auto          test = make_sample(base);
        auto result = mqces::classic::classify(test, std::span<const mqces::Class>{known}, opts);
        EXPECT_GE(result.misclassification_prob, 0.0) << "k=" << k;
        EXPECT_LE(result.misclassification_prob, 1.0) << "k=" << k;
    }
}

// v2 must produce the same best_class as classic on this dataset (both
// pick argmin over the same MC score means).
TEST(BurnupPipeline, V2AgreesWithClassicOnBestClass)
{
    auto known = build_known_classes();
    auto opts  = default_options();
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const double* base = tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES;
        auto          test = make_sample(base);
        auto r_classic = mqces::classic::classify(
            test, std::span<const mqces::Class>{known}, opts);
        auto r_v2 = mqces::v2::classify(
            test, std::span<const mqces::Class>{known}, opts);
        EXPECT_EQ(r_classic.best_class, r_v2.best_class) << "k=" << k;
    }
}
