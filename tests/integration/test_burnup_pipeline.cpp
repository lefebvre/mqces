#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include "../data/fixtures_data.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cstdlib>
#include <span>
#include <string>
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

mqces::Sample test_sample(int k)
{
    return make_sample(tf::test_data + k * tf::N_SPECIMENS * tf::N_FEATURES);
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

int class_index(const std::string& name)
{
    for (int j = 0; j < tf::N_CLASSES; ++j) {
        if (name == tf::class_names[j]) {
            return j;
        }
    }
    return -1;
}

}  // namespace

// End-to-end: every test sample drawn from class k is assigned to class k,
// and NOTA stays quiet for all but the occasional sample (its false-positive
// rate is nota_threshold = 5%, so ~1 of 19 is expected).
TEST(BurnupPipeline, ClassifiesGenuineSamplesToTheirTrueClass)
{
    const auto known = build_known_classes();
    const auto opts  = default_options();

    int nota_fired = 0;
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        const auto result
            = mqces::classic::classify(test_sample(k), std::span<const mqces::Class>{known}, opts);
        EXPECT_EQ(class_index(result.best_class), k);
        EXPECT_LT(result.misclassification_prob, 0.5) << "k=" << k;
        nota_fired += result.none_of_the_above ? 1 : 0;
    }
    EXPECT_LE(nota_fired, 3);
}

// Withhold each sample's true class. The classifier must fall back to an
// adjacent time step and NOTA must flag that the sample belongs to none of
// the remaining classes.
//
// For interior k the sample sits between two neighbours that are nearly
// equally good matches, so an honest misclassification probability is
// substantial. Before specimen sampling variability entered the standard
// error, these came out orders of magnitude too small.
TEST(BurnupPipeline, WithheldTrueClassIsFlagged)
{
    const auto known = build_known_classes();
    auto       opts  = default_options();

    int    nota_fired   = 0;
    double interior_sum_classic = 0.0;
    double interior_sum_v2      = 0.0;
    for (int k = 0; k < tf::N_CLASSES; ++k) {
        std::vector<mqces::Class> others;
        for (int j = 0; j < tf::N_CLASSES; ++j) {
            if (j != k) {
                others.push_back(known[static_cast<std::size_t>(j)]);
            }
        }
        const auto test = test_sample(k);
        const std::span<const mqces::Class> span{others};

        opts.nota_permutations = mqces::ClassifierOptions{}.nota_permutations;
        const auto r_classic   = mqces::classic::classify(test, span, opts);
        opts.nota_permutations = 0;  // NOTA is shared; skip recomputing it
        const auto r_v2        = mqces::v2::classify(test, span, opts);

        EXPECT_EQ(std::abs(class_index(r_classic.best_class) - k), 1) << "k=" << k;
        nota_fired += r_classic.none_of_the_above ? 1 : 0;
        if (k > 0 && k < tf::N_CLASSES - 1) {
            interior_sum_classic += r_classic.misclassification_prob;
            interior_sum_v2 += r_v2.misclassification_prob;
        }
    }
    EXPECT_GE(nota_fired, tf::N_CLASSES - 1);

    const double n_interior = tf::N_CLASSES - 2;
    EXPECT_GT(interior_sum_classic / n_interior, 0.1);
    EXPECT_GT(interior_sum_v2 / n_interior, 0.1);
}
