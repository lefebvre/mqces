#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mqces {

// A sample is an (n_specimens × n_features) row-major matrix. Rows are
// individual specimens drawn from the underlying distribution; columns are
// features (e.g. isotope ratios). Row-major lets us iterate one specimen at
// a time without strided loads.
using Sample = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Diagonal of W in Eq. 3 — one weight per feature.
using FeatureWeights = Eigen::VectorXd;

struct Class {
    std::string name;
    Sample      specimens;
};

struct Score {
    std::string class_name;
    double      s_xy;  // Eq. 3, evaluated on the observed (unperturbed) samples
};

struct ClassificationResult {
    std::string        best_class;
    std::vector<Score> scores;  // sorted ascending; scores.front().class_name == best_class
    double             misclassification_prob = 0.0;  // Eq. 7 vs the runner-up
    bool               none_of_the_above = false;
};

// Controls the measurement-error term of the misclassification probability.
// Specimen sampling variability is always estimated (by leave-one-out
// jackknife) and needs no configuration.
struct UncertaintyConfig {
    double        epsilon = 0.0;       // measurement error fraction (Eq. 10); 0 disables the term
    std::size_t   mc_samples = 10;     // perturbation replicates, >= 2; used only when epsilon > 0
    std::uint64_t seed = 0xC0FFEEULL;  // deterministic; per-replicate streams derived
};

struct ClassifierOptions {
    FeatureWeights    weights;  // diag(W); required, no default — see Eq. 4
    UncertaintyConfig uncertainty{};
    double            nota_threshold = 0.05;  // p-value in (0, 1)
    // Random splits drawn by the none-of-the-above permutation test; 0 skips
    // the test. The smallest attainable p-value is 1 / (nota_permutations + 1),
    // so this must be at least 1 / nota_threshold - 1.
    std::size_t nota_permutations = 199;
    int         n_threads = 0;  // 0 == auto (OpenMP runtime default)
};

}  // namespace mqces
