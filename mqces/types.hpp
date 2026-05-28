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
    double      s_xy;  // Eq. 3
};

struct ClassificationResult {
    std::string        best_class;
    std::vector<Score> scores;  // sorted ascending; scores.front().class_name == best_class
    double             misclassification_prob = 0.0;  // Eq. 7 vs the runner-up
    bool               none_of_the_above = false;
};

struct UncertaintyConfig {
    double        epsilon = 0.0;       // measurement error fraction (Eq. 10)
    std::size_t   mc_samples = 10;     // replicate count per classification call
    std::uint64_t seed = 0xC0FFEEULL;  // deterministic; per-thread streams derived
};

struct ClassifierOptions {
    FeatureWeights    weights;  // diag(W); required, no default — see Eq. 4
    UncertaintyConfig uncertainty{};
    double            nota_threshold = 0.05;
    int               n_threads = 0;  // 0 == auto (OpenMP runtime default)
};

}  // namespace mqces
