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

// Inner-solver selection for inverse_spatial_rank (Eq. 2). VardiZhang and
// VardiZhangAA are reserved here so the public API stabilizes early; their
// implementations land in tranches (k) and (l). For now, only Weiszfeld is
// honored — selecting another kind falls back to Weiszfeld.
enum class SolverKind : std::uint8_t {
    Weiszfeld,       // v1, v2 — current behavior
    VardiZhang,      // v3 — reserved
    VardiZhangAA,    // v4 — reserved
};

struct SolverConfig {
    SolverKind  kind          = SolverKind::Weiszfeld;
    double      tol           = 1e-9;
    std::size_t max_iters     = 200;
    // VardiZhang knobs (ignored for plain Weiszfeld):
    double      vz_vertex_eps = 1e-8;  // "near a y_i" radius for subgradient correction
    // Anderson-acceleration knobs (ignored for non-AA kinds):
    int         aa_window     = 5;
    double      aa_reg        = 1e-12;
    bool        aa_safeguard  = true;
};

// Reference-subset approximation for spatial_rank / inverse_spatial_rank.
// `reference_size == 0` means "exact" (use all N points); positive values
// switch to the O(N·R) approximate kernels. Strategies:
//
//   - Uniform: random R-subset, O(N·R) cost. Default.
//   - Stratified: reserved for a future refinement.
//   - KdTreeLocalExact: Barnes-Hut traversal over a balanced k-d tree of
//     the cloud. Nearby points contribute exactly; subtrees that subtend
//     a small angle from the query (controlled by `kd_opening_theta`)
//     contribute via their centroid + count (multipole-zero
//     approximation). The `reference_size` field is unused for this
//     strategy — the cost is determined by `kd_opening_theta` and
//     `kd_leaf_size` instead.
struct SamplingConfig {
    enum class Strategy : std::uint8_t {
        Uniform,
        Stratified,
        KdTreeLocalExact,
    };
    std::size_t   reference_size   = 0;          // 0 == exact (Uniform path)
    Strategy      strategy         = Strategy::Uniform;
    std::uint64_t seed             = 0xACEBEEFULL;
    // K-d tree knobs (only meaningful when strategy == KdTreeLocalExact):
    double        kd_opening_theta = 0.5;   // Barnes-Hut opening criterion
    std::size_t   kd_leaf_size     = 32;    // max points per leaf node
};

struct ClassifierOptions {
    FeatureWeights    weights;  // diag(W); required, no default — see Eq. 4
    UncertaintyConfig uncertainty{};
    SolverConfig      solver{};
    SamplingConfig    sampling{};
    double            nota_threshold = 0.05;
    int               n_threads = 0;  // 0 == auto (OpenMP runtime default)
};

}  // namespace mqces
