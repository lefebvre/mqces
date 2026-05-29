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

// Inner-solver selection for inverse_spatial_rank (Eq. 2). All three kinds
// are implemented (Weiszfeld in v1/v2, VardiZhang in v3, VardiZhangAA in
// v4) — see mqces/detail/inverse_solvers.hpp.
enum class SolverKind : std::uint8_t {
    Weiszfeld,       // v1, v2 — plain Weiszfeld
    VardiZhang,      // v3 — VZ with subgradient vertex correction
    VardiZhangAA,    // v4 — VZ + Type-II Anderson acceleration
};

struct SolverConfig {
    SolverKind  kind          = SolverKind::Weiszfeld;
    double      tol           = 1e-9;
    std::size_t max_iters     = 200;
    // VardiZhang knobs (ignored for plain Weiszfeld):
    double      vz_vertex_eps = 1e-8;  // "near a y_i" radius for subgradient correction
    // Anderson-acceleration knobs (ignored for non-AA kinds). Convergence
    // for AA is judged on the fixed-point residual ||G(x) − x||, not the
    // step length, since AA's history-mixing makes step length a poor
    // proxy for proximity to the solution.
    int         aa_window     = 5;
    double      aa_reg        = 1e-12;
    bool        aa_safeguard  = true;
};

// Reference-subset approximation for spatial_rank / inverse_spatial_rank.
// `reference_size == 0` means "exact" (use all N points); positive values
// switch to the O(N·R) approximate kernels. Strategies:
//
//   - Uniform: random R-subset, O(N·R) cost. The recommended production
//     strategy — combines well with VardiZhangAA to give an ~20× wall-time
//     win at the N=2k proxy scale vs exact Weiszfeld.
//   - Stratified: reserved for a future refinement.
//   - KdTreeLocalExact: Barnes-Hut traversal over a balanced k-d tree of
//     the cloud. Nearby points contribute exactly; subtrees that subtend
//     a small angle from the query (controlled by `kd_opening_theta`)
//     contribute via their centroid + count (multipole-zero
//     approximation). The `reference_size` field is unused for this
//     strategy — the cost is determined by `kd_opening_theta` and
//     `kd_leaf_size` instead.
//
//     CAVEAT (inverse_spatial_rank): the Barnes-Hut approximation
//     introduces a per-step bias in the Weiszfeld fixed-point map. At
//     the operating points benchmarked (d=9, 25; M=500..5000;
//     Gaussian-like clouds), the bias either (a) is dominated by tree
//     overhead at tight theta — i.e. the criterion rarely fires, so the
//     tree path matches the matrix path to machine precision but pays
//     traversal overhead, or (b) exceeds the solver tolerance at loose
//     theta — the iteration stalls and `inverse_spatial_rank` throws
//     non-convergence. Spatial_rank itself benefits from the tree (no
//     fixed-point structure to break); inverse_spatial_rank under
//     KdTreeLocalExact is best treated as a correctness-equivalent of
//     the exact path until a contractivity-preserving tree variant
//     replaces vanilla Barnes-Hut.
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
    // 0 == auto (OpenMP runtime default). NOTE: when > 0, classify()
    // calls omp_set_num_threads(), which mutates the *global* OpenMP
    // thread cap for the calling process. Subsequent OpenMP work in the
    // same process (mqces or otherwise) will see this cap until something
    // overrides it. Callers embedding mqces alongside other OpenMP code
    // should either leave this at 0 (no global mutation) or save/restore
    // omp_get_max_threads() around classify() calls.
    int               n_threads = 0;
};

}  // namespace mqces
