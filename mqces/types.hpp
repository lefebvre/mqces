#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mqces {

/**
 * @brief A sample is an (n_specimens × n_features) row-major matrix.
 *
 * Rows are individual specimens drawn from the underlying distribution;
 * columns are features (e.g. isotope ratios). Row-major lets us iterate
 * one specimen at a time without strided loads.
 */
using Sample = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** @brief Diagonal of W in Eq. 3 — one weight per feature. */
using FeatureWeights = Eigen::VectorXd;

/** @brief A named cloud of specimens — one of the candidate classes. */
struct Class {
    std::string name;       ///< Human-readable identifier (must be unique within a call).
    Sample      specimens;  ///< (n_specimens × n_features) measurements.
};

/** @brief Per-class similarity score result (Eq. 3). */
struct Score {
    std::string class_name;  ///< Identifier matching `Class::name`.
    double      s_xy;        ///< Eq. 3 score; lower is better.
};

/** @brief Aggregate result of a single `classify()` call. */
struct ClassificationResult {
    std::string        best_class;  ///< Identifier of the winning class.
    /**
     * @brief Per-class scores, sorted ascending by `s_xy`.
     *
     * `scores.front().class_name == best_class` always holds.
     */
    std::vector<Score> scores;
    double             misclassification_prob = 0.0;  ///< Eq. 7 vs the runner-up.
    bool               none_of_the_above = false;    ///< NOTA flag (page 8 of paper).
};

/**
 * @brief Monte-Carlo measurement-error configuration (Eq. 10).
 *
 * Setting `epsilon = 0` disables perturbation entirely; `mc_samples` is
 * then unused.
 */
struct UncertaintyConfig {
    double        epsilon    = 0.0;            ///< Multiplicative-Gaussian error fraction.
    std::size_t   mc_samples = 10;             ///< Replicate count per classification call.
    std::uint64_t seed       = 0xC0FFEEULL;    ///< Deterministic; per-thread streams derived.
};

/**
 * @brief Inner-solver selection for `inverse_spatial_rank` (Eq. 2).
 *
 * See `mqces/detail/inverse_solvers.hpp` for implementation notes.
 */
enum class SolverKind : std::uint8_t {
    Weiszfeld,     ///< v1, v2 — plain Weiszfeld.
    VardiZhang,    ///< v3 — VZ with subgradient vertex correction.
    VardiZhangAA,  ///< v4 — VZ + Type-II Anderson acceleration.
};

/** @brief Per-call inner-solver tuning. */
struct SolverConfig {
    SolverKind  kind          = SolverKind::Weiszfeld;  ///< Which iteration to run.
    double      tol           = 1e-9;                   ///< Convergence tolerance.
    std::size_t max_iters     = 200;                    ///< Iteration cap.

    /// Vardi-Zhang subgradient "near a y_i" radius. Ignored for plain Weiszfeld.
    double      vz_vertex_eps = 1e-8;

    /**
     * @brief Anderson-acceleration rolling-window size.
     *
     * Convergence for AA is judged on the fixed-point residual
     * @f$\|G(x) - x\|@f$, not the step length — AA's history mixing
     * makes step length a poor proxy for proximity to the solution.
     * Ignored for non-AA kinds.
     */
    int         aa_window     = 5;
    double      aa_reg        = 1e-12;  ///< Anderson Tikhonov regularization.
    bool        aa_safeguard  = true;   ///< Toth-Kelley safeguard; revert to VZ on inflation.
};

/**
 * @brief Reference-subset configuration for `spatial_rank` and `inverse_spatial_rank`.
 *
 * `reference_size == 0` is the special "exact" sentinel — both kernels
 * dispatch to the O(N²) path. Positive values switch to the O(N·R)
 * approximate kernels.
 *
 * Strategies:
 * - **Uniform**: random R-subset, O(N·R) cost. Recommended production
 *   default; combines well with `VardiZhangAA` to give an ~20× wall-time
 *   win at the N = 2k proxy scale vs exact Weiszfeld.
 * - **Stratified**: reserved for a future refinement.
 * - **KdTreeLocalExact**: Barnes-Hut traversal over a balanced k-d tree
 *   of the cloud. Nearby points contribute exactly; subtrees that
 *   subtend a small angle from the query (controlled by
 *   `kd_opening_theta`) contribute via their centroid + count
 *   (multipole-zero approximation). `reference_size` is unused for this
 *   strategy — cost is controlled by `kd_opening_theta` and
 *   `kd_leaf_size` instead.
 *
 * @warning The `KdTreeLocalExact` strategy is production-safe only for
 *          `spatial_rank`. For `inverse_spatial_rank` the Barnes-Hut
 *          approximation introduces a per-step bias in the Weiszfeld
 *          fixed-point map: at tight `kd_opening_theta` it matches the
 *          matrix path but pays traversal overhead, and at loose
 *          `kd_opening_theta` the iteration stalls and throws
 *          non-convergence. Treat as correctness-equivalent of the
 *          exact path until a contractivity-preserving variant lands.
 */
struct SamplingConfig {
    enum class Strategy : std::uint8_t {
        Uniform,
        Stratified,
        KdTreeLocalExact,
    };
    std::size_t   reference_size   = 0;                 ///< 0 == exact (Uniform path).
    Strategy      strategy         = Strategy::Uniform; ///< Subset-selection strategy.
    std::uint64_t seed             = 0xACEBEEFULL;      ///< Deterministic stream selector.
    double        kd_opening_theta = 0.5;   ///< Barnes-Hut opening criterion (KdTreeLocalExact only).
    std::size_t   kd_leaf_size     = 32;    ///< Max points per leaf node (KdTreeLocalExact only).
};

/**
 * @brief Aggregate options forwarded to every `classify()` entry point.
 *
 * @note When `n_threads > 0`, `classify()` calls `omp_set_num_threads()`,
 *       which mutates the **process-global** OpenMP thread cap.
 *       Subsequent OpenMP work in the same process will see this cap
 *       until something overrides it. Callers embedding mqces alongside
 *       other OpenMP code should either leave `n_threads = 0` (no
 *       global mutation) or save/restore `omp_get_max_threads()` around
 *       `classify()` calls.
 */
struct ClassifierOptions {
    FeatureWeights    weights;                ///< diag(W); required, no default — see Eq. 4.
    UncertaintyConfig uncertainty{};          ///< Monte-Carlo / measurement-error knobs.
    SolverConfig      solver{};               ///< Inner-solver tuning.
    SamplingConfig    sampling{};             ///< Reference-subset tuning.
    double            nota_threshold = 0.05;  ///< NOTA p-value threshold.
    int               n_threads      = 0;     ///< 0 = auto (OpenMP default).
};

}  // namespace mqces
