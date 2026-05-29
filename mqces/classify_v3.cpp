#include <mqces/classify.hpp>
#include <mqces/nota.hpp>

#include <mqces/detail/classify_internal.hpp>
#include <mqces/detail/stats.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

// v3 differences vs v2:
//
//   * Same paired-difference t-statistic as v2 (the corrected misclass
//     probability), so any best_class agreement / disagreement vs v2 is
//     purely solver-driven, not statistics-driven.
//   * Inner solver is Vardi-Zhang, which subgradient-corrects the
//     iterate when it hovers near some y_{i*}. Plain Weiszfeld plateaus
//     there at residual ~1e-6 to ~1e-7; VZ converges to the configured
//     tol (default 1e-9 here, vs the loose 1e-5 v1/v2 use).
//   * The same MC machinery as v2 — collect_replicate_scores does the
//     parallel replicate loop, only the SolverConfig differs.

namespace mqces::v3 {

namespace {

SolverConfig default_solver_for_v3(const ClassifierOptions& options)
{
    // Honor an explicit VardiZhang / VardiZhangAA override verbatim.
    // When the caller hasn't customized the solver (kind == Weiszfeld,
    // which is SolverConfig's global default), stamp in v3's tight-VZ
    // defaults: tol=1e-7 with max_iters=500. VZ converges much tighter
    // than plain Weiszfeld in principle, but on small-d clustered inputs
    // some configurations still need a few hundred iterations to reach
    // the tight tol — 200 (the global default) isn't enough.
    if (options.solver.kind == SolverKind::VardiZhang
        || options.solver.kind == SolverKind::VardiZhangAA) {
        return options.solver;
    }
    SolverConfig s;
    s.kind          = SolverKind::VardiZhang;
    s.tol           = 1e-7;
    s.max_iters     = 500;
    s.vz_vertex_eps = options.solver.vz_vertex_eps;
    return s;
}

}  // namespace

ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options)
{
    detail::validate_classify_inputs(test, known, options);

    const auto K = static_cast<Eigen::Index>(known.size());
    const auto N = static_cast<Eigen::Index>(options.uncertainty.mc_samples);

    const SolverConfig solver = default_solver_for_v3(options);
    const Eigen::MatrixXd scores
        = detail::collect_replicate_scores(test, known, options, solver);

    Eigen::VectorXd means(K);
    for (Eigen::Index k = 0; k < K; ++k) {
        means(k) = detail::row_mean(scores, k);
    }

    Eigen::Index best_idx = 0;
    means.minCoeff(&best_idx);

    std::vector<std::pair<double, Eigen::Index>> ranking;
    ranking.reserve(static_cast<std::size_t>(K));
    for (Eigen::Index k = 0; k < K; ++k) {
        ranking.emplace_back(means(k), k);
    }
    std::sort(ranking.begin(), ranking.end());

    ClassificationResult result;
    result.scores.reserve(static_cast<std::size_t>(K));
    for (const auto& [m, k] : ranking) {
        result.scores.push_back({known[static_cast<std::size_t>(k)].name, m});
    }
    result.best_class = known[static_cast<std::size_t>(best_idx)].name;

    if (K >= 2) {
        const Eigen::Index runner_up = ranking[1].second;
        // Paired-difference t-statistic: same as v2.
        Eigen::ArrayXd d = scores.row(runner_up).array() - scores.row(best_idx).array();
        const double   mean_d = d.mean();
        double         var_d  = 0.0;
        if (N > 1) {
            var_d = (d - mean_d).square().sum() / static_cast<double>(N - 1);
        }
        const double se = std::sqrt(var_d / static_cast<double>(N));
        if (se > 0.0 && std::isfinite(se)) {
            const double t  = mean_d / se;
            const double df = static_cast<double>(N - 1);
            result.misclassification_prob = detail::students_t_cdf(-t, df);
        } else {
            result.misclassification_prob = (mean_d == 0.0) ? 0.5 : 0.0;
        }
    }

    Eigen::VectorXd best_row = scores.row(best_idx).transpose();
    result.none_of_the_above = is_none_of_the_above(
        test, known[static_cast<std::size_t>(best_idx)], best_row, options);

    return result;
}

}  // namespace mqces::v3
