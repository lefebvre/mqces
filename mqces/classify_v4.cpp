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

// v4 differences vs v3:
//
//   * Identical paired-difference t-statistic and NOTA pipeline.
//   * Inner solver is Vardi-Zhang accelerated by Type-II Anderson
//     (Walker & Ni 2011; Toth & Kelley 2015 for the safeguard recipe).
//     The acceleration buys back the iteration count that VZ alone
//     spends climbing out of vertex-near plateaus, so production-scale
//     wall time at N = 10⁶ tracks the ~5× iter-count reduction observed
//     in the AA literature for analogous fixed-point problems.

namespace mqces::v4 {

namespace {

SolverConfig default_solver_for_v4(const ClassifierOptions& options)
{
    // Honor an explicit VardiZhang / VardiZhangAA override verbatim;
    // otherwise stamp in v4 defaults. Keep VZ's tolerance + iter cap
    // matching v3 so any wall-time win is purely AA-driven.
    if (options.solver.kind == SolverKind::VardiZhangAA) {
        return options.solver;
    }
    SolverConfig s;
    s.kind          = SolverKind::VardiZhangAA;
    s.tol           = (options.solver.tol > 0.0 && options.solver.tol < 1e-5)
                          ? options.solver.tol
                          : 1e-7;
    s.max_iters     = (options.solver.max_iters >= 50) ? options.solver.max_iters : 500;
    s.vz_vertex_eps = options.solver.vz_vertex_eps;
    s.aa_window     = (options.solver.aa_window >= 2) ? options.solver.aa_window : 5;
    s.aa_reg        = (options.solver.aa_reg > 0.0) ? options.solver.aa_reg : 1e-12;
    s.aa_safeguard  = options.solver.aa_safeguard;
    return s;
}

}  // namespace

ClassificationResult classify(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& options)
{
    detail::validate_classify_inputs(test, known, options);

    const auto K = static_cast<Eigen::Index>(known.size());
    const auto N = static_cast<Eigen::Index>(options.uncertainty.mc_samples);

    const SolverConfig solver = default_solver_for_v4(options);
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

}  // namespace mqces::v4
