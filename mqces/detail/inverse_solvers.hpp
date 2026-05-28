#pragma once

#include <mqces/types.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

// Inner solver for the inverse-spatial-rank equation (Eq. 2 of
// Weber & Dayman):
//
//     u = (1/M) Σ_i (x − y_i) / ||x − y_i||
//
// Rearrange:
//
//     x = (M u + Σ_i y_i / d_i) / Σ_i 1/d_i,    d_i = ||x − y_i||
//
// This file factors out the per-row solver behind a SolverConfig so
// future kinds (VardiZhang in tranche k, VardiZhangAA in tranche l) can
// be slotted in without touching call sites. Today only
// SolverKind::Weiszfeld is implemented; other kinds fall back to it.
//
// TODO: Plain Weiszfeld plateaus around residual ~1e-6 to ~1e-7 when the
// iterate sits near (but not on) a y_i — a known failure mode that VZ
// fixes via subgradient correction (Vardi & Zhang 2000). AA layered on
// top of VZ cuts iteration count and is the production-scale target.

namespace mqces::detail {

struct WeiszfeldResult {
    Eigen::VectorXd x;
    std::size_t     iters = 0;
    double          residual = 0.0;
    bool            converged = false;
    // Reserved for tranche (l): AA diagnostics.
    std::size_t     aa_fallbacks = 0;
    std::size_t     aa_restarts  = 0;
};

namespace internal {

// Plain Weiszfeld iteration with coincident-point skipping.
inline WeiszfeldResult weiszfeld_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& y,
    double                                                  tol,
    std::size_t                                             max_iters,
    double                                                  eps)
{
    const auto M = static_cast<double>(y.rows());

    // Initial guess: mean of y, shifted by u so that u == 0 doesn't get
    // stuck at the trivial fixed point.
    Eigen::VectorXd x = y.colwise().mean().transpose() + u;

    WeiszfeldResult r;
    for (std::size_t k = 0; k < max_iters; ++k) {
        Eigen::VectorXd numerator = M * u;
        double          inv_d_sum = 0.0;

        // Skipping y_i with d < eps is exactly the i ≠ j exclusion of Eq. 2:
        // a coincident y_i contributes an undefined unit vector and must be
        // omitted regardless of whether the iterate transiently lands on it.
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            Eigen::VectorXd diff = x - y.row(i).transpose();
            double          d    = diff.norm();
            if (d < eps) {
                continue;
            }
            numerator += y.row(i).transpose() / d;
            inv_d_sum += 1.0 / d;
        }

        if (inv_d_sum <= 0.0) {
            // All points coincided with x — pathological; bail.
            r.x         = x;
            r.iters     = k;
            r.residual  = std::numeric_limits<double>::infinity();
            r.converged = false;
            return r;
        }

        Eigen::VectorXd x_next   = numerator / inv_d_sum;
        double          step_len = (x_next - x).norm();
        x                        = std::move(x_next);
        r.iters                  = k + 1;
        r.residual               = step_len;

        if (step_len < tol) {
            r.x         = std::move(x);
            r.converged = true;
            return r;
        }
    }

    r.x         = std::move(x);
    r.converged = false;
    return r;
}

}  // namespace internal

// Dispatch by SolverConfig.kind. Today: Weiszfeld is the only path; the
// VZ and AA enum values fall back to Weiszfeld so existing callers don't
// break before tranches (k)/(l) land.
inline WeiszfeldResult solve_inverse_rank_row(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& y,
    const SolverConfig&                                     cfg,
    double                                                  eps = 1e-12)
{
    // Switch falls through to Weiszfeld for the not-yet-implemented kinds;
    // when VZ/AA land in (k)/(l) this becomes a real dispatch.
    switch (cfg.kind) {
        case SolverKind::Weiszfeld:
        case SolverKind::VardiZhang:
        case SolverKind::VardiZhangAA:
        default:
            return internal::weiszfeld_iterate(u, y, cfg.tol, cfg.max_iters, eps);
    }
}

}  // namespace mqces::detail
