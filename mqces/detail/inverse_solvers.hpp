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

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Plain Weiszfeld iteration with coincident-point skipping.
inline WeiszfeldResult weiszfeld_iterate(
    const Eigen::Ref<const Eigen::VectorXd>&       u,
    const Eigen::Ref<const RowMajorMatrix>&        y,
    double                                         tol,
    std::size_t                                    max_iters,
    double                                         eps)
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

// Vardi-Zhang iteration. Identical to plain Weiszfeld when the iterate is
// safely far from every y_i; when the iterate falls within
// `vz_vertex_eps` of some y_{i*}, applies the subgradient correction of
// Vardi & Zhang (2000) PNAS 97:1423–1426 (adapted for the
// solve-u = G(x) form of Eq. 2):
//
//   T(x)    = standard Weiszfeld step
//   g       = M·u + Σ_{j ≠ i*} (y_j − y_{i*}) / ||y_{i*} − y_j||
//   R_mag   = ||g||             (= M · ||residual at y_{i*}||)
//   γ       = min(1, R_mag / ||T(x) − y_{i*}||)
//   x_next  = (1 − γ)·y_{i*} + γ·T(x)
//
// γ → 0 when y_{i*} is near a true solution (small residual), so the
// iterate stays put and convergence is signalled. γ → 1 when the vertex
// is far from a solution, so we take the full Weiszfeld step. This
// removes the residual plateau of plain Weiszfeld near vertices.
inline WeiszfeldResult vardi_zhang_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const RowMajorMatrix>&  y,
    double                                   tol,
    std::size_t                              max_iters,
    double                                   vz_vertex_eps,
    double                                   eps)
{
    const auto      M_d = static_cast<double>(y.rows());
    Eigen::VectorXd x   = y.colwise().mean().transpose() + u;

    WeiszfeldResult r;
    for (std::size_t k = 0; k < max_iters; ++k) {
        // ---- Standard Weiszfeld step T(x) ----
        Eigen::VectorXd numerator = M_d * u;
        double          inv_d_sum = 0.0;
        double          min_d_x   = std::numeric_limits<double>::infinity();
        Eigen::Index    i_star    = -1;
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            Eigen::VectorXd diff = x - y.row(i).transpose();
            double          d    = diff.norm();
            if (d < min_d_x) {
                min_d_x = d;
                i_star  = i;
            }
            if (d < eps) {
                continue;
            }
            numerator += y.row(i).transpose() / d;
            inv_d_sum += 1.0 / d;
        }
        if (inv_d_sum <= 0.0) {
            r.x         = x;
            r.iters     = k;
            r.residual  = std::numeric_limits<double>::infinity();
            r.converged = false;
            return r;
        }
        Eigen::VectorXd T_x = numerator / inv_d_sum;

        // ---- Vertex correction when iterate is near some y_{i*} ----
        Eigen::VectorXd x_next;
        if (min_d_x >= vz_vertex_eps || i_star < 0) {
            x_next = T_x;
        } else {
            const auto y_star = y.row(i_star).transpose().eval();
            // g = M·u + Σ_{j ≠ i*} (y_j − y_{i*}) / ||y_{i*} − y_j||
            Eigen::VectorXd g = M_d * u;
            for (Eigen::Index j = 0; j < y.rows(); ++j) {
                if (j == i_star) {
                    continue;
                }
                Eigen::VectorXd diff = y.row(j).transpose() - y_star;
                double          d    = diff.norm();
                if (d < eps) {
                    continue;
                }
                g += diff / d;
            }
            const double R_mag      = g.norm();
            const double step_to_T  = (T_x - y_star).norm();
            const double gamma      = std::min(1.0, R_mag / std::max(eps, step_to_T));
            x_next                  = (1.0 - gamma) * y_star + gamma * T_x;
        }

        const double step_len = (x_next - x).norm();
        x                     = std::move(x_next);
        r.iters               = k + 1;
        r.residual            = step_len;
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

// Dispatch by SolverConfig.kind. Weiszfeld and VardiZhang are implemented
// today; VardiZhangAA falls back to VardiZhang until tranche (l).
inline WeiszfeldResult solve_inverse_rank_row(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& y,
    const SolverConfig&                                     cfg,
    double                                                  eps = 1e-12)
{
    switch (cfg.kind) {
        case SolverKind::Weiszfeld:
            return internal::weiszfeld_iterate(u, y, cfg.tol, cfg.max_iters, eps);
        case SolverKind::VardiZhang:
        case SolverKind::VardiZhangAA:  // AA wrapping arrives in tranche (l)
            return internal::vardi_zhang_iterate(
                u, y, cfg.tol, cfg.max_iters, cfg.vz_vertex_eps, eps);
        default:
            return internal::weiszfeld_iterate(u, y, cfg.tol, cfg.max_iters, eps);
    }
}

}  // namespace mqces::detail
