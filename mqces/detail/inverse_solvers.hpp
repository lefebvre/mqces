#pragma once

#include <mqces/types.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
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

// One Vardi-Zhang fixed-point step at x. Returns G(x) — the next iterate
// under VZ — or std::nullopt if the iterate has collapsed to a
// pathological configuration (all y_i coincident with x). Shared between
// plain VZ (called once per outer iteration) and VZ+AA (called twice per
// outer iteration: once for the regular step, once for the safeguard).
inline std::optional<Eigen::VectorXd> vz_step(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const RowMajorMatrix>&  y,
    const Eigen::VectorXd&                   x,
    double                                   M_d,
    double                                   vz_vertex_eps,
    double                                   eps)
{
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
        return std::nullopt;
    }
    Eigen::VectorXd T_x = numerator / inv_d_sum;

    // ---- Vertex correction when iterate is near some y_{i*} ----
    if (min_d_x >= vz_vertex_eps || i_star < 0) {
        return T_x;
    }
    const auto      y_star = y.row(i_star).transpose().eval();
    Eigen::VectorXd g      = M_d * u;
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
    const double R_mag     = g.norm();
    const double step_to_T = (T_x - y_star).norm();
    const double gamma     = std::min(1.0, R_mag / std::max(eps, step_to_T));
    Eigen::VectorXd result = (1.0 - gamma) * y_star + gamma * T_x;
    return result;
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
        auto step_opt = vz_step(u, y, x, M_d, vz_vertex_eps, eps);
        if (!step_opt) {
            r.x         = x;
            r.iters     = k;
            r.residual  = std::numeric_limits<double>::infinity();
            r.converged = false;
            return r;
        }
        const double step_len = (*step_opt - x).norm();
        x                     = std::move(*step_opt);
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

// Vardi-Zhang inner iteration accelerated by Type-II Anderson with
// Toth-Kelley regularization (Walker & Ni 2011; Toth & Kelley 2015).
//
// At each step we form difference matrices over the last m iterates:
//
//   dF = [f_2 − f_1, ..., f_m − f_{m-1}]    (d × (m-1))
//   dG = [G_2 − G_1, ..., G_m − G_{m-1}]    (d × (m-1))
//
// where f_i = G(x_i) − x_i. We solve
//
//   γ = (dFᵀ dF + λI)^{-1} dFᵀ f_k          with λ = aa_reg · trace/(m-1)
//
// then take x_{k+1} = G(x_k) − dG·γ. The candidate is accepted only when
// the resulting residual is finite and doesn't exceed 1.5× the residual
// of the previous step; otherwise we fall back to the plain VZ step and
// reset the AA history. This safeguard is the standard recipe for
// avoiding AA-induced divergence at the cost of one extra vz_step per
// fallback.
inline WeiszfeldResult vardi_zhang_aa_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const RowMajorMatrix>&  y,
    double                                   tol,
    std::size_t                              max_iters,
    double                                   vz_vertex_eps,
    int                                      aa_window,
    double                                   aa_reg,
    bool                                     aa_safeguard,
    double                                   eps)
{
    const auto       M_d    = static_cast<double>(y.rows());
    const auto       d_dim  = static_cast<Eigen::Index>(u.size());
    const auto        m     = static_cast<std::size_t>(std::max(2, aa_window));
    Eigen::VectorXd  x      = y.colwise().mean().transpose() + u;

    WeiszfeldResult r;

    // Seed history with the initial point's G(x).
    auto gx_opt = vz_step(u, y, x, M_d, vz_vertex_eps, eps);
    if (!gx_opt) {
        r.x         = x;
        r.iters     = 0;
        r.residual  = std::numeric_limits<double>::infinity();
        r.converged = false;
        return r;
    }
    Eigen::VectorXd gx = std::move(*gx_opt);

    std::deque<Eigen::VectorXd> x_hist;
    std::deque<Eigen::VectorXd> gx_hist;
    x_hist.push_back(x);
    gx_hist.push_back(gx);

    for (std::size_t k = 0; k < max_iters; ++k) {
        const Eigen::VectorXd f_x = gx - x;

        // ---- Build the candidate via AA when we have ≥ 2 history pairs. ----
        Eigen::VectorXd x_next;
        bool            used_aa = false;
        if (x_hist.size() >= 2) {
            const auto cols = static_cast<Eigen::Index>(x_hist.size() - 1);
            Eigen::MatrixXd dF(d_dim, cols);
            Eigen::MatrixXd dG(d_dim, cols);
            for (Eigen::Index i = 0; i < cols; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                Eigen::VectorXd f_i  = gx_hist[idx]     - x_hist[idx];
                Eigen::VectorXd f_ip = gx_hist[idx + 1] - x_hist[idx + 1];
                dF.col(i)            = f_ip - f_i;
                dG.col(i)            = gx_hist[idx + 1] - gx_hist[idx];
            }

            Eigen::MatrixXd FtF    = dF.transpose() * dF;  // (m-1) × (m-1)
            const double    lambda = aa_reg * FtF.trace() / static_cast<double>(cols);
            FtF.diagonal().array() += lambda;

            const Eigen::VectorXd Ftfk  = dF.transpose() * f_x;
            const Eigen::VectorXd gamma = FtF.llt().solve(Ftfk);

            x_next  = gx - dG * gamma;
            used_aa = true;
        } else {
            x_next = gx;
        }

        // ---- Compute G(x_next) for both the safeguard and the next iter. ----
        auto gx_next_opt = vz_step(u, y, x_next, M_d, vz_vertex_eps, eps);

        bool fallback_needed = false;
        if (used_aa && aa_safeguard) {
            if (!gx_next_opt) {
                fallback_needed = true;
            } else {
                const Eigen::VectorXd f_next = *gx_next_opt - x_next;
                if (!f_next.allFinite() || f_next.norm() > 1.5 * f_x.norm()) {
                    fallback_needed = true;
                }
            }
        } else if (!gx_next_opt) {
            r.x         = x;
            r.iters     = k;
            r.residual  = std::numeric_limits<double>::infinity();
            r.converged = false;
            return r;
        }

        if (fallback_needed) {
            x_next      = gx;
            gx_next_opt = vz_step(u, y, x_next, M_d, vz_vertex_eps, eps);
            if (!gx_next_opt) {
                r.x         = x;
                r.iters     = k;
                r.residual  = std::numeric_limits<double>::infinity();
                r.converged = false;
                return r;
            }
            ++r.aa_fallbacks;
            ++r.aa_restarts;
            x_hist.clear();
            gx_hist.clear();
        }

        const double step_len = (x_next - x).norm();
        x                     = std::move(x_next);
        gx                    = std::move(*gx_next_opt);
        r.iters               = k + 1;
        r.residual            = step_len;

        x_hist.push_back(x);
        gx_hist.push_back(gx);
        while (x_hist.size() > m) {
            x_hist.pop_front();
            gx_hist.pop_front();
        }

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

// Dispatch by SolverConfig.kind. All three kinds are implemented as of
// tranche (l). The legacy fallthrough on `default` is preserved for
// forward-compat with any future enum additions.
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
            return internal::vardi_zhang_iterate(
                u, y, cfg.tol, cfg.max_iters, cfg.vz_vertex_eps, eps);
        case SolverKind::VardiZhangAA:
            return internal::vardi_zhang_aa_iterate(
                u, y, cfg.tol, cfg.max_iters, cfg.vz_vertex_eps,
                cfg.aa_window, cfg.aa_reg, cfg.aa_safeguard, eps);
        default:
            return internal::weiszfeld_iterate(u, y, cfg.tol, cfg.max_iters, eps);
    }
}

}  // namespace mqces::detail
