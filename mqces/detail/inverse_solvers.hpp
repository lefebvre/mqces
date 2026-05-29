#pragma once

#include <mqces/types.hpp>

#include <mqces/detail/kdtree.hpp>

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
// Three solver kinds are implemented behind a single SolverConfig:
//
//   Weiszfeld    — plain fixed-point iteration; plateaus near a y_i.
//   VardiZhang   — adds the Vardi & Zhang (2000) subgradient correction
//                  when the iterate falls within `vz_vertex_eps` of some
//                  y_{i*}, which removes the plateau.
//   VardiZhangAA — VZ inner step with Type-II Anderson acceleration on
//                  top (Walker & Ni 2011; Toth & Kelley 2015). Cuts
//                  iteration count substantially at high feature counts.
//
// The iteration loops are templated over a `Cloud` concept so that the
// same loop body works for two backings of `y`:
//
//   MatrixCloud  — y as a dense (M × d) row-major matrix, O(M) per step.
//                  Exact; the default for inverse_spatial_rank.
//   TreeCloud    — y wrapped in a balanced k-d tree, with Barnes-Hut
//                  multipole-zero summarization of far subtrees. O(k·log M)
//                  per step on average, where k is the number of leaves
//                  the iterate has to open exactly. Engaged when the user
//                  selects SamplingConfig::Strategy::KdTreeLocalExact.

namespace mqces::detail {

struct WeiszfeldResult {
    Eigen::VectorXd x;
    std::size_t     iters = 0;
    double          residual = 0.0;
    bool            converged = false;
    // Populated when using SolverKind::VardiZhangAA; zero otherwise.
    std::size_t     aa_fallbacks = 0;
    std::size_t     aa_restarts  = 0;
};

namespace internal {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Result of one Weiszfeld-style sum at a query point `x`:
//   numerator = M·u + Σ y_i / d_i           (component-wise vector sum)
//   inv_d_sum = Σ 1 / d_i                   (scalar)
//   min_d     = ||x − y_{i*}||              (distance to nearest point)
//   i_star    = index of the nearest point (or −1 if M == 0)
//
// Points coincident with x (within `eps`) are skipped — they contribute
// an undefined unit vector.
struct WeiszfeldSums {
    Eigen::VectorXd numerator;
    double          inv_d_sum = 0.0;
    double          min_d     = std::numeric_limits<double>::infinity();
    Eigen::Index    i_star    = -1;
};

// Cloud backing the per-iterate sums with a dense matrix. Each step costs
// O(M·d).
struct MatrixCloud {
    Eigen::Ref<const RowMajorMatrix> y;
    double                           eps;

    [[nodiscard]] double M() const { return static_cast<double>(y.rows()); }

    [[nodiscard]] Eigen::VectorXd initial_guess(
        const Eigen::Ref<const Eigen::VectorXd>& u) const
    {
        return y.colwise().mean().transpose() + u;
    }

    [[nodiscard]] Eigen::VectorXd point(Eigen::Index idx) const
    {
        return y.row(idx).transpose();
    }

    [[nodiscard]] WeiszfeldSums weiszfeld_sums(
        const Eigen::VectorXd& x, const Eigen::VectorXd& mu) const
    {
        WeiszfeldSums s;
        s.numerator = mu;
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            Eigen::VectorXd diff = x - y.row(i).transpose();
            double          d    = diff.norm();
            if (d < s.min_d) {
                s.min_d  = d;
                s.i_star = i;
            }
            if (d < eps) {
                continue;
            }
            s.numerator += y.row(i).transpose() / d;
            s.inv_d_sum += 1.0 / d;
        }
        return s;
    }

    // Subgradient sum centered at y_{i*}:
    //   g = M·u + Σ_{j ≠ i*} (y_j − y_{i*}) / ||y_{i*} − y_j||
    [[nodiscard]] Eigen::VectorXd subgradient_at(
        Eigen::Index i_star, const Eigen::VectorXd& mu) const
    {
        Eigen::VectorXd y_star = y.row(i_star).transpose();
        Eigen::VectorXd g      = mu;
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
        return g;
    }
};

// Cloud backing the per-iterate sums with a Barnes-Hut k-d tree traversal.
// Far subtrees are summarized via (centroid, count) — i.e. Σ_{i in subtree}
// y_i / d_i is approximated by (count · centroid) / d_centroid. Near
// subtrees are opened exactly. The opening criterion is bbox_extent /
// dist_to_centroid < opening_theta; smaller theta → fewer approximations,
// closer to exact but more work per step.
struct TreeCloud {
    const KdTree* tree;
    double        opening_theta;
    double        eps;

    [[nodiscard]] double M() const { return static_cast<double>(tree->size()); }

    [[nodiscard]] Eigen::VectorXd initial_guess(
        const Eigen::Ref<const Eigen::VectorXd>& u) const
    {
        return tree->points().colwise().mean().transpose() + u;
    }

    [[nodiscard]] Eigen::VectorXd point(Eigen::Index idx) const
    {
        return tree->points().row(idx).transpose();
    }

    [[nodiscard]] WeiszfeldSums weiszfeld_sums(
        const Eigen::VectorXd& x, const Eigen::VectorXd& mu) const
    {
        const auto&   pts = tree->points();
        WeiszfeldSums s;
        s.numerator = mu;
        tree->traverse(
            x, opening_theta,
            [&](Eigen::Index idx) {
                Eigen::VectorXd diff = x - pts.row(idx).transpose();
                double          d    = diff.norm();
                if (d < s.min_d) {
                    s.min_d  = d;
                    s.i_star = idx;
                }
                if (d < eps) {
                    return;
                }
                s.numerator += pts.row(idx).transpose() / d;
                s.inv_d_sum += 1.0 / d;
            },
            [&](const Eigen::VectorXd& centroid, std::size_t count) {
                Eigen::VectorXd diff = x - centroid;
                double          d    = diff.norm();
                if (d < eps) {
                    return;
                }
                const double cnt = static_cast<double>(count);
                // Multipole-zero approximation: treat the subtree as `count`
                // copies of a single point at `centroid`. Bias is O(extent²),
                // controlled by opening_theta.
                s.numerator += cnt * centroid / d;
                s.inv_d_sum += cnt / d;
            });
        return s;
    }

    // Tree-aware subgradient at y_{i*}. Since y_star is itself in the
    // cloud, the leaf containing i_star will always be opened exactly
    // (the centroid of that leaf is within eps of y_star, so the opening
    // criterion fires); the inner exact_fn skips j == i_star.
    [[nodiscard]] Eigen::VectorXd subgradient_at(
        Eigen::Index i_star, const Eigen::VectorXd& mu) const
    {
        const auto&     pts    = tree->points();
        Eigen::VectorXd y_star = pts.row(i_star).transpose();
        Eigen::VectorXd g      = mu;
        tree->traverse(
            y_star, opening_theta,
            [&](Eigen::Index idx) {
                if (idx == i_star) {
                    return;
                }
                Eigen::VectorXd diff = pts.row(idx).transpose() - y_star;
                double          d    = diff.norm();
                if (d < eps) {
                    return;
                }
                g += diff / d;
            },
            [&](const Eigen::VectorXd& centroid, std::size_t count) {
                Eigen::VectorXd diff = centroid - y_star;
                double          d    = diff.norm();
                if (d < eps) {
                    return;
                }
                g += static_cast<double>(count) * diff / d;
            });
        return g;
    }
};

// Plain Weiszfeld iteration with coincident-point skipping. Convergence
// criterion: step length (which for plain Weiszfeld equals the
// fixed-point residual since x_{k+1} = G(x_k)).
template <class Cloud>
inline WeiszfeldResult weiszfeld_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Cloud&                             cloud,
    double                                   tol,
    std::size_t                              max_iters)
{
    const double          M_d = cloud.M();
    const Eigen::VectorXd mu  = M_d * u;
    Eigen::VectorXd       x   = cloud.initial_guess(u);

    WeiszfeldResult r;
    for (std::size_t k = 0; k < max_iters; ++k) {
        WeiszfeldSums sums = cloud.weiszfeld_sums(x, mu);
        if (sums.inv_d_sum <= 0.0) {
            // All points coincided with x — pathological; bail.
            r.x         = x;
            r.iters     = k;
            r.residual  = std::numeric_limits<double>::infinity();
            r.converged = false;
            return r;
        }
        Eigen::VectorXd x_next   = sums.numerator / sums.inv_d_sum;
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
template <class Cloud>
inline std::optional<Eigen::VectorXd> vz_step(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Cloud&                             cloud,
    const Eigen::VectorXd&                   x,
    double                                   vz_vertex_eps)
{
    const double          M_d = cloud.M();
    const Eigen::VectorXd mu  = M_d * u;

    WeiszfeldSums sums = cloud.weiszfeld_sums(x, mu);
    if (sums.inv_d_sum <= 0.0) {
        return std::nullopt;
    }
    Eigen::VectorXd T_x = sums.numerator / sums.inv_d_sum;

    if (sums.min_d >= vz_vertex_eps || sums.i_star < 0) {
        return T_x;
    }

    Eigen::VectorXd y_star    = cloud.point(sums.i_star);
    Eigen::VectorXd g         = cloud.subgradient_at(sums.i_star, mu);
    const double    R_mag     = g.norm();
    const double    step_to_T = (T_x - y_star).norm();
    const double    gamma     = std::min(1.0, R_mag / std::max(cloud.eps, step_to_T));
    Eigen::VectorXd result    = (1.0 - gamma) * y_star + gamma * T_x;
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
template <class Cloud>
inline WeiszfeldResult vardi_zhang_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Cloud&                             cloud,
    double                                   tol,
    std::size_t                              max_iters,
    double                                   vz_vertex_eps)
{
    Eigen::VectorXd x = cloud.initial_guess(u);
    WeiszfeldResult r;
    for (std::size_t k = 0; k < max_iters; ++k) {
        auto step_opt = vz_step(u, cloud, x, vz_vertex_eps);
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
//
// Convergence is checked on the fixed-point residual ||G(x_{k+1}) − x_{k+1}||,
// NOT the step length ||x_{k+1} − x_k||. AA's history mixing can produce
// small step lengths while the iterate is still far from a fixed point;
// using the residual avoids accepting such intermediates as converged.
template <class Cloud>
inline WeiszfeldResult vardi_zhang_aa_iterate(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Cloud&                             cloud,
    double                                   tol,
    std::size_t                              max_iters,
    double                                   vz_vertex_eps,
    int                                      aa_window,
    double                                   aa_reg,
    bool                                     aa_safeguard)
{
    const Eigen::Index d_dim = u.size();
    const auto         m     = static_cast<std::size_t>(std::max(2, aa_window));
    Eigen::VectorXd    x     = cloud.initial_guess(u);

    WeiszfeldResult r;

    // Seed history with the initial point's G(x).
    auto gx_opt = vz_step(u, cloud, x, vz_vertex_eps);
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
        auto gx_next_opt = vz_step(u, cloud, x_next, vz_vertex_eps);

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
            gx_next_opt = vz_step(u, cloud, x_next, vz_vertex_eps);
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

        x  = std::move(x_next);
        gx = std::move(*gx_next_opt);
        // AA's step_len is ||x_{k+1} - x_k||, which can be small even when
        // the iterate is not near a fixed point (AA mixes history). The
        // fixed-point residual ||G(x_{k+1}) - x_{k+1}|| is the right
        // convergence signal; use it for both the reported residual and
        // the stopping criterion.
        const double residual = (gx - x).norm();
        r.iters               = k + 1;
        r.residual            = residual;

        x_hist.push_back(x);
        gx_hist.push_back(gx);
        while (x_hist.size() > m) {
            x_hist.pop_front();
            gx_hist.pop_front();
        }

        if (residual < tol) {
            r.x         = std::move(x);
            r.converged = true;
            return r;
        }
    }
    r.x         = std::move(x);
    r.converged = false;
    return r;
}

// Dispatch by SolverConfig.kind over an arbitrary cloud type. The
// `default` arm preserves forward-compat with any future enum additions
// by falling back to plain Weiszfeld.
template <class Cloud>
inline WeiszfeldResult solve_inverse_rank_row_with_cloud(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Cloud&                             cloud,
    const SolverConfig&                      cfg)
{
    switch (cfg.kind) {
        case SolverKind::Weiszfeld:
            return weiszfeld_iterate(u, cloud, cfg.tol, cfg.max_iters);
        case SolverKind::VardiZhang:
            return vardi_zhang_iterate(u, cloud, cfg.tol, cfg.max_iters, cfg.vz_vertex_eps);
        case SolverKind::VardiZhangAA:
            return vardi_zhang_aa_iterate(
                u, cloud, cfg.tol, cfg.max_iters, cfg.vz_vertex_eps,
                cfg.aa_window, cfg.aa_reg, cfg.aa_safeguard);
        default:
            return weiszfeld_iterate(u, cloud, cfg.tol, cfg.max_iters);
    }
}

}  // namespace internal

// Matrix-backed per-row solve. The historical surface — `y` as a dense
// matrix view, all M points contribute exactly to every step.
inline WeiszfeldResult solve_inverse_rank_row(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& y,
    const SolverConfig&                                     cfg,
    double                                                  eps = 1e-12)
{
    internal::MatrixCloud cloud{y, eps};
    return internal::solve_inverse_rank_row_with_cloud(u, cloud, cfg);
}

// Tree-backed per-row solve. `tree` wraps the same y cloud but enables
// Barnes-Hut approximation of far subtrees during each iteration. Used
// when SamplingConfig::Strategy::KdTreeLocalExact is selected. Caller
// builds the tree once and reuses it across rows of u.
inline WeiszfeldResult solve_inverse_rank_row(
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const KdTree&                            tree,
    double                                   opening_theta,
    const SolverConfig&                      cfg,
    double                                   eps = 1e-12)
{
    internal::TreeCloud cloud{&tree, opening_theta, eps};
    return internal::solve_inverse_rank_row_with_cloud(u, cloud, cfg);
}

}  // namespace mqces::detail
