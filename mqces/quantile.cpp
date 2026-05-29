#include <mqces/quantile.hpp>

#include <mqces/detail/inverse_solvers.hpp>
#include <mqces/detail/kdtree.hpp>
#include <mqces/detail/reference_sample.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace mqces {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Tile size for the GEMM-based pair-distance kernel. Chosen so a T×T
// double-precision Gram block (T²·8 bytes) fits comfortably in L2 cache
// with room for the two row strips: at T=256 that's 512 KB Gram + 2·T·d·8
// row-data, which is well within typical L2 sizes for d ≤ ~50.
constexpr Eigen::Index kTileRows = 256;

// Process one pair (i, j) given pre-computed squared norms s_i, s_j and
// their cross dot-product G_ij. Updates u.row(i) and u.row(j) with the
// symmetric ±(x_j − x_i)/d contributions.
//
// Pre-computed d² via the GEMM identity ||x_i − x_j||² = s_i + s_j − 2·G_ij
// is fast but loses precision when x_i and x_j are nearly equal
// (catastrophic cancellation). When the result is suspiciously small
// relative to s_i + s_j we recompute directly via subtraction.
inline void accumulate_pair(
    RowMajorMatrix&                                  u,
    const Eigen::Ref<const RowMajorMatrix>&          x_i_block,
    const Eigen::Ref<const RowMajorMatrix>&          x_j_block,
    Eigen::Index                                     ii,
    Eigen::Index                                     jj,
    Eigen::Index                                     i,
    Eigen::Index                                     j,
    double                                           s_i,
    double                                           s_j,
    double                                           g_ij,
    double                                           eps2)
{
    double d2 = s_i + s_j - 2.0 * g_ij;
    // Cancellation guard: if the fast formula gives a value much smaller
    // than the inputs, recompute via direct subtraction.
    const double cancel_threshold = 1e-12 * (std::abs(s_i) + std::abs(s_j));
    if (d2 < cancel_threshold) {
        Eigen::RowVectorXd diff = x_j_block.row(jj) - x_i_block.row(ii);
        d2                      = diff.squaredNorm();
    }
    if (d2 < eps2) {
        return;  // coincident; skip per Eq. 2's i ≠ j rule
    }
    const double             d_inv = 1.0 / std::sqrt(d2);
    const Eigen::RowVectorXd delta = (x_j_block.row(jj) - x_i_block.row(ii)) * d_inv;
    u.row(j) += delta;
    u.row(i) -= delta;
}

}  // namespace

RowMajorMatrix spatial_rank(const Sample& x, double eps)
{
    const Eigen::Index N = x.rows();
    const Eigen::Index d = x.cols();
    RowMajorMatrix     u = RowMajorMatrix::Zero(N, d);
    if (N <= 1) {
        return u;
    }
    const double eps2 = eps * eps;

    // Pre-compute row squared-norms s_i = ||x_i||². Used by the GEMM
    // identity to recover pairwise distances without a per-pair subtract.
    Eigen::VectorXd s(N);
    for (Eigen::Index i = 0; i < N; ++i) {
        s(i) = x.row(i).squaredNorm();
    }

    // Tile the upper triangle. The diagonal tiles process pairs within a
    // single block (ii < jj); off-diagonal tiles process every (ii, jj)
    // pair across two distinct blocks.
    for (Eigen::Index a = 0; a < N; a += kTileRows) {
        const Eigen::Index Ta  = std::min(kTileRows, N - a);
        const auto         X_a = x.middleRows(a, Ta);

        // ---- Diagonal tile (block a × block a) ----
        {
            const Eigen::MatrixXd G_aa = X_a * X_a.transpose();
            for (Eigen::Index ii = 0; ii < Ta; ++ii) {
                for (Eigen::Index jj = ii + 1; jj < Ta; ++jj) {
                    accumulate_pair(u, X_a, X_a, ii, jj, a + ii, a + jj,
                                    s(a + ii), s(a + jj), G_aa(ii, jj), eps2);
                }
            }
        }

        // ---- Off-diagonal tiles (block a × block b > a) ----
        for (Eigen::Index b = a + kTileRows; b < N; b += kTileRows) {
            const Eigen::Index    Tb   = std::min(kTileRows, N - b);
            const auto            X_b  = x.middleRows(b, Tb);
            const Eigen::MatrixXd G_ab = X_a * X_b.transpose();
            for (Eigen::Index ii = 0; ii < Ta; ++ii) {
                for (Eigen::Index jj = 0; jj < Tb; ++jj) {
                    accumulate_pair(u, X_a, X_b, ii, jj, a + ii, b + jj,
                                    s(a + ii), s(b + jj), G_ab(ii, jj), eps2);
                }
            }
        }
    }

    u /= static_cast<double>(N);
    return u;
}

namespace {

// Approximate spatial rank: each query row j accumulates contributions from
// an R-sized reference subset of the same cloud. The divisor is R (the
// configured reference size), matching Eq. 1's convention of "divide by
// the sample size" — when j happens to be in Refs, the i == j term is
// skipped but the divisor is unchanged, introducing an O(1/R) bias that
// is dominated by the O(1/√R) variance for any practical R.
RowMajorMatrix spatial_rank_approx_uniform(
    const Sample& x, std::size_t R_in, std::uint64_t seed, double eps)
{
    const Eigen::Index N = x.rows();
    const Eigen::Index d = x.cols();
    RowMajorMatrix     u = RowMajorMatrix::Zero(N, d);
    if (N <= 1) {
        return u;
    }
    const auto indices = detail::uniform_reference_indices(
        static_cast<std::size_t>(N), R_in, seed);
    const auto         R  = static_cast<Eigen::Index>(indices.size());
    if (R <= 0) {
        return u;
    }
    const RowMajorMatrix x_refs = detail::gather_rows(x, indices);
    const double         eps2   = eps * eps;

    // Membership test: which query rows are themselves reference points?
    // Constant lookup avoids an inner branch over the sorted vector.
    std::vector<unsigned char> in_refs(static_cast<std::size_t>(N), 0);
    for (auto idx : indices) {
        in_refs[static_cast<std::size_t>(idx)] = 1;
    }

    // Pre-compute squared norms.
    Eigen::VectorXd s_x(N);
    for (Eigen::Index i = 0; i < N; ++i) {
        s_x(i) = x.row(i).squaredNorm();
    }
    Eigen::VectorXd s_refs(R);
    for (Eigen::Index k = 0; k < R; ++k) {
        s_refs(k) = x_refs.row(k).squaredNorm();
    }

    // Tile over query rows for cache locality. Each tile materializes a
    // (kTileRows × R) Gram block ~= 256 × R doubles, fits in L2 for R up
    // to ~16 K (32 MB).
    for (Eigen::Index a = 0; a < N; a += kTileRows) {
        const Eigen::Index    Ta = std::min(kTileRows, N - a);
        const auto            X_a = x.middleRows(a, Ta);
        const Eigen::MatrixXd G   = X_a * x_refs.transpose();  // Ta × R

        for (Eigen::Index ii = 0; ii < Ta; ++ii) {
            const Eigen::Index j     = a + ii;
            const double       s_j   = s_x(j);
            const bool         skip_self = in_refs[static_cast<std::size_t>(j)] != 0;
            Eigen::RowVectorXd acc = Eigen::RowVectorXd::Zero(d);
            for (Eigen::Index k = 0; k < R; ++k) {
                if (skip_self && indices[static_cast<std::size_t>(k)] == j) {
                    continue;
                }
                double d2 = s_j + s_refs(k) - 2.0 * G(ii, k);
                const double cancel_threshold = 1e-12 * (std::abs(s_j) + std::abs(s_refs(k)));
                if (d2 < cancel_threshold) {
                    Eigen::RowVectorXd diff = X_a.row(ii) - x_refs.row(k);
                    d2                      = diff.squaredNorm();
                }
                if (d2 < eps2) {
                    continue;
                }
                const double d_inv = 1.0 / std::sqrt(d2);
                acc += (X_a.row(ii) - x_refs.row(k)) * d_inv;
            }
            u.row(j) = acc / static_cast<double>(R);
        }
    }
    return u;
}

// K-d tree Barnes-Hut approximation of spatial_rank. For each query row,
// traverse a balanced k-d tree of the full cloud and accumulate exact
// (x_j − x_i)/d contributions for "near" points and centroid-summarized
// contributions for "far" subtrees. Divisor is N (the full cloud size),
// matching Eq. 1's convention.
RowMajorMatrix spatial_rank_kdtree(
    const Sample& x, double opening_theta, std::size_t leaf_size, double eps)
{
    const Eigen::Index N = x.rows();
    const Eigen::Index d = x.cols();
    RowMajorMatrix     u = RowMajorMatrix::Zero(N, d);
    if (N <= 1) {
        return u;
    }

    detail::KdTree tree(x, leaf_size);

    for (Eigen::Index j = 0; j < N; ++j) {
        const Eigen::VectorXd query = x.row(j).transpose();
        Eigen::VectorXd       acc   = Eigen::VectorXd::Zero(d);
        tree.traverse(
            query, opening_theta,
            // exact_fn: open one point.
            [&](Eigen::Index idx) {
                if (idx == j) {
                    return;  // skip self
                }
                const Eigen::VectorXd diff = query - x.row(idx).transpose();
                const double          dist = diff.norm();
                if (dist < eps) {
                    return;
                }
                acc += diff / dist;
            },
            // approx_fn: summarize subtree by centroid + count.
            [&](const Eigen::VectorXd& centroid, std::size_t count) {
                const Eigen::VectorXd diff = query - centroid;
                const double          dist = diff.norm();
                if (dist < eps) {
                    return;
                }
                acc += static_cast<double>(count) * diff / dist;
            });
        u.row(j) = (acc / static_cast<double>(N)).transpose();
    }
    return u;
}

}  // namespace

RowMajorMatrix spatial_rank(const Sample& x, const SamplingConfig& sampling, double eps)
{
    const auto N = static_cast<std::size_t>(x.rows());
    switch (sampling.strategy) {
        case SamplingConfig::Strategy::Uniform:
            if (sampling.reference_size == 0 || sampling.reference_size >= N) {
                return spatial_rank(x, eps);
            }
            return spatial_rank_approx_uniform(
                x, sampling.reference_size, sampling.seed, eps);
        case SamplingConfig::Strategy::KdTreeLocalExact:
            return spatial_rank_kdtree(
                x, sampling.kd_opening_theta, sampling.kd_leaf_size, eps);
        case SamplingConfig::Strategy::Stratified:
        default:
            throw std::invalid_argument(
                "spatial_rank: SamplingConfig::Strategy::Stratified is reserved "
                "for a future tranche; use Uniform or KdTreeLocalExact");
    }
}

InverseRankResult inverse_spatial_rank(
    const RowMajorMatrix& u, const Sample& y, const SolverConfig& solver)
{
    if (u.cols() != y.cols()) {
        throw std::invalid_argument(
            "inverse_spatial_rank: u.cols() (" + std::to_string(u.cols())
            + ") must equal y.cols() (" + std::to_string(y.cols()) + ")");
    }

    InverseRankResult result;
    result.x_tilde = RowMajorMatrix::Zero(u.rows(), u.cols());

    for (Eigen::Index j = 0; j < u.rows(); ++j) {
        Eigen::VectorXd u_j        = u.row(j).transpose();
        auto            row_result = detail::solve_inverse_rank_row(u_j, y, solver);

        if (!row_result.converged) {
            throw std::runtime_error(
                "inverse_spatial_rank: row " + std::to_string(j)
                + " failed to converge after " + std::to_string(row_result.iters)
                + " iterations (residual=" + std::to_string(row_result.residual) + ")");
        }

        result.x_tilde.row(j) = row_result.x.transpose();
        result.max_iters_used = std::max(result.max_iters_used, row_result.iters);
        result.max_residual   = std::max(result.max_residual, row_result.residual);
        result.aa_fallbacks += row_result.aa_fallbacks;
        result.aa_restarts  += row_result.aa_restarts;
    }
    return result;
}

InverseRankResult inverse_spatial_rank(
    const RowMajorMatrix& u, const Sample& y, const SolverConfig& solver,
    const SamplingConfig& sampling)
{
    const auto M = static_cast<std::size_t>(y.rows());
    if (sampling.reference_size == 0 || sampling.reference_size >= M) {
        return inverse_spatial_rank(u, y, solver);
    }
    switch (sampling.strategy) {
        case SamplingConfig::Strategy::Uniform:
        case SamplingConfig::Strategy::KdTreeLocalExact: {
            // KdTreeLocalExact's tree-aware Weiszfeld variant would require
            // refactoring the solver API to take a cloud accessor instead
            // of a raw y matrix — scoped out of tranche (o). For now the
            // strategy falls through to the uniform-subsample path so
            // similarity_score with KdTreeLocalExact still works
            // (spatial_rank uses the tree; inverse_spatial_rank uses
            // uniform R-subset of y).
            auto indices
                = detail::uniform_reference_indices(M, sampling.reference_size, sampling.seed);
            Sample y_refs = detail::gather_rows(y, indices);
            return inverse_spatial_rank(u, y_refs, solver);
        }
        case SamplingConfig::Strategy::Stratified:
        default:
            throw std::invalid_argument(
                "inverse_spatial_rank: SamplingConfig::Strategy::Stratified "
                "is reserved for a future tranche; use Uniform or "
                "KdTreeLocalExact");
    }
}

InverseRankResult inverse_spatial_rank(
    const RowMajorMatrix& u, const Sample& y, double tol, std::size_t max_iters)
{
    SolverConfig cfg;
    cfg.kind      = SolverKind::Weiszfeld;
    cfg.tol       = tol;
    cfg.max_iters = max_iters;
    return inverse_spatial_rank(u, y, cfg);
}

}  // namespace mqces
