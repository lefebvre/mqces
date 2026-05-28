#include <mqces/quantile.hpp>

#include <mqces/detail/inverse_solvers.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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
    }
    return result;
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
