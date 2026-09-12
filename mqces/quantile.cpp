#include <mqces/quantile.hpp>

#include <mqces/detail/nonlinear_solve.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mqces {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

RowMajorMatrix spatial_rank(const Sample& x, double eps)
{
    const Eigen::Index n = x.rows();
    const Eigen::Index d = x.cols();
    RowMajorMatrix     u = RowMajorMatrix::Zero(n, d);
    if (n <= 1) {
        return u;
    }
    const double inv_n = 1.0 / static_cast<double>(n);

    // Scratch vectors are allocated once; assigning an expression of the
    // same size into them does not reallocate inside the O(N^2) loop.
    Eigen::VectorXd acc(d);
    Eigen::VectorXd diff(d);
    for (Eigen::Index j = 0; j < n; ++j) {
        acc.setZero();
        for (Eigen::Index i = 0; i < n; ++i) {
            if (i == j) {
                continue;
            }
            diff.noalias() = x.row(j).transpose() - x.row(i).transpose();
            const double norm = diff.norm();
            if (norm < eps) {
                continue;
            }
            acc.noalias() += diff / norm;
        }
        u.row(j) = (inv_n * acc).transpose();
    }
    return u;
}

InverseRankResult inverse_spatial_rank(
    const RowMajorMatrix& u, const Sample& y, double tol, std::size_t max_iters)
{
    if (u.cols() != y.cols()) {
        throw std::invalid_argument(
            "inverse_spatial_rank: u.cols() (" + std::to_string(u.cols())
            + ") must equal y.cols() (" + std::to_string(y.cols()) + ")");
    }

    InverseRankResult result;
    result.x_tilde = RowMajorMatrix::Zero(u.rows(), u.cols());

    for (Eigen::Index j = 0; j < u.rows(); ++j) {
        Eigen::VectorXd u_j = u.row(j).transpose();
        auto row_result     = detail::solve_inverse_rank_row(u_j, y, tol, max_iters);

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

}  // namespace mqces
