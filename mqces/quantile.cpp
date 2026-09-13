#include <mqces/detail/nonlinear_solve.hpp>
#include <mqces/detail/parallel.hpp>
#include <mqces/quantile.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mqces {

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

RowMajorMatrix spatial_rank(const Sample& x, double eps) {
  const Eigen::Index n = x.rows();
  const Eigen::Index d = x.cols();
  RowMajorMatrix u = RowMajorMatrix::Zero(n, d);
  if (n <= 1) {
    return u;
  }
  const double inv_n = 1.0 / static_cast<double>(n);

  // Rows are independent, so they run in parallel; each writes only its own
  // row of u. Scratch vectors are allocated once per row; assigning an
  // expression of the same size into them does not reallocate inside the
  // inner O(N) loop.
  detail::parallel_for(static_cast<std::size_t>(n), [&](std::size_t row) {
    const auto j = static_cast<Eigen::Index>(row);
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
    Eigen::VectorXd diff(d);
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
  });
  return u;
}

InverseRankResult inverse_spatial_rank(const RowMajorMatrix& u,
                                       const Sample& y,
                                       double tol,
                                       std::size_t max_iters) {
  if (u.cols() != y.cols()) {
    throw std::invalid_argument("inverse_spatial_rank: u.cols() (" + std::to_string(u.cols()) +
                                ") must equal y.cols() (" + std::to_string(y.cols()) + ")");
  }

  const Eigen::Index rows = u.rows();
  InverseRankResult result;
  result.x_tilde = RowMajorMatrix::Zero(rows, u.cols());

  // Rows are solved independently and in parallel. Per-row iteration counts
  // and residuals are kept so the summary is reduced after the loop instead
  // of racing on shared maxima. If several rows fail, parallel_for rethrows
  // the first failure it recorded, which is not necessarily the lowest row.
  std::vector<std::size_t> iters(static_cast<std::size_t>(rows), 0);
  std::vector<double> residuals(static_cast<std::size_t>(rows), 0.0);
  detail::parallel_for(static_cast<std::size_t>(rows), [&](std::size_t row) {
    const auto j = static_cast<Eigen::Index>(row);
    auto row_result = detail::solve_inverse_rank_row(u.row(j).transpose(), y, tol, max_iters);

    if (!row_result.converged) {
      throw std::runtime_error("inverse_spatial_rank: row " + std::to_string(j) +
                               " failed to converge after " + std::to_string(row_result.iters) +
                               " iterations (residual=" + std::to_string(row_result.residual) +
                               ")");
    }

    result.x_tilde.row(j) = row_result.x.transpose();
    iters[row] = row_result.iters;
    residuals[row] = row_result.residual;
  });

  if (rows > 0) {
    result.max_iters_used = *std::max_element(iters.begin(), iters.end());
    result.max_residual = *std::max_element(residuals.begin(), residuals.end());
  }
  return result;
}

}  // namespace mqces
