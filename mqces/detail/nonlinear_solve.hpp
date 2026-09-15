#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

// Weiszfeld-style fixed-point iteration for the inverse-spatial-rank
// equation (Eq. 2 of Weber & Dayman):
//
//     u = (1/M) Σ_i (x − y_i) / ||x − y_i||
//
// Rearrange:
//
//     x = (M u + Σ_i y_i / d_i) / Σ_i 1/d_i,    d_i = ||x − y_i||
//
// This is a straightforward fixed-point map; convergence is geometric for
// generic configurations (and very fast for the small d ≤ 10, M ~ 50 cases
// the QC method actually exercises).
//
// TODO: Plain Weiszfeld plateaus around residual ~1e-6 to ~1e-7 when the
// iterate sits near (but not on) a y_i — a known failure mode. Anderson
// acceleration or the Vardi-Zhang regularization would close that gap.
// similarity_score loosens the tolerance to 1e-5 to work around it.
//
// Convergence is also slow (thousands of iterations) when the solution lies
// between well-separated clusters, and the step-length criterion is in data
// units, so iteration counts grow with the data's scale. A damped Newton
// iteration on f(x) = mean_i ||x − y_i|| − uᵀx (d × d Hessian, d ≤ 10)
// converges in tens of iterations there, but needs Vardi-Zhang-style
// handling of solutions that coincide with a y_i before it can replace this.

namespace mqces::detail {

struct WeiszfeldResult {
  Eigen::VectorXd x;
  std::size_t iters = 0;
  double residual = 0.0;
  bool converged = false;
};

// Solve for x given the target rank `u` and the cloud `y` (rows are points).
// `eps` guards the degenerate case where the iterate coincides with a y_i.
inline WeiszfeldResult solve_inverse_rank_row(
  const Eigen::Ref<const Eigen::VectorXd>& u,
  const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& y,
  double tol,
  std::size_t max_iters,
  double eps = 1e-12) {
  const auto m = static_cast<double>(y.rows());

  // Initial guess: mean of y, shifted by u to break the symmetric stationary
  // point (mean(y) is an exact solution only when u == 0).
  Eigen::VectorXd x = y.colwise().mean().transpose() + u;

  WeiszfeldResult r;
  Eigen::VectorXd numerator(x.size());
  for (std::size_t k = 0; k < max_iters; ++k) {
    numerator.noalias() = m * u;
    double inv_d_sum = 0.0;

    // Skipping y_i with d < eps is exactly the i ≠ j exclusion of Eq. 2:
    // a coincident y_i contributes an undefined unit vector and must be
    // omitted regardless of whether the iterate transiently lands on it.
    for (Eigen::Index i = 0; i < y.rows(); ++i) {
      const double d = (x - y.row(i).transpose()).norm();
      if (d < eps) {
        continue;
      }
      numerator.noalias() += y.row(i).transpose() / d;
      inv_d_sum += 1.0 / d;
    }

    if (inv_d_sum <= 0.0) {
      // All points coincided with x — pathological; bail.
      r.x = x;
      r.iters = k;
      r.residual = std::numeric_limits<double>::infinity();
      r.converged = false;
      return r;
    }

    Eigen::VectorXd x_next = numerator / inv_d_sum;
    double step_len = (x_next - x).norm();
    x = std::move(x_next);
    r.iters = k + 1;
    r.residual = step_len;

    if (step_len < tol) {
      r.x = std::move(x);
      r.converged = true;
      return r;
    }
  }

  r.x = std::move(x);
  r.converged = false;
  return r;
}

}  // namespace mqces::detail
