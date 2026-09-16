#pragma once

#include <mqces/types.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

// Solvers for the inverse-spatial-rank equation (Eq. 2 of Weber & Dayman):
//
//     u = (1/M) Σ_i (x − y_i) / ||x − y_i||
//
// Rearranged as a fixed point:
//
//     x = G(x) = (M u + Σ_i y_i / d_i) / Σ_i 1/d_i,    d_i = ||x − y_i||
//
// G is the Weiszfeld map. Its fixed points minimize the convex objective
//
//     f(x) = Σ_i ||x − y_i|| − M uᵀx
//
// and every Weiszfeld step lowers f, which is why plain iteration converges
// from any start. Convergence is linear, though, and slows to thousands of
// iterations when the solution lies in the gap between well-separated
// clusters.
//
// Both solvers stop when the fixed-point residual ||G(x) − x|| falls below
// `tol` and return G(x). That does not make their answers interchangeable:
//
//   * When the points of y are collinear (all on one line, including a single
//     point or all-identical points) the sum of distances is not strictly
//     convex along that line and the minimizer can be a whole segment. Which
//     point an iteration reaches depends on its start and path, and no
//     tolerance changes that. solver_for_cloud() therefore sends collinear
//     clouds to plain Weiszfeld whichever solver was requested.
//
//   * When y is close to collinear the minimizer is unique but the objective
//     is nearly flat along the line, so a small residual still allows a large
//     distance from the minimizer. Plain Weiszfeld, being slower, tends to
//     stop further from it at a given tolerance than Anderson does, so the
//     two can give materially different scores.
//
// Plain Weiszfeld plateaus around residual ~1e-6 to ~1e-7 when the iterate
// sits near (but not on) a y_i; similarity_score uses a 1e-5 tolerance
// partly for that reason.

namespace mqces::detail {

struct WeiszfeldResult {
  Eigen::VectorXd x;
  std::size_t iters = 0;
  double residual = 0.0;
  bool converged = false;
};

using RowMajorCloud = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// One sweep over the cloud at point p: the Weiszfeld map G(p) and the
// objective f(p) share the same distances. Returns false when every y_i
// coincides with p (within eps), where G is undefined.
//
// Skipping y_i with d < eps is exactly the i ≠ j exclusion of Eq. 2: a
// coincident y_i contributes an undefined unit vector and must be omitted
// regardless of whether the iterate transiently lands on it.
inline bool weiszfeld_eval(const Eigen::Ref<const Eigen::VectorXd>& u,
                           const Eigen::Ref<const RowMajorCloud>& y,
                           const Eigen::Ref<const Eigen::VectorXd>& p,
                           double eps,
                           Eigen::VectorXd& map,
                           double& objective) {
  const auto m = static_cast<double>(y.rows());
  map.noalias() = m * u;
  double inv_d_sum = 0.0;
  double d_sum = 0.0;
  for (Eigen::Index i = 0; i < y.rows(); ++i) {
    const double d = (p - y.row(i).transpose()).norm();
    d_sum += d;
    if (d < eps) {
      continue;
    }
    map.noalias() += y.row(i).transpose() / d;
    inv_d_sum += 1.0 / d;
  }
  objective = d_sum - m * u.dot(p);
  if (inv_d_sum <= 0.0) {
    return false;
  }
  map /= inv_d_sum;
  return true;
}

// Plain Weiszfeld iteration from x0 = mean(y) + u.
inline WeiszfeldResult solve_inverse_rank_row(const Eigen::Ref<const Eigen::VectorXd>& u,
                                              const Eigen::Ref<const RowMajorCloud>& y,
                                              double tol,
                                              std::size_t max_iters,
                                              double eps = 1e-12) {
  // Shifting the centroid by u breaks the symmetric stationary point
  // (mean(y) is an exact solution only when u == 0).
  Eigen::VectorXd x = y.colwise().mean().transpose() + u;

  WeiszfeldResult r;
  Eigen::VectorXd x_next(x.size());
  double objective = 0.0;
  for (std::size_t k = 0; k < max_iters; ++k) {
    if (!weiszfeld_eval(u, y, x, eps, x_next, objective)) {
      // All points coincided with x — pathological; bail.
      r.x = x;
      r.iters = k;
      r.residual = std::numeric_limits<double>::infinity();
      return r;
    }
    const double step_len = (x_next - x).norm();
    x.swap(x_next);
    r.iters = k + 1;
    r.residual = step_len;
    if (step_len < tol) {
      r.x = std::move(x);
      r.converged = true;
      return r;
    }
  }
  r.x = std::move(x);
  return r;
}

// True when the rows of y lie on one line to within what floating-point
// distances can resolve. A point offset by b from a line changes its
// distance a from a point on that line only by about b²/(2a), so once the
// cloud's extent across its best-fit line falls below √ε times its extent
// along it, the objective cannot tell the cloud from a collinear one. The
// test compares the two largest singular values of the centered cloud; a
// single point, identical points and any one-feature cloud all pass it.
inline bool cloud_is_collinear(const Eigen::Ref<const RowMajorCloud>& y) {
  if (y.rows() <= 2 || y.cols() <= 1) {
    return true;
  }
  const Eigen::MatrixXd centered = y.rowwise() - y.colwise().mean();
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered);
  const auto& sigma = svd.singularValues();
  return sigma(1) <= std::sqrt(std::numeric_limits<double>::epsilon()) * sigma(0);
}

// The solver actually used for cloud y: the requested one, except that a
// collinear cloud always uses plain Weiszfeld so that the choice of solver
// cannot change which of its non-unique solutions is selected.
inline SolverConfig solver_for_cloud(const Eigen::Ref<const RowMajorCloud>& y,
                                     const SolverConfig& requested) {
  if (requested.kind == InverseRankSolver::Anderson && cloud_is_collinear(y)) {
    return SolverConfig{InverseRankSolver::Weiszfeld, requested.anderson_window};
  }
  return requested;
}

// Weiszfeld accelerated by Anderson mixing (Type II, Walker & Ni 2011) over
// the last `window` iterates.
//
// Each outer iteration extrapolates from the history of G-values to a
// candidate point, and accepts it only if it lowers the objective below its
// value at the current iterate; otherwise it takes the plain Weiszfeld step,
// which lowers the objective by itself. The objective therefore strictly
// decreases every iteration, so the iteration cannot cycle. A plain residual
// safeguard (accept unless the residual grows by some factor) does cycle: an
// early extrapolation can land in a distant cluster and alternate there
// between rejected and accepted steps indefinitely. The history is kept
// across rejections; clearing it slows convergence several-fold.
//
// The start is x0 = mean(y) + s·u, with s the RMS distance of y from its
// centroid. u is dimensionless, so plain Weiszfeld's mean(y) + u starts
// far from the solution when the data's scale is small; with s the iteration
// is scale-invariant: scaling y by c scales every iterate by c.
//
// `iters` counts outer iterations. Each costs one or two sweeps over the
// cloud (one when the extrapolated point is accepted).
//
// Meant for clouds that are not collinear (see solver_for_cloud). A cloud
// with zero spread is handed to plain Weiszfeld here as well: its start would
// coincide with every point, where the Weiszfeld map is undefined.
inline WeiszfeldResult solve_inverse_rank_row_anderson(const Eigen::Ref<const Eigen::VectorXd>& u,
                                                       const Eigen::Ref<const RowMajorCloud>& y,
                                                       double tol,
                                                       std::size_t max_iters,
                                                       int window,
                                                       double eps = 1e-12) {
  const Eigen::Index d = y.cols();
  const Eigen::Index m = std::max(2, window);
  // Relative Tikhonov regularization of the least-squares mixing problem.
  constexpr double kMixingRegularization = 1e-12;

  const Eigen::RowVectorXd centroid = y.colwise().mean();
  const double spread =
    std::sqrt((y.rowwise() - centroid).squaredNorm() / static_cast<double>(y.rows()));
  if (spread == 0.0) {
    return solve_inverse_rank_row(u, y, tol, max_iters, eps);
  }

  WeiszfeldResult r;
  Eigen::VectorXd x = centroid.transpose() + spread * u;
  Eigen::VectorXd gx(d);
  double fx = 0.0;
  if (!weiszfeld_eval(u, y, x, eps, gx, fx)) {
    r.x = x;
    r.residual = std::numeric_limits<double>::infinity();
    return r;
  }

  // Columns hold the most recent iterates and their G-values, oldest first.
  Eigen::MatrixXd x_hist(d, m);
  Eigen::MatrixXd g_hist(d, m);
  Eigen::Index count = 0;
  const auto push = [&](const Eigen::VectorXd& xi, const Eigen::VectorXd& gi) {
    if (count == m) {
      x_hist.leftCols(m - 1) = x_hist.rightCols(m - 1).eval();
      g_hist.leftCols(m - 1) = g_hist.rightCols(m - 1).eval();
      --count;
    }
    x_hist.col(count) = xi;
    g_hist.col(count) = gi;
    ++count;
  };
  push(x, gx);

  Eigen::VectorXd candidate(d);
  Eigen::VectorXd g_candidate(d);
  double f_candidate = 0.0;
  for (std::size_t k = 0; k < max_iters; ++k) {
    const Eigen::VectorXd residual = gx - x;
    r.residual = residual.norm();
    if (r.residual < tol) {
      r.x = std::move(gx);
      r.iters = k;
      r.converged = true;
      return r;
    }

    bool accepted = false;
    if (count >= 2) {
      const Eigen::Index cols = count - 1;
      const Eigen::MatrixXd f_hist = g_hist.leftCols(count) - x_hist.leftCols(count);
      const Eigen::MatrixXd d_f = f_hist.rightCols(cols) - f_hist.leftCols(cols);
      const Eigen::MatrixXd d_g = g_hist.middleCols(1, cols) - g_hist.leftCols(cols);
      Eigen::MatrixXd normal = d_f.transpose() * d_f;
      normal.diagonal().array() +=
        kMixingRegularization * normal.trace() / static_cast<double>(cols);
      const Eigen::VectorXd gamma = normal.llt().solve(d_f.transpose() * residual);
      candidate.noalias() = gx - d_g * gamma;
      accepted = candidate.allFinite() &&
                 weiszfeld_eval(u, y, candidate, eps, g_candidate, f_candidate) && f_candidate < fx;
    }
    if (accepted) {
      x = candidate;
      gx = g_candidate;
      fx = f_candidate;
    } else {
      x = gx;
      if (!weiszfeld_eval(u, y, x, eps, gx, fx)) {
        r.x = x;
        r.iters = k + 1;
        r.residual = std::numeric_limits<double>::infinity();
        return r;
      }
    }
    push(x, gx);
    r.iters = k + 1;
  }
  r.residual = (gx - x).norm();
  r.x = std::move(x);
  return r;
}

// Dispatch on the configured solver, as given. inverse_spatial_rank applies
// solver_for_cloud() once per cloud before calling this for each row.
inline WeiszfeldResult solve_inverse_rank_row(const Eigen::Ref<const Eigen::VectorXd>& u,
                                              const Eigen::Ref<const RowMajorCloud>& y,
                                              double tol,
                                              std::size_t max_iters,
                                              const SolverConfig& solver,
                                              double eps = 1e-12) {
  switch (solver.kind) {
    case InverseRankSolver::Anderson:
      return solve_inverse_rank_row_anderson(u, y, tol, max_iters, solver.anderson_window, eps);
    case InverseRankSolver::Weiszfeld:
      break;
  }
  return solve_inverse_rank_row(u, y, tol, max_iters, eps);
}

}  // namespace mqces::detail
