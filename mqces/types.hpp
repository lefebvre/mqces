#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mqces {

// A sample is an (n_specimens × n_features) row-major matrix. Rows are
// individual specimens drawn from the underlying distribution; columns are
// features (e.g. isotope ratios). Row-major lets us iterate one specimen at
// a time without strided loads.
using Sample = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Diagonal of W in Eq. 3 — one weight per feature.
using FeatureWeights = Eigen::VectorXd;

struct Class {
  std::string name;
  Sample specimens;
};

struct Score {
  std::string class_name;
  double s_xy;  // Eq. 3, evaluated on the observed (unperturbed) samples
};

struct ClassificationResult {
  std::string best_class;
  std::vector<Score> scores;            // sorted ascending; scores.front().class_name == best_class
  double misclassification_prob = 0.0;  // Eq. 7 vs the runner-up
  bool none_of_the_above = false;
};

// Controls the measurement-error term of the misclassification probability.
// Specimen sampling variability is always estimated (by leave-one-out
// jackknife) and needs no configuration.
struct UncertaintyConfig {
  double epsilon = 0.0;              // measurement error fraction (Eq. 10); 0 disables the term
  std::size_t mc_samples = 10;       // perturbation replicates, >= 2; used only when epsilon > 0
  std::uint64_t seed = 0xC0FFEEULL;  // deterministic; per-replicate streams derived
};

// Iteration used to solve the inverse-spatial-rank equation (Eq. 2) inside
// every similarity score. On well-spread data both give the same scores to
// well within the score's tolerance. A cloud whose points are collinear (for
// example any one-feature sample, or a jackknifed sample of two specimens)
// can have many solutions, so it always uses Weiszfeld. On nearly collinear
// data the solution is unique but poorly determined at the score's tolerance,
// and the two solvers can give different scores. See
// detail/nonlinear_solve.hpp.
enum class InverseRankSolver : std::uint8_t {
  Weiszfeld,  // plain fixed-point iteration, as in the paper
  Anderson,   // Weiszfeld with Anderson acceleration; far fewer iterations
};

// Solver names as spelled by the CLI and the Python bindings.
inline std::optional<InverseRankSolver> inverse_rank_solver_from_name(std::string_view name) {
  if (name == "weiszfeld") {
    return InverseRankSolver::Weiszfeld;
  }
  if (name == "anderson") {
    return InverseRankSolver::Anderson;
  }
  return std::nullopt;
}

struct SolverConfig {
  InverseRankSolver kind = InverseRankSolver::Weiszfeld;
  int anderson_window = 5;  // iterates mixed per step, >= 2; used only by Anderson
};

struct ClassifierOptions {
  FeatureWeights weights;  // diag(W); required, no default — see Eq. 4
  UncertaintyConfig uncertainty{};
  SolverConfig solver{};
  double nota_threshold = 0.05;  // p-value in (0, 1)
  // Random splits drawn by the none-of-the-above permutation test; 0 skips
  // the test. The smallest attainable p-value is 1 / (nota_permutations + 1),
  // and NOTA fires only when p < nota_threshold, so this must exceed
  // 1 / nota_threshold - 1.
  std::size_t nota_permutations = 199;
  int n_threads = 0;  // 0 == auto (OpenMP runtime default)
};

}  // namespace mqces
