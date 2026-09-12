#pragma once

#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>
#include <mqces/uncertainty.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Machinery shared by classic::classify and v2::classify.
//
// Uncertainty model. The score of the test sample against a class varies
// from one draw of specimens to the next by far more than it varies under
// measurement noise with the specimens held fixed, so the standard error of
// a score is built from two terms:
//
//   * sampling — delete-one jackknife over the specimens of both samples:
//       var = (n − 1)/n · Σ_i (S_(−i) − S̄_(−))²,
//     summed over the test rows and the class rows. The jackknife was chosen
//     over the bootstrap because resampling with replacement duplicates
//     rows, which distorts spatial ranks and inflates the score itself.
//
//   * measurement — when epsilon > 0, the sample variance of the score over
//     mc_samples independent perturbations of both samples (Eq. 10).
//
// Each term estimates the variance of a single score, not of a replicate
// mean, so neither is divided by a replicate count.

namespace mqces::detail {

inline void validate_classify_inputs(const Sample& test,
                                     std::span<const Class> known,
                                     const ClassifierOptions& opts) {
  if (known.empty()) {
    throw std::invalid_argument("classify: known classes list is empty");
  }
  if (test.rows() < 2) {
    throw std::invalid_argument("classify: test sample needs at least 2 specimens");
  }
  const Eigen::Index d = test.cols();
  if (opts.weights.size() != d) {
    throw std::invalid_argument("classify: weights.size() (" + std::to_string(opts.weights.size()) +
                                ") must equal test.cols() (" + std::to_string(d) + ")");
  }
  for (const auto& cls : known) {
    if (cls.specimens.cols() != d) {
      throw std::invalid_argument("classify: class '" + cls.name + "' has feature count " +
                                  std::to_string(cls.specimens.cols()) + " but test has " +
                                  std::to_string(d));
    }
    if (cls.specimens.rows() < 2) {
      throw std::invalid_argument("classify: class '" + cls.name + "' needs at least 2 specimens");
    }
  }
  const double eps = opts.uncertainty.epsilon;
  if (!std::isfinite(eps) || eps < 0.0) {
    throw std::invalid_argument("classify: uncertainty.epsilon must be finite and >= 0");
  }
  if (opts.uncertainty.mc_samples < 2) {
    throw std::invalid_argument("classify: uncertainty.mc_samples must be >= 2");
  }
  const double alpha = opts.nota_threshold;
  if (!std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) {
    throw std::invalid_argument("classify: nota_threshold must lie in (0, 1)");
  }
  if (opts.nota_permutations > 0 && static_cast<double>(opts.nota_permutations + 1) * alpha < 1.0) {
    throw std::invalid_argument(
      "classify: nota_permutations (" + std::to_string(opts.nota_permutations) +
      ") is too small for the smallest p-value to fall below nota_threshold");
  }
  if (opts.n_threads < 0) {
    throw std::invalid_argument("classify: n_threads must be >= 0");
  }
}

// Derive a deterministic seed for replicate i and class k. The test sample
// uses k = static_cast<std::size_t>(-1) (i.e. SIZE_MAX) as a sentinel.
inline std::uint64_t replicate_seed(std::uint64_t base,
                                    std::size_t replicate,
                                    std::size_t class_index) {
  // splitmix64 of (base ^ replicate_mix ^ class_mix); each component is
  // independently splitmix'd so the bits are well-decorrelated.
  return splitmix64(base ^
                    splitmix64(static_cast<std::uint64_t>(replicate) * 0x9E3779B97F4A7C15ULL) ^
                    splitmix64(static_cast<std::uint64_t>(class_index) + 1));
}

// Score of the test sample against every class, on the observed data.
inline Eigen::VectorXd observed_scores(const Sample& test,
                                       std::span<const Class> known,
                                       const ClassifierOptions& opts) {
  Eigen::VectorXd scores(static_cast<Eigen::Index>(known.size()));
  parallel_for(known.size(), opts.n_threads, [&](std::size_t k) {
    scores(static_cast<Eigen::Index>(k)) = similarity_score(test, known[k].specimens, opts.weights);
  });
  return scores;
}

inline Sample drop_row(const Sample& x, Eigen::Index row) {
  Sample out(x.rows() - 1, x.cols());
  out.topRows(row) = x.topRows(row);
  out.bottomRows(x.rows() - row - 1) = x.bottomRows(x.rows() - row - 1);
  return out;
}

// Replicate scores of the test sample against one class.
struct Replicates {
  Eigen::VectorXd drop_test;   // [i]: test row i left out
  Eigen::VectorXd drop_class;  // [j]: class row j left out
  Eigen::VectorXd perturbed;   // [r]: both samples perturbed; empty when epsilon == 0
};

// Replicates for class `class_index`. Perturbation replicate r of the test
// sample uses the same seed for every class, so perturbed scores of two
// classes are paired replicate by replicate.
inline Replicates collect_replicates(const Sample& test,
                                     const Class& cls,
                                     std::size_t class_index,
                                     const ClassifierOptions& opts) {
  const Eigen::Index n_test = test.rows();
  const Eigen::Index n_class = cls.specimens.rows();
  const double eps = opts.uncertainty.epsilon;
  const Eigen::Index n_mc = eps > 0.0 ? static_cast<Eigen::Index>(opts.uncertainty.mc_samples) : 0;

  Replicates rep;
  rep.drop_test.resize(n_test);
  rep.drop_class.resize(n_class);
  rep.perturbed.resize(n_mc);

  const Eigen::Index n_tasks = n_test + n_class + n_mc;
  parallel_for(static_cast<std::size_t>(n_tasks), opts.n_threads, [&](std::size_t task) {
    const auto t = static_cast<Eigen::Index>(task);
    if (t < n_test) {
      rep.drop_test(t) = similarity_score(drop_row(test, t), cls.specimens, opts.weights);
    } else if (t < n_test + n_class) {
      const Eigen::Index j = t - n_test;
      rep.drop_class(j) = similarity_score(test, drop_row(cls.specimens, j), opts.weights);
    } else {
      const Eigen::Index r = t - n_test - n_class;
      const auto ri = static_cast<std::size_t>(r);
      const auto base = opts.uncertainty.seed;
      rep.perturbed(r) =
        similarity_score(perturb(test, eps, replicate_seed(base, ri, static_cast<std::size_t>(-1))),
                         perturb(cls.specimens, eps, replicate_seed(base, ri, class_index)),
                         opts.weights);
    }
  });
  return rep;
}

// Delete-one jackknife variance: (n − 1)/n · Σ (v_i − v̄)².
inline double jackknife_variance(const Eigen::Ref<const Eigen::VectorXd>& v) {
  const auto n = static_cast<double>(v.size());
  if (v.size() < 2) {
    return 0.0;
  }
  return (n - 1.0) / n * (v.array() - v.mean()).square().sum();
}

// Bessel-corrected sample variance; 0 for fewer than two values.
inline double sample_variance(const Eigen::Ref<const Eigen::VectorXd>& v) {
  if (v.size() < 2) {
    return 0.0;
  }
  return (v.array() - v.mean()).square().sum() / static_cast<double>(v.size() - 1);
}

// Degrees of freedom for the t reference distribution: one less than the
// smallest sample entering the comparison.
inline double comparison_df(const Sample& test, const Class& a, const Class& b) {
  const Eigen::Index n = std::min({test.rows(), a.specimens.rows(), b.specimens.rows()});
  return static_cast<double>(n - 1);
}

struct Ranking {
  ClassificationResult result;  // best_class and sorted scores filled in
  Eigen::VectorXd observed;     // observed score per class, in input order
  std::size_t best = 0;
  std::size_t runner_up = 0;  // == best when there is a single class
};

// Score every class on the observed data and rank ascending.
inline Ranking rank_classes(const Sample& test,
                            std::span<const Class> known,
                            const ClassifierOptions& opts) {
  Ranking r;
  r.observed = observed_scores(test, known, opts);

  std::vector<std::pair<double, std::size_t>> order;
  order.reserve(known.size());
  for (std::size_t k = 0; k < known.size(); ++k) {
    order.emplace_back(r.observed(static_cast<Eigen::Index>(k)), k);
  }
  std::sort(order.begin(), order.end());

  r.result.scores.reserve(known.size());
  for (const auto& [s, k] : order) {
    r.result.scores.push_back({known[k].name, s});
  }
  r.best = order.front().second;
  r.runner_up = order.size() > 1 ? order[1].second : r.best;
  r.result.best_class = known[r.best].name;
  return r;
}

}  // namespace mqces::detail
