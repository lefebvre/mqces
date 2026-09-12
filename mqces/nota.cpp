#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>
#include <mqces/nota.hpp>
#include <mqces/score.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mqces {

namespace {

// Keeps NOTA's permutation streams disjoint from the classifier's
// perturbation draws even when both derive from the same user seed.
constexpr std::uint64_t kNotaPermSalt = 0x12345678ABCDEF01ULL;

}  // namespace

bool is_none_of_the_above(const Sample& test,
                          const Class& best,
                          double observed_score,
                          const ClassifierOptions& options) {
  const std::size_t n_perm = options.nota_permutations;
  if (n_perm == 0) {
    return false;
  }
  if (test.cols() != best.specimens.cols()) {
    throw std::invalid_argument("is_none_of_the_above: test and class '" + best.name +
                                "' have different feature counts");
  }

  const Eigen::Index n_test = test.rows();
  const Eigen::Index n_pool = n_test + best.specimens.rows();
  const Eigen::Index d = test.cols();

  Sample pooled(n_pool, d);
  pooled.topRows(n_test) = test;
  pooled.bottomRows(best.specimens.rows()) = best.specimens;

  std::vector<char> at_least_observed(n_perm, 0);
  detail::parallel_for(n_perm, options.n_threads, [&](std::size_t i) {
    auto rng = detail::make_stream(options.uncertainty.seed ^ kNotaPermSalt, i);

    std::vector<Eigen::Index> perm(static_cast<std::size_t>(n_pool));
    std::iota(perm.begin(), perm.end(), Eigen::Index{0});
    detail::shuffle(perm.begin(), perm.size(), rng);

    Sample group_test(n_test, d);
    Sample group_class(n_pool - n_test, d);
    for (Eigen::Index j = 0; j < n_pool; ++j) {
      const auto src = pooled.row(perm[static_cast<std::size_t>(j)]);
      if (j < n_test) {
        group_test.row(j) = src;
      } else {
        group_class.row(j - n_test) = src;
      }
    }
    if (similarity_score(group_test, group_class, options.weights) >= observed_score) {
      at_least_observed[i] = 1;
    }
  });

  const auto extreme = std::count(at_least_observed.begin(), at_least_observed.end(), char{1});
  const double p_value = static_cast<double>(1 + extreme) / static_cast<double>(1 + n_perm);
  return p_value < options.nota_threshold;
}

}  // namespace mqces
