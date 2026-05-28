#include <mqces/nota.hpp>

#include <mqces/score.hpp>
#include <mqces/uncertainty.hpp>

#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>
#include <mqces/detail/stats.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace mqces {

namespace {

// Offset constants used to keep NOTA's RNG streams disjoint from the main
// classifier's MC draws (which use replicate_seed(base, i, class_index)).
// XOR-mixing with these masks guarantees that NOTA's draws are independent
// even when the user passes the same `options.uncertainty.seed`.
constexpr std::uint64_t kNotaSeedMixA   = 0xC0FFEEBADD06DE57ULL;
constexpr std::uint64_t kNotaSeedMixB   = 0xDEADBEEFFACEFEEDULL;
constexpr std::uint64_t kNotaPermSalt   = 0x12345678ABCDEF01ULL;

}  // namespace

bool is_none_of_the_above(
    const Sample&            test,
    const Class&             best,
    const Eigen::VectorXd&   test_vs_best_scores,
    const ClassifierOptions& options)
{
    (void)test;  // present in the signature for future per-test tweaks; unused now.

    const auto N      = static_cast<Eigen::Index>(options.uncertainty.mc_samples);
    const auto n_spec = best.specimens.rows();
    if (n_spec < 4) {
        // Not enough specimens to split into two non-trivial halves.
        return false;
    }
    if (test_vs_best_scores.size() != N) {
        return false;
    }

    const Eigen::Index half = n_spec / 2;

    Eigen::VectorXd within_scores(N);

    detail::parallel_for(static_cast<std::size_t>(N), [&](std::size_t i_) {
        const auto i  = static_cast<Eigen::Index>(i_);
        auto       rng = detail::make_stream(
            options.uncertainty.seed ^ kNotaPermSalt, i_);

        std::vector<Eigen::Index> perm(static_cast<std::size_t>(n_spec));
        std::iota(perm.begin(), perm.end(), Eigen::Index{0});
        std::shuffle(perm.begin(), perm.end(), rng);

        Sample half_a(half, best.specimens.cols());
        Sample half_b(n_spec - half, best.specimens.cols());
        for (Eigen::Index j = 0; j < half; ++j) {
            half_a.row(j) = best.specimens.row(perm[static_cast<std::size_t>(j)]);
        }
        for (Eigen::Index j = half; j < n_spec; ++j) {
            half_b.row(j - half)
                = best.specimens.row(perm[static_cast<std::size_t>(j)]);
        }

        const auto seed_a = detail::splitmix64(
            options.uncertainty.seed ^ kNotaSeedMixA ^ static_cast<std::uint64_t>(i_));
        const auto seed_b = detail::splitmix64(
            options.uncertainty.seed ^ kNotaSeedMixB ^ static_cast<std::uint64_t>(i_));
        Sample half_a_p   = perturb(half_a, options.uncertainty.epsilon, seed_a);
        Sample half_b_p   = perturb(half_b, options.uncertainty.epsilon, seed_b);

        within_scores(i) = similarity_score(half_a_p, half_b_p, options.weights);
    });

    const double mean_test   = test_vs_best_scores.mean();
    const double mean_within = within_scores.mean();
    if (mean_test <= mean_within) {
        // Test fit is at least as good as the class fits itself — accept.
        return false;
    }

    double var_test   = 0.0;
    double var_within = 0.0;
    if (N > 1) {
        var_test = (test_vs_best_scores.array() - mean_test).square().sum()
                 / static_cast<double>(N - 1);
        var_within = (within_scores.array() - mean_within).square().sum()
                   / static_cast<double>(N - 1);
    }

    const double se = std::sqrt((var_test + var_within) / static_cast<double>(N));
    if (!(se > 0.0) || !std::isfinite(se)) {
        return false;
    }

    const double t  = (mean_test - mean_within) / se;
    const double df = static_cast<double>(N - 1);
    const double p_one_sided = 1.0 - detail::students_t_cdf(t, df);

    return p_one_sided < options.nota_threshold;
}

}  // namespace mqces
