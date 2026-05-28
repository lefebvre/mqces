#pragma once

#include <mqces/score.hpp>
#include <mqces/types.hpp>
#include <mqces/uncertainty.hpp>

#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace mqces::detail {

inline void validate_classify_inputs(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& opts)
{
    if (known.empty()) {
        throw std::invalid_argument("classify: known classes list is empty");
    }
    if (test.rows() == 0) {
        throw std::invalid_argument("classify: test sample has zero rows");
    }
    const Eigen::Index d = test.cols();
    if (opts.weights.size() != d) {
        throw std::invalid_argument(
            "classify: weights.size() (" + std::to_string(opts.weights.size())
            + ") must equal test.cols() (" + std::to_string(d) + ")");
    }
    for (std::size_t k = 0; k < known.size(); ++k) {
        if (known[k].specimens.cols() != d) {
            throw std::invalid_argument(
                "classify: class '" + known[k].name + "' has feature count "
                + std::to_string(known[k].specimens.cols())
                + " but test has " + std::to_string(d));
        }
        if (known[k].specimens.rows() == 0) {
            throw std::invalid_argument(
                "classify: class '" + known[k].name + "' has zero specimens");
        }
    }
    if (opts.uncertainty.mc_samples == 0) {
        throw std::invalid_argument("classify: uncertainty.mc_samples must be > 0");
    }
}

// Derive a deterministic seed for replicate i and class k. The test sample
// uses k = static_cast<std::size_t>(-1) (i.e. SIZE_MAX) as a sentinel.
inline std::uint64_t replicate_seed(
    std::uint64_t base, std::size_t replicate, std::size_t class_index)
{
    // splitmix64 of (base ^ replicate_mix ^ class_mix); each component is
    // independently splitmix'd so the bits are well-decorrelated.
    return splitmix64(base
        ^ splitmix64(static_cast<std::uint64_t>(replicate) * 0x9E3779B97F4A7C15ULL)
        ^ splitmix64(static_cast<std::uint64_t>(class_index) + 1));
}

// Run the Monte-Carlo replicate loop. Returns a (K × N) matrix where row k
// holds the N similarity scores of (perturbed test) vs (perturbed class k).
inline Eigen::MatrixXd collect_replicate_scores(
    const Sample& test, std::span<const Class> known, const ClassifierOptions& opts)
{
    const auto K = static_cast<Eigen::Index>(known.size());
    const auto N = static_cast<Eigen::Index>(opts.uncertainty.mc_samples);
    Eigen::MatrixXd scores(K, N);

    if (opts.n_threads > 0) {
        set_max_threads(opts.n_threads);
    }
    const double eps  = opts.uncertainty.epsilon;
    const auto   base = opts.uncertainty.seed;

    parallel_for(static_cast<std::size_t>(N), [&](std::size_t i_) {
        const auto i      = static_cast<Eigen::Index>(i_);
        Sample     test_p = perturb(test, eps, replicate_seed(base, i_, static_cast<std::size_t>(-1)));
        for (Eigen::Index k = 0; k < K; ++k) {
            Sample class_p = perturb(
                known[static_cast<std::size_t>(k)].specimens, eps,
                replicate_seed(base, i_, static_cast<std::size_t>(k)));
            // similarity_score's 3-arg form uses its own (loose) Weiszfeld
            // default; v1/v2 don't honor opts.solver. v3/v4 (tranches k/l)
            // will introduce their own MC paths that pass opts.solver.
            scores(k, i) = similarity_score(test_p, class_p, opts.weights);
        }
    });
    return scores;
}

// Sample mean of a row of `scores`.
inline double row_mean(const Eigen::MatrixXd& scores, Eigen::Index k)
{
    return scores.row(k).mean();
}

// Sample variance (Bessel-corrected, divisor N − 1) of a row.
inline double row_variance(const Eigen::MatrixXd& scores, Eigen::Index k)
{
    const auto   N = scores.cols();
    const double m = row_mean(scores, k);
    if (N <= 1) {
        return 0.0;
    }
    return (scores.row(k).array() - m).square().sum() / static_cast<double>(N - 1);
}

}  // namespace mqces::detail
