#pragma once

#include <mqces/types.hpp>

#include <mqces/detail/rng.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace mqces::detail {

// Floyd's algorithm for sampling `R` distinct indices uniformly at random
// from {0, 1, ..., N − 1}, deterministically given `seed`. O(R) expected
// time and memory — much better than an O(N) Fisher-Yates shuffle when
// R ≪ N (the production case at N = 10⁶, R = 10⁴).
//
// Returns sorted indices so downstream loops over them are cache-friendly.
//
// Edge cases:
//   * R == 0 returns an empty vector.
//   * R >= N returns {0, 1, ..., N−1} (the "exact" reference set).
inline std::vector<Eigen::Index> uniform_reference_indices(
    std::size_t N, std::size_t R, std::uint64_t seed)
{
    if (R == 0 || N == 0) {
        return {};
    }
    if (R >= N) {
        std::vector<Eigen::Index> all(N);
        std::iota(all.begin(), all.end(), Eigen::Index{0});
        return all;
    }

    auto rng = make_stream(seed, 0);

    // Floyd's algorithm (Bentley 1987): for j = N−R..N−1, draw t ∈ [0, j];
    // if t is already selected, add j; else add t. Produces a uniformly
    // random R-subset.
    std::unordered_set<Eigen::Index> selected;
    selected.reserve(R);
    for (std::size_t j = N - R; j < N; ++j) {
        std::uniform_int_distribution<std::size_t> dist(0, j);
        auto t = static_cast<Eigen::Index>(dist(rng));
        if (selected.contains(t)) {
            selected.insert(static_cast<Eigen::Index>(j));
        } else {
            selected.insert(t);
        }
    }

    std::vector<Eigen::Index> result(selected.begin(), selected.end());
    std::sort(result.begin(), result.end());
    return result;
}

// Materialize a row-major submatrix from `x` at the given row indices.
// Caller controls the storage order so downstream solver loops can rely on
// contiguous row access.
inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
gather_rows(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x,
    const std::vector<Eigen::Index>& indices)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(
        static_cast<Eigen::Index>(indices.size()), x.cols());
    for (std::size_t k = 0; k < indices.size(); ++k) {
        out.row(static_cast<Eigen::Index>(k)) = x.row(indices[k]);
    }
    return out;
}

}  // namespace mqces::detail
