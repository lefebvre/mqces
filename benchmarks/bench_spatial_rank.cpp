#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <random>

namespace {

mqces::Sample random_sample(int n, int d, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<> dist(0.0, 1.0);
  mqces::Sample s(n, d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      s(i, j) = dist(rng);
    }
  }
  return s;
}

}  // namespace

// spatial_rank is O(N^2 * d). Sweep N at fixed d = 9 (paper's feature count).
static void BM_SpatialRank(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  const int d = 9;
  const auto x = random_sample(n, d, 0xBEEF);
  for (auto _ : state) {
    auto u = mqces::spatial_rank(x);
    benchmark::DoNotOptimize(u);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * n);
  state.SetLabel("N×N pair evaluations");
}
BENCHMARK(BM_SpatialRank)->Arg(50)->Arg(100)->Arg(250)->Arg(500)->Arg(1000);
