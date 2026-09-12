#include <mqces/classify.hpp>
#include <mqces/detail/parallel.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

mqces::Sample gaussian(int n, int d, double center, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<> dist(center, 0.5);
  mqces::Sample s(n, d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      s(i, j) = dist(rng);
    }
  }
  return s;
}

}  // namespace

// Strong scaling: fixed problem size, vary thread count. The classifier
// parallelizes over classes, jackknife replicates, perturbation replicates
// and NOTA permutations.
static void BM_ClassifyStrongScaling(benchmark::State& state) {
  const int n_threads = static_cast<int>(state.range(0));
  const int n_classes = 19;
  const int n_specimens = 50;
  const int d = 9;
  const std::size_t mc = 64;  // wider than usual so threading pays

  std::vector<mqces::Class> known;
  known.reserve(static_cast<std::size_t>(n_classes));
  for (int k = 0; k < n_classes; ++k) {
    const double center = static_cast<double>(k) - 0.5 * n_classes;
    known.push_back({"k=" + std::to_string(k), gaussian(n_specimens, d, center, 0x200u + k)});
  }
  mqces::Sample test = gaussian(n_specimens, d, 0.0, 0xFEED);

  mqces::ClassifierOptions opts;
  opts.weights = mqces::FeatureWeights::Ones(d);
  opts.uncertainty.epsilon = 0.01;
  opts.uncertainty.mc_samples = mc;
  opts.uncertainty.seed = 1;
  opts.n_threads = n_threads;

  for (auto _ : state) {
    auto r = mqces::classic::classify(test, std::span<const mqces::Class>{known}, opts);
    benchmark::DoNotOptimize(r);
  }
  state.counters["threads"] = n_threads;
}

// Argue thread counts up to logical cores (capped at 32 for sanity).
static int max_thread_arg() {
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  if (hw <= 0) {
    hw = 1;
  }
  return std::min(hw, 32);
}

BENCHMARK(BM_ClassifyStrongScaling)
  ->Arg(1)
  ->Arg(2)
  ->Arg(4)
  ->Arg(8)
  ->Arg(max_thread_arg())
  ->Unit(benchmark::kMillisecond);
