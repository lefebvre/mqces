#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <random>
#include <span>
#include <string>
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

struct Setup {
  mqces::Sample test;
  std::vector<mqces::Class> known;
  mqces::ClassifierOptions opts;
};

Setup build_setup(int n_classes, int n_specimens, int d, double epsilon, std::size_t mc_samples) {
  Setup s;
  s.known.reserve(static_cast<std::size_t>(n_classes));
  for (int k = 0; k < n_classes; ++k) {
    const double center = static_cast<double>(k) - 0.5 * n_classes;
    s.known.push_back({"k=" + std::to_string(k),
                       gaussian(n_specimens, d, center, 0x100u + static_cast<unsigned>(k))});
  }
  s.test = gaussian(n_specimens, d, 0.0, 0xFEED);
  s.opts.weights = mqces::FeatureWeights::Ones(d);
  s.opts.uncertainty.epsilon = epsilon;
  s.opts.uncertainty.mc_samples = mc_samples;
  s.opts.uncertainty.seed = 0xBEEF;
  s.opts.n_threads = 0;
  return s;
}

}  // namespace

// Full classifier pipeline on K classes: K observed scores, jackknife
// replicates for the best and runner-up classes, and the NOTA permutation
// test. Items processed counts whole classifications.
static void BM_ClassifyClassic_End2End(benchmark::State& state) {
  const int n_classes = static_cast<int>(state.range(0));
  const int n_specimens = 50;
  const int d = 9;
  const double epsilon = 0.0;
  const std::size_t mc = 10;

  auto s = build_setup(n_classes, n_specimens, d, epsilon, mc);
  for (auto _ : state) {
    auto r = mqces::classic::classify(s.test, std::span<const mqces::Class>{s.known}, s.opts);
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["K"] = n_classes;
  state.counters["N"] = static_cast<double>(mc);
}
BENCHMARK(BM_ClassifyClassic_End2End)->Arg(5)->Arg(10)->Arg(19);

// Epsilon sweep at fixed K, N.
static void BM_ClassifyClassic_EpsilonSweep(benchmark::State& state) {
  const int n_classes = 19;
  const int n_specimens = 50;
  const int d = 9;
  const double epsilon = static_cast<double>(state.range(0)) / 100.0;  // 1, 5, 10, 20, 40 %
  const std::size_t mc = 10;
  auto s = build_setup(n_classes, n_specimens, d, epsilon, mc);
  for (auto _ : state) {
    auto r = mqces::classic::classify(s.test, std::span<const mqces::Class>{s.known}, s.opts);
    benchmark::DoNotOptimize(r);
  }
  state.counters["epsilon_pct"] = static_cast<double>(state.range(0));
}
BENCHMARK(BM_ClassifyClassic_EpsilonSweep)->Arg(1)->Arg(5)->Arg(10)->Arg(20)->Arg(40);

// The same 19-class classification with each inverse-rank solver. Arg 0 is
// Weiszfeld, 1 is Anderson.
static void BM_ClassifyClassic_Solver(benchmark::State& state) {
  auto s = build_setup(19, 50, 9, 0.0, 10);
  if (state.range(0) == 1) {
    s.opts.solver.kind = mqces::InverseRankSolver::Anderson;
  }
  for (auto _ : state) {
    auto r = mqces::classic::classify(s.test, std::span<const mqces::Class>{s.known}, s.opts);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_ClassifyClassic_Solver)->Arg(0)->Arg(1)->Unit(benchmark::kMillisecond);

// A test sample far from every class. None-of-the-above then pools it with
// the best class, and its permutations produce inverse-rank problems whose
// solutions lie between the two groups, where plain Weiszfeld is slowest.
// Arg 0 is Weiszfeld, 1 is Anderson.
static void BM_ClassifyClassic_Outlier(benchmark::State& state) {
  auto s = build_setup(5, 50, 9, 0.0, 10);
  s.test = gaussian(50, 9, 200.0, 0xD1CE);
  if (state.range(0) == 1) {
    s.opts.solver.kind = mqces::InverseRankSolver::Anderson;
  }
  for (auto _ : state) {
    auto r = mqces::classic::classify(s.test, std::span<const mqces::Class>{s.known}, s.opts);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_ClassifyClassic_Outlier)->Arg(0)->Arg(1)->Unit(benchmark::kMillisecond);
