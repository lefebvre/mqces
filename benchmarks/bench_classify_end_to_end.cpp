#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

mqces::Sample gaussian(int n, int d, double center, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> dist(center, 0.5);
    mqces::Sample              s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = dist(rng);
        }
    }
    return s;
}

struct Setup {
    mqces::Sample             test;
    std::vector<mqces::Class> known;
    mqces::ClassifierOptions  opts;
};

Setup build_setup(int n_classes, int n_specimens, int d, double epsilon,
                  std::size_t mc_samples)
{
    Setup s;
    s.known.reserve(static_cast<std::size_t>(n_classes));
    for (int k = 0; k < n_classes; ++k) {
        const double center = static_cast<double>(k) - 0.5 * n_classes;
        s.known.push_back({"k=" + std::to_string(k),
                           gaussian(n_specimens, d, center, 0x100u + k)});
    }
    s.test                        = gaussian(n_specimens, d, 0.0, 0xFEED);
    s.opts.weights                = mqces::FeatureWeights::Ones(d);
    s.opts.uncertainty.epsilon    = epsilon;
    s.opts.uncertainty.mc_samples = mc_samples;
    s.opts.uncertainty.seed       = 0xBEEF;
    s.opts.n_threads              = 0;
    return s;
}

}  // namespace

// Full classifier pipeline: build options, run K × N classifications,
// produce a result.
static void BM_ClassifyClassic_End2End(benchmark::State& state)
{
    const int    n_classes   = static_cast<int>(state.range(0));
    const int    n_specimens = 50;
    const int    d           = 9;
    const double epsilon     = 0.0;
    const std::size_t mc      = 10;

    auto s = build_setup(n_classes, n_specimens, d, epsilon, mc);
    for (auto _ : state) {
        auto r = mqces::classic::classify(
            s.test, std::span<const mqces::Class>{s.known}, s.opts);
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(state.iterations()
                            * static_cast<int64_t>(n_classes) * mc);
    state.counters["K"]  = n_classes;
    state.counters["N"]  = static_cast<double>(mc);
}
BENCHMARK(BM_ClassifyClassic_End2End)->Arg(5)->Arg(10)->Arg(19);

// Epsilon sweep at fixed K, N.
static void BM_ClassifyClassic_EpsilonSweep(benchmark::State& state)
{
    const int    n_classes   = 19;
    const int    n_specimens = 50;
    const int    d           = 9;
    const double epsilon     = state.range(0) / 100.0;  // 1, 5, 10, 20, 40 %
    const std::size_t mc      = 10;
    auto s = build_setup(n_classes, n_specimens, d, epsilon, mc);
    for (auto _ : state) {
        auto r = mqces::classic::classify(
            s.test, std::span<const mqces::Class>{s.known}, s.opts);
        benchmark::DoNotOptimize(r);
    }
    state.counters["epsilon_pct"] = static_cast<double>(state.range(0));
}
BENCHMARK(BM_ClassifyClassic_EpsilonSweep)->Arg(1)->Arg(5)->Arg(10)->Arg(20)->Arg(40);

// v2: paired-difference t but same loose Weiszfeld inner solver.
static void BM_ClassifyV2_End2End(benchmark::State& state)
{
    const int         n_classes   = static_cast<int>(state.range(0));
    const int         n_specimens = 50;
    const int         d           = 9;
    const double      epsilon     = 0.0;
    const std::size_t mc          = 10;
    auto              s = build_setup(n_classes, n_specimens, d, epsilon, mc);
    for (auto _ : state) {
        auto r = mqces::v2::classify(
            s.test, std::span<const mqces::Class>{s.known}, s.opts);
        benchmark::DoNotOptimize(r);
    }
    state.counters["K"] = n_classes;
}
BENCHMARK(BM_ClassifyV2_End2End)->Arg(5)->Arg(10)->Arg(19);

// v3: Vardi-Zhang inner solver. Per-call cost should be similar to v2
// when no Weiszfeld plateaus are hit; otherwise v3 stays below v2's
// worst-case (no throw + tight convergence).
static void BM_ClassifyV3_End2End(benchmark::State& state)
{
    const int         n_classes   = static_cast<int>(state.range(0));
    const int         n_specimens = 50;
    const int         d           = 9;
    const double      epsilon     = 0.0;
    const std::size_t mc          = 10;
    auto              s = build_setup(n_classes, n_specimens, d, epsilon, mc);
    for (auto _ : state) {
        auto r = mqces::v3::classify(
            s.test, std::span<const mqces::Class>{s.known}, s.opts);
        benchmark::DoNotOptimize(r);
    }
    state.counters["K"] = n_classes;
}
BENCHMARK(BM_ClassifyV3_End2End)->Arg(5)->Arg(10)->Arg(19);

// v4: VZ + Anderson acceleration. The per-call wall time is expected to
// drop notably as N grows (each replicate of similarity_score benefits
// from AA's reduced iteration count).
static void BM_ClassifyV4_End2End(benchmark::State& state)
{
    const int         n_classes   = static_cast<int>(state.range(0));
    const int         n_specimens = 50;
    const int         d           = 9;
    const double      epsilon     = 0.0;
    const std::size_t mc          = 10;
    auto              s = build_setup(n_classes, n_specimens, d, epsilon, mc);
    for (auto _ : state) {
        auto r = mqces::v4::classify(
            s.test, std::span<const mqces::Class>{s.known}, s.opts);
        benchmark::DoNotOptimize(r);
    }
    state.counters["K"] = n_classes;
}
BENCHMARK(BM_ClassifyV4_End2End)->Arg(5)->Arg(10)->Arg(19);
