#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <random>

namespace {

mqces::Sample gaussian(int n, int d, double center, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> dist(center, 1.0);
    mqces::Sample              s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = dist(rng);
        }
    }
    return s;
}

}  // namespace

// One similarity_score call does spatial_rank twice (O(N²)) plus two
// inverse_spatial_rank passes (Weiszfeld × N rows × O(N) per iter).
// Sweep symmetric N at fixed d = 9.
static void BM_SimilarityScore(benchmark::State& state)
{
    const int n = static_cast<int>(state.range(0));
    const int d = 9;
    const auto                  x = gaussian(n, d, 0.0, 0x1);
    const auto                  y = gaussian(n, d, 0.5, 0x2);
    const mqces::FeatureWeights w = mqces::FeatureWeights::Ones(d);
    for (auto _ : state) {
        double s = mqces::similarity_score(x, y, w);
        benchmark::DoNotOptimize(s);
    }
    state.SetItemsProcessed(state.iterations());
    state.counters["N"] = n;
}
BENCHMARK(BM_SimilarityScore)->Arg(20)->Arg(50)->Arg(100)->Arg(250);
