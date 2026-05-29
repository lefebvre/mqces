#include <mqces/quantile.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdint>
#include <random>

// Proxy for the production-scale similarity_score call at N = 10⁶ rows.
// We can't actually run that in CI (a single exact similarity_score at
// N = 10⁶ would be hours), so we benchmark the SUBSAMPLED path at
// N = 2000 with R = 200 — a 10× subsampling ratio at d = 25. This is
// representative of the production target ratio (N = 10⁶, R = 10⁴, 100×
// subsampling): same per-row arithmetic structure, just less iteration
// count over the outer N loop.
//
// Bench params:
//   - "exact_w" — Weiszfeld, sampling.reference_size == 0 (full N).
//   - "sub_w"   — Weiszfeld with R = 200.
//   - "sub_vz"  — Vardi-Zhang with R = 200.
//   - "sub_aa"  — VZ + Anderson with R = 200 (production target).

namespace {

mqces::Sample random_sample(int n, int d, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> n01(0.0, 1.0);
    mqces::Sample              s(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            s(i, j) = n01(rng);
        }
    }
    return s;
}

constexpr int kProxyN = 2000;
constexpr int kProxyD = 25;
constexpr int kProxyR = 200;

mqces::SolverConfig solver_for(mqces::SolverKind kind)
{
    mqces::SolverConfig s;
    s.kind          = kind;
    s.tol           = (kind == mqces::SolverKind::Weiszfeld) ? 1e-5 : 1e-7;
    s.max_iters     = 1000;
    s.vz_vertex_eps = 1e-6;
    s.aa_window     = 5;
    s.aa_reg        = 1e-12;
    s.aa_safeguard  = true;
    return s;
}

// One-time setup for similarity_score inputs — shared across all bench
// targets so the per-iteration body is purely the call we're measuring.
struct Fixture {
    mqces::Sample         x;
    mqces::Sample         y;
    mqces::FeatureWeights w;
    Fixture() :
        x(random_sample(kProxyN, kProxyD, 1)),
        y(random_sample(kProxyN, kProxyD, 2)),
        w(mqces::FeatureWeights::Ones(kProxyD)) {}
};

const Fixture& proxy_fixture()
{
    static const Fixture f;
    return f;
}

mqces::SamplingConfig subsample(std::size_t r)
{
    mqces::SamplingConfig c;
    c.reference_size = r;
    c.seed           = 0xACE1ULL;
    return c;
}

}  // namespace

// Exact Weiszfeld at the proxy scale. Slow — set MinTime small or
// Iterations(1) so it doesn't dominate the benchmark suite wall time.
static void BM_ScalingProxy_ExactWeiszfeld(benchmark::State& state)
{
    const auto& f = proxy_fixture();
    const auto  s = solver_for(mqces::SolverKind::Weiszfeld);
    for (auto _ : state) {
        double sc = mqces::similarity_score(f.x, f.y, f.w, s, mqces::SamplingConfig{});
        benchmark::DoNotOptimize(sc);
    }
    state.counters["N"] = kProxyN;
    state.counters["R"] = kProxyN;
}
BENCHMARK(BM_ScalingProxy_ExactWeiszfeld)
    ->Unit(benchmark::kSecond)
    ->Iterations(1);

// Subsampled at R = 200, Weiszfeld inner solver.
static void BM_ScalingProxy_SubsampledWeiszfeld(benchmark::State& state)
{
    const auto& f = proxy_fixture();
    const auto  s = solver_for(mqces::SolverKind::Weiszfeld);
    const auto  cfg = subsample(kProxyR);
    for (auto _ : state) {
        double sc = mqces::similarity_score(f.x, f.y, f.w, s, cfg);
        benchmark::DoNotOptimize(sc);
    }
    state.counters["N"] = kProxyN;
    state.counters["R"] = kProxyR;
}
BENCHMARK(BM_ScalingProxy_SubsampledWeiszfeld)->Unit(benchmark::kMillisecond);

// Subsampled at R = 200, Vardi-Zhang inner solver.
static void BM_ScalingProxy_SubsampledVZ(benchmark::State& state)
{
    const auto& f = proxy_fixture();
    const auto  s = solver_for(mqces::SolverKind::VardiZhang);
    const auto  cfg = subsample(kProxyR);
    for (auto _ : state) {
        double sc = mqces::similarity_score(f.x, f.y, f.w, s, cfg);
        benchmark::DoNotOptimize(sc);
    }
    state.counters["N"] = kProxyN;
    state.counters["R"] = kProxyR;
}
BENCHMARK(BM_ScalingProxy_SubsampledVZ)->Unit(benchmark::kMillisecond);

// Subsampled at R = 200, VZ + Anderson acceleration. This is the
// production-target combination.
static void BM_ScalingProxy_SubsampledVZAA(benchmark::State& state)
{
    const auto& f = proxy_fixture();
    const auto  s = solver_for(mqces::SolverKind::VardiZhangAA);
    const auto  cfg = subsample(kProxyR);
    for (auto _ : state) {
        double sc = mqces::similarity_score(f.x, f.y, f.w, s, cfg);
        benchmark::DoNotOptimize(sc);
    }
    state.counters["N"] = kProxyN;
    state.counters["R"] = kProxyR;
}
BENCHMARK(BM_ScalingProxy_SubsampledVZAA)->Unit(benchmark::kMillisecond);
