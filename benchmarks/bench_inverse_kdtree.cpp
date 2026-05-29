#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdint>
#include <random>

// Tree-aware inverse_spatial_rank vs matrix-based baseline. Each per-row
// Weiszfeld solve under the matrix path costs O(M·d) per iteration; the
// tree path costs O(k·log M) where k is the leaves opened by the
// Barnes-Hut traversal at the given opening_theta. The crossover should
// land in the M ~ a few hundred range at d = 25 for tight theta, and the
// gap should widen as M grows.
//
// Arg layout: (n_query, m_cloud, d). theta is varied across templated
// instantiations so Google Benchmark's labels group cleanly.

namespace {

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

mqces::Sample random_cloud(int m, int d, std::uint64_t seed)
{
    std::mt19937_64            rng(seed);
    std::normal_distribution<> n01(0.0, 1.0);
    mqces::Sample              y(m, d);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d; ++j) {
            y(i, j) = n01(rng);
        }
    }
    return y;
}

// Build n_query rank vectors against the cloud `y`. Each row is a valid
// rank (||u|| < 1) so the inverse solver gets well-formed targets.
RowMajor random_targets(int n_query, const mqces::Sample& y, std::uint64_t seed)
{
    const auto                 d = y.cols();
    std::mt19937_64            rng(seed);
    std::normal_distribution<> n01(0.0, 1.0);
    RowMajor                   u(n_query, d);
    for (int k = 0; k < n_query; ++k) {
        Eigen::VectorXd q(d);
        for (Eigen::Index j = 0; j < d; ++j) {
            q(j) = n01(rng);
        }
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            Eigen::VectorXd diff = q - y.row(i).transpose();
            double          nrm  = diff.norm();
            if (nrm > 1e-15) {
                acc += diff / nrm;
            }
        }
        u.row(k) = (acc / static_cast<double>(y.rows())).transpose();
    }
    return u;
}

mqces::SolverConfig make_solver()
{
    mqces::SolverConfig s;
    s.kind          = mqces::SolverKind::Weiszfeld;
    s.tol           = 1e-7;
    s.max_iters     = 1000;
    return s;
}

}  // namespace

// Matrix-based inverse Weiszfeld baseline. Per-iter cost O(M·d).
static void BM_InverseExactMatrix(benchmark::State& state)
{
    const int  n_query = static_cast<int>(state.range(0));
    const int  m_cloud = static_cast<int>(state.range(1));
    const int  d       = static_cast<int>(state.range(2));
    const auto y       = random_cloud(m_cloud, d, 0xBEEF);
    const auto u       = random_targets(n_query, y, 0xCAFE);
    const auto solver  = make_solver();

    std::size_t total_iters = 0;
    std::size_t total_calls = 0;
    for (auto _ : state) {
        auto r = mqces::inverse_spatial_rank(u, y, solver);
        benchmark::DoNotOptimize(r.x_tilde);
        total_iters += r.max_iters_used;
        ++total_calls;
    }
    if (total_calls > 0) {
        state.counters["mean_max_iters"]
            = static_cast<double>(total_iters) / static_cast<double>(total_calls);
    }
    state.counters["M"] = m_cloud;
    state.counters["d"] = d;
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_query));
}
BENCHMARK(BM_InverseExactMatrix)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{50}, {500, 2000, 5000}, {9, 25}});

// Tree-aware path. opening_theta varied across instantiations. Wrapped in
// try/catch because at loose theta the Barnes-Hut bias can be larger than
// the solver tolerance, causing the fixed-point iteration to stall and
// the per-row solve to throw std::runtime_error on non-convergence. We
// record those configurations with a `convergence_failed` counter rather
// than aborting the suite.
template <int Theta100>
static void BM_InverseKdTree(benchmark::State& state)
{
    const int  n_query = static_cast<int>(state.range(0));
    const int  m_cloud = static_cast<int>(state.range(1));
    const int  d       = static_cast<int>(state.range(2));
    const auto y       = random_cloud(m_cloud, d, 0xBEEF);
    const auto u       = random_targets(n_query, y, 0xCAFE);
    const auto solver  = make_solver();

    mqces::SamplingConfig cfg;
    cfg.strategy         = mqces::SamplingConfig::Strategy::KdTreeLocalExact;
    cfg.kd_opening_theta = static_cast<double>(Theta100) / 100.0;
    cfg.kd_leaf_size     = 16;

    std::size_t total_iters     = 0;
    std::size_t total_calls     = 0;
    std::size_t total_failures  = 0;
    for (auto _ : state) {
        try {
            auto r = mqces::inverse_spatial_rank(u, y, solver, cfg);
            benchmark::DoNotOptimize(r.x_tilde);
            total_iters += r.max_iters_used;
        } catch (const std::runtime_error&) {
            ++total_failures;
        }
        ++total_calls;
    }
    if (total_calls > 0) {
        state.counters["mean_max_iters"]
            = static_cast<double>(total_iters)
            / static_cast<double>(std::max<std::size_t>(1, total_calls - total_failures));
        state.counters["convergence_failed"] = static_cast<double>(total_failures);
    }
    state.counters["M"]      = m_cloud;
    state.counters["d"]      = d;
    state.counters["theta"]  = cfg.kd_opening_theta;
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_query));
}
BENCHMARK_TEMPLATE(BM_InverseKdTree, 50)   // theta = 0.50 — tight
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{50}, {500, 2000, 5000}, {9, 25}});
BENCHMARK_TEMPLATE(BM_InverseKdTree, 100)  // theta = 1.00 — moderate
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{50}, {500, 2000, 5000}, {9, 25}});
BENCHMARK_TEMPLATE(BM_InverseKdTree, 200)  // theta = 2.00 — loose
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{50}, {500, 2000, 5000}, {9, 25}});

// Accuracy probe: how far does the tree result drift from the matrix
// result? We report the max coordinate error as a counter so the
// benchmark output captures the bias/speed tradeoff at each theta.
template <int Theta100>
static void BM_InverseKdTreeAccuracy(benchmark::State& state)
{
    const int  n_query = static_cast<int>(state.range(0));
    const int  m_cloud = static_cast<int>(state.range(1));
    const int  d       = static_cast<int>(state.range(2));
    const auto y       = random_cloud(m_cloud, d, 0xBEEF);
    const auto u       = random_targets(n_query, y, 0xCAFE);
    const auto solver  = make_solver();

    mqces::SamplingConfig cfg;
    cfg.strategy         = mqces::SamplingConfig::Strategy::KdTreeLocalExact;
    cfg.kd_opening_theta = static_cast<double>(Theta100) / 100.0;
    cfg.kd_leaf_size     = 16;

    const auto exact = mqces::inverse_spatial_rank(u, y, solver);

    for (auto _ : state) {
        try {
            auto r = mqces::inverse_spatial_rank(u, y, solver, cfg);
            benchmark::DoNotOptimize(r.x_tilde);
            state.PauseTiming();
            const double max_err = (r.x_tilde - exact.x_tilde).array().abs().maxCoeff();
            state.counters["max_coord_err"] = max_err;
            state.ResumeTiming();
        } catch (const std::runtime_error&) {
            state.PauseTiming();
            state.counters["convergence_failed"] = 1.0;
            state.ResumeTiming();
        }
    }
    state.counters["M"]     = m_cloud;
    state.counters["theta"] = cfg.kd_opening_theta;
}
BENCHMARK_TEMPLATE(BM_InverseKdTreeAccuracy, 50)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{20}, {500, 2000}, {9, 25}})
    ->Iterations(1);
BENCHMARK_TEMPLATE(BM_InverseKdTreeAccuracy, 100)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{20}, {500, 2000}, {9, 25}})
    ->Iterations(1);
BENCHMARK_TEMPLATE(BM_InverseKdTreeAccuracy, 200)
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({{20}, {500, 2000}, {9, 25}})
    ->Iterations(1);
