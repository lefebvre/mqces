#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdint>
#include <random>
#include <vector>

// Iteration-count + wall-time comparison across the three inner solvers
// (Weiszfeld, Vardi-Zhang, VZ+Anderson) on a representative inverse-rank
// workload. Args are (N_query, M_cloud, d) and each solver kind has its
// own templated benchmark family so Google Benchmark's output groups the
// solvers separately and the dispatched function is monomorphic in the
// hot path.

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

// Build N_query rank vectors against the cloud `y`. Each row is a valid
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

mqces::SolverConfig make_solver(mqces::SolverKind kind)
{
    mqces::SolverConfig s;
    s.kind          = kind;
    s.tol           = 1e-7;
    s.max_iters     = 1000;
    s.vz_vertex_eps = 1e-6;
    s.aa_window     = 5;
    s.aa_reg        = 1e-12;
    s.aa_safeguard  = true;
    return s;
}

}  // namespace

template <mqces::SolverKind kind>
static void BM_Solver(benchmark::State& state)
{
    const int  n_query = static_cast<int>(state.range(0));
    const int  m_cloud = static_cast<int>(state.range(1));
    const int  d       = static_cast<int>(state.range(2));
    const auto y       = random_cloud(m_cloud, d, 0xBEEF);
    const auto u       = random_targets(n_query, y, 0xCAFE);
    const auto solver  = make_solver(kind);

    std::size_t total_iters  = 0;
    std::size_t total_calls  = 0;
    std::size_t total_fallbacks = 0;

    for (auto _ : state) {
        auto r = mqces::inverse_spatial_rank(u, y, solver);
        benchmark::DoNotOptimize(r);
        total_iters += r.max_iters_used;
        total_fallbacks += r.aa_fallbacks;
        ++total_calls;
    }
    if (total_calls > 0) {
        state.counters["mean_max_iters"]
            = static_cast<double>(total_iters) / static_cast<double>(total_calls);
        state.counters["mean_aa_fallbacks"]
            = static_cast<double>(total_fallbacks) / static_cast<double>(total_calls);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_query));
}

BENCHMARK_TEMPLATE(BM_Solver, mqces::SolverKind::Weiszfeld)
    ->ArgsProduct({{1, 10, 100}, {100, 1000}, {9, 25}});
BENCHMARK_TEMPLATE(BM_Solver, mqces::SolverKind::VardiZhang)
    ->ArgsProduct({{1, 10, 100}, {100, 1000}, {9, 25}});
BENCHMARK_TEMPLATE(BM_Solver, mqces::SolverKind::VardiZhangAA)
    ->ArgsProduct({{1, 10, 100}, {100, 1000}, {9, 25}});
