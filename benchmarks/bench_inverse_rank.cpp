#include <mqces/quantile.hpp>
#include <mqces/types.hpp>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <random>

namespace {

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Inputs {
  RowMajor u;       // (N, d) rank vectors derived from a real query cloud
  mqces::Sample y;  // (M, d) reference cloud
};

Inputs make_inputs(int n_query, int m_cloud, int d, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<> dist(0.0, 1.0);

  mqces::Sample y(m_cloud, d);
  for (int i = 0; i < m_cloud; ++i) {
    for (int j = 0; j < d; ++j) {
      y(i, j) = dist(rng);
    }
  }

  // Build n_query target rank vectors from random reference points so
  // each row is guaranteed to be a valid (||u|| < 1) rank vector.
  RowMajor u(n_query, d);
  for (int k = 0; k < n_query; ++k) {
    Eigen::VectorXd q(d);
    for (int j = 0; j < d; ++j) {
      q(j) = dist(rng);
    }
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(d);
    for (int i = 0; i < m_cloud; ++i) {
      Eigen::VectorXd diff = q - y.row(i).transpose();
      double nrm = diff.norm();
      if (nrm > 1e-15) {
        acc += diff / nrm;
      }
    }
    u.row(k) = (acc / m_cloud).transpose();
  }
  return {std::move(u), std::move(y)};
}

}  // namespace

// Cost is dominated by the Weiszfeld iterations × per-iter O(M * d) work,
// per query row. Sweep query-row count at fixed cloud size and dim.
static void BM_InverseSpatialRank(benchmark::State& state) {
  const int n_query = static_cast<int>(state.range(0));
  const int m_cloud = 100;
  const int d = 9;
  auto in = make_inputs(n_query, m_cloud, d, 0xCAFE);
  for (auto _ : state) {
    auto result = mqces::inverse_spatial_rank(in.u, in.y, 1e-7, 500);
    benchmark::DoNotOptimize(result.x_tilde);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_query));
  state.counters["M"] = m_cloud;
}
BENCHMARK(BM_InverseSpatialRank)->Arg(10)->Arg(50)->Arg(100)->Arg(250);
