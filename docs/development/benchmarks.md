# Benchmarks

Google Benchmark microbenchmarks live under `benchmarks/`. They're opt-in
via `MQCES_ENABLE_BENCHMARKS=ON`.

## Running

```bash
cmake -B build-bench -DMQCES_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench -j
./build-bench/benchmarks/mqces_bench
```

Useful Google Benchmark flags:

```bash
# Filter to specific benchmarks (regex).
./build-bench/benchmarks/mqces_bench --benchmark_filter=score

# Dump JSON for downstream analysis.
./build-bench/benchmarks/mqces_bench --benchmark_format=json > results.json

# Repeat each benchmark for stable medians.
./build-bench/benchmarks/mqces_bench --benchmark_repetitions=5
```

## What's measured

The benchmark suite covers:

- `similarity_score` at varying N, d, and reference-size R.
- `spatial_rank` exact vs subsample paths.
- `inverse_spatial_rank` per solver kind (Weiszfeld / VZ / VZ+AA).
- End-to-end `classify` across the four variants.
- Scaling sweeps used to validate the O(N·R) sampling claim.

## Interpreting results

mqces's headline performance claim — ~100× speedup at N = 10⁶ with
R = 10⁴ — comes from these benchmarks. When proposing a change that
touches the score, rank, or solver paths, run the benchmark suite
before/after and include the delta in the PR description.
