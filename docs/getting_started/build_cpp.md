# Building the C++ library

## Prerequisites

- C++20-capable compiler: GCC ≥ 12, Clang ≥ 15, or MSVC 2022.
- CMake ≥ 3.24.
- (Optional) Python ≥ 3.9 if you plan to build the nanobind bindings.

All other dependencies — Eigen 5.0.1, GoogleTest, Google Benchmark, nanobind,
nlohmann/json, and optionally Kokkos — are fetched at configure time via
CMake's `FetchContent`. Nothing needs to be installed system-wide.

## Quick build

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## CMake options

Every option is prefixed `MQCES_ENABLE_*` for grep-ability.

| Option                      | Default | Purpose                                                                                  |
|-----------------------------|---------|------------------------------------------------------------------------------------------|
| `MQCES_ENABLE_TESTS`        | ON      | Build unit + integration tests.                                                          |
| `MQCES_ENABLE_BENCHMARKS`   | OFF     | Build Google Benchmark microbenchmarks.                                                  |
| `MQCES_ENABLE_PYTHON`       | OFF     | Build the nanobind Python module.                                                        |
| `MQCES_ENABLE_CLI`          | ON      | Build the `mqces` command-line interface.                                                |
| `MQCES_ENABLE_OPENMP`       | ON      | Use OpenMP for the parallel shim.                                                        |
| `MQCES_ENABLE_KOKKOS`       | OFF     | Use Kokkos as the parallel backend instead.                                              |
| `MQCES_ENABLE_KOKKOS_CUDA`  | OFF     | Enable the Kokkos CUDA execution space.                                                  |
| `MQCES_ENABLE_SANITIZER`    | none    | One of `none / asan / ubsan / tsan`.                                                     |
| `MQCES_ENABLE_COVERAGE`     | OFF     | Instrument for gcovr / llvm-cov.                                                         |
| `MQCES_ENABLE_CLANG_TIDY`   | OFF     | Run clang-tidy at compile time.                                                          |
| `MQCES_ENABLE_INSTALL`      | OFF     | Emit install rules. Requires a system-installed Eigen3.                                  |
| `MQCES_ENABLE_DOCS`         | OFF     | Build the Sphinx + Breathe + Doxygen documentation (this site).                          |

## Consuming mqces from another CMake project

When configured with `MQCES_ENABLE_INSTALL=ON` and installed, mqces emits a
`mqcesConfig.cmake` package. Consumers can then use it via `find_package`:

```cmake
find_package(mqces CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE mqces::mqces)
```

```{note}
`MQCES_ENABLE_INSTALL=ON` requires a system-installed Eigen3 — the
`FetchContent`-pulled Eigen cannot be re-exported through `mqcesTargets`.
```
