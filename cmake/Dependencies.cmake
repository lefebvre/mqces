# ----------------------------------------------------------------------------
# Dependencies.cmake
#
# All third-party deps are fetched via FetchContent and pinned to the latest
# stable release tag at the time of bootstrap. Bump a tag by editing the
# corresponding MQCES_DEP_*_TAG variable below, or run
# `python3 scripts/refresh_deps.py --apply` to refresh them all.
#
# Resolved 2026-05-27 via `git ls-remote --tags --refs`, filtered to stable
# semver tags (no rc/alpha/beta).
# ----------------------------------------------------------------------------

include(FetchContent)

set(MQCES_DEP_EIGEN_TAG     "5.0.1"   CACHE STRING "Eigen release tag")
set(MQCES_DEP_GTEST_TAG     "v1.17.0" CACHE STRING "GoogleTest release tag")
set(MQCES_DEP_BENCHMARK_TAG "v1.9.5"  CACHE STRING "Google Benchmark release tag")
set(MQCES_DEP_NANOBIND_TAG  "v2.12.0" CACHE STRING "nanobind release tag")
set(MQCES_DEP_KOKKOS_TAG    "5.1.1"   CACHE STRING "Kokkos release tag")
set(MQCES_DEP_JSON_TAG      "v3.12.0" CACHE STRING "nlohmann/json release tag")

# ---- Eigen -----------------------------------------------------------------
# The core library depends on Eigen unconditionally.
set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS      OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        ${MQCES_DEP_EIGEN_TAG}
    GIT_SHALLOW    TRUE
    SYSTEM
)
FetchContent_MakeAvailable(eigen)

# ---- GoogleTest ------------------------------------------------------------
if(MQCES_ENABLE_TESTS)
    set(BUILD_GMOCK            ON  CACHE BOOL "" FORCE)
    set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON  CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        ${MQCES_DEP_GTEST_TAG}
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(googletest)
endif()

# ---- Google Benchmark ------------------------------------------------------
if(MQCES_ENABLE_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING        OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL        OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR         OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON  CACHE BOOL "" FORCE)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        ${MQCES_DEP_BENCHMARK_TAG}
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(benchmark)
endif()

# ---- nanobind --------------------------------------------------------------
if(MQCES_ENABLE_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        ${MQCES_DEP_NANOBIND_TAG}
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# ---- Kokkos ----------------------------------------------------------------
if(MQCES_ENABLE_KOKKOS)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
    if(MQCES_ENABLE_OPENMP)
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    endif()
    if(MQCES_ENABLE_KOKKOS_CUDA)
        set(Kokkos_ENABLE_CUDA                         ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_LAMBDA                  ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_CONSTEXPR               ON CACHE BOOL "" FORCE)
        set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        ${MQCES_DEP_KOKKOS_TAG}
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(kokkos)
endif()

# ---- nlohmann/json --------------------------------------------------------
# Used by the CLI and (optionally) other consumers. Header-only.
if(MQCES_ENABLE_CLI)
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install    OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        ${MQCES_DEP_JSON_TAG}
        GIT_SHALLOW    TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(json)
endif()

# ---- OpenMP (system dep, not FetchContent) ---------------------------------
# When Kokkos is on, it brings its own OpenMP wiring; only seek it directly
# when we're using the OpenMP-only parallel backend.
if(MQCES_ENABLE_OPENMP AND NOT MQCES_ENABLE_KOKKOS)
    find_package(OpenMP REQUIRED)
endif()
