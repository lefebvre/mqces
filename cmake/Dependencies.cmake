# ----------------------------------------------------------------------------
# Dependencies.cmake
#
# All third-party deps are fetched via FetchContent and pinned by commit.
# Each MQCES_DEP_*_TAG names the stable release the commit was resolved from;
# only the MQCES_DEP_*_COMMIT is passed to git, since a tag can be moved
# upstream and would silently change what a build compiles. Pinning a commit
# rules out GIT_SHALLOW (git can only shallow-clone a branch or tag name).
#
# Bump with `python3 scripts/refresh_deps.py --apply`, which rewrites both the
# tag and its commit and reports tags that no longer match their pin.
#
# Resolved 2026-05-27 via `git ls-remote --tags --refs`, filtered to stable
# semver tags (no rc/alpha/beta).
# ----------------------------------------------------------------------------

include(FetchContent)

set(MQCES_DEP_EIGEN_TAG        "5.0.1"                                    CACHE STRING "Eigen release tag")
set(MQCES_DEP_EIGEN_COMMIT     "bc3b39870ecb690a623a3f49149a358b95c5781d" CACHE STRING "Eigen commit for MQCES_DEP_EIGEN_TAG")

set(MQCES_DEP_GTEST_TAG        "v1.17.0"                                  CACHE STRING "GoogleTest release tag")
set(MQCES_DEP_GTEST_COMMIT     "52eb8108c5bdec04579160ae17225d66034bd723" CACHE STRING "GoogleTest commit for MQCES_DEP_GTEST_TAG")

set(MQCES_DEP_BENCHMARK_TAG    "v1.9.5"                                   CACHE STRING "Google Benchmark release tag")
set(MQCES_DEP_BENCHMARK_COMMIT "192ef10025eb2c4cdd392bc502f0c852196baa48" CACHE STRING "Google Benchmark commit for MQCES_DEP_BENCHMARK_TAG")

set(MQCES_DEP_NANOBIND_TAG     "v2.12.0"                                  CACHE STRING "nanobind release tag")
set(MQCES_DEP_NANOBIND_COMMIT  "2a61ad2494d09fecb2e13322c1383342c299900d" CACHE STRING "nanobind commit for MQCES_DEP_NANOBIND_TAG")

set(MQCES_DEP_JSON_TAG         "v3.12.0"                                  CACHE STRING "nlohmann/json release tag")
set(MQCES_DEP_JSON_COMMIT      "55f93686c01528224f448c19128836e7df245f72" CACHE STRING "nlohmann/json commit for MQCES_DEP_JSON_TAG")

# ---- Eigen -----------------------------------------------------------------
# The core library depends on Eigen unconditionally.
set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS      OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        ${MQCES_DEP_EIGEN_COMMIT}
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
        GIT_TAG        ${MQCES_DEP_GTEST_COMMIT}
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
        GIT_TAG        ${MQCES_DEP_BENCHMARK_COMMIT}
        SYSTEM
    )
    FetchContent_MakeAvailable(benchmark)
endif()

# ---- nanobind --------------------------------------------------------------
if(MQCES_ENABLE_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        ${MQCES_DEP_NANOBIND_COMMIT}
        SYSTEM
    )
    FetchContent_MakeAvailable(nanobind)
endif()

# ---- nlohmann/json --------------------------------------------------------
# Used by the CLI and (optionally) other consumers. Header-only.
if(MQCES_ENABLE_CLI)
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install    OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        ${MQCES_DEP_JSON_COMMIT}
        SYSTEM
    )
    FetchContent_MakeAvailable(json)
endif()

# ---- OpenMP (system dep, not FetchContent) ---------------------------------
if(MQCES_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
