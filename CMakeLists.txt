cmake_minimum_required(VERSION 3.24)

project(mqces
    VERSION 0.1.0
    DESCRIPTION "Multivariate Quantile Comparison for Environmental Sample classification"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Options ---------------------------------------------------------------
option(MQCES_ENABLE_TESTS       "Build unit/integration/property tests"   ON)
option(MQCES_ENABLE_BENCHMARKS  "Build Google Benchmark microbenchmarks"  OFF)
option(MQCES_ENABLE_PYTHON      "Build nanobind Python module"            OFF)
option(MQCES_ENABLE_CLI         "Build mqces command-line interface"      ON)
option(MQCES_ENABLE_OPENMP      "Enable OpenMP parallel backend"          ON)
option(MQCES_ENABLE_COVERAGE    "Instrument for gcovr / llvm-cov"         OFF)
option(MQCES_ENABLE_CLANG_TIDY  "Run clang-tidy at compile time"          OFF)
option(MQCES_ENABLE_INSTALL     "Emit install + export rules (mqcesConfig.cmake); requires system-installed Eigen3" OFF)
option(MQCES_WERROR             "Treat compiler warnings as errors (CI sets this)" OFF)

set(MQCES_ENABLE_SANITIZER "none" CACHE STRING "Sanitizer to enable: none|asan|ubsan|tsan")
set_property(CACHE MQCES_ENABLE_SANITIZER PROPERTY STRINGS none asan ubsan tsan)

# ---- Helper modules --------------------------------------------------------
include(CompilerWarnings)
include(Sanitizers)
include(ClangTidy)
include(Dependencies)

# ---- Subdirectories (added as they land in later tranches) -----------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mqces/CMakeLists.txt")
    add_subdirectory(mqces)
endif()

if(MQCES_ENABLE_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mqces_apps/CMakeLists.txt")
    add_subdirectory(mqces_apps)
endif()

if(MQCES_ENABLE_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

if(MQCES_ENABLE_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

if(MQCES_ENABLE_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
    add_subdirectory(python)
endif()

# ---- Summary ---------------------------------------------------------------
message(STATUS "")
message(STATUS "mqces ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type ............ ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard .......... ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests ................. ${MQCES_ENABLE_TESTS}")
message(STATUS "  Benchmarks ............ ${MQCES_ENABLE_BENCHMARKS}")
message(STATUS "  Python bindings ....... ${MQCES_ENABLE_PYTHON}")
message(STATUS "  CLI ................... ${MQCES_ENABLE_CLI}")
message(STATUS "  OpenMP backend ........ ${MQCES_ENABLE_OPENMP}")
message(STATUS "  Sanitizer ............. ${MQCES_ENABLE_SANITIZER}")
message(STATUS "  Coverage .............. ${MQCES_ENABLE_COVERAGE}")
message(STATUS "  clang-tidy ............ ${MQCES_ENABLE_CLANG_TIDY}")
message(STATUS "  Install rules ......... ${MQCES_ENABLE_INSTALL}")
message(STATUS "")
