# ----------------------------------------------------------------------------
# mqces library target
#
# Sources are listed explicitly (no file(GLOB)) so adding/removing files is
# always an explicit, reviewable change.
# ----------------------------------------------------------------------------

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/mqces/version.hpp"
    @ONLY
)

set(MQCES_PUBLIC_HEADERS
    mqces.hpp
    types.hpp
    quantile.hpp
    score.hpp
    uncertainty.hpp
    classify.hpp
    nota.hpp
)

set(MQCES_SOURCES
    quantile.cpp
    score.cpp
    uncertainty.cpp
    classify_classic.cpp
    classify_v2.cpp
    nota.cpp
)

add_library(mqces ${MQCES_SOURCES})
add_library(mqces::mqces ALIAS mqces)

target_compile_features(mqces PUBLIC cxx_std_20)

target_include_directories(mqces
    PUBLIC
        # In-tree: enable #include <mqces/...> from anywhere.
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        # Generated headers (version.hpp).
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        # Post-install location.
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mqces
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        mqces::warnings
)

if(MQCES_ENABLE_OPENMP AND NOT MQCES_ENABLE_KOKKOS)
    target_link_libraries(mqces PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(mqces PUBLIC MQCES_HAVE_OPENMP=1)
endif()

if(MQCES_ENABLE_KOKKOS)
    target_link_libraries(mqces PUBLIC Kokkos::kokkos)
    target_compile_definitions(mqces PUBLIC MQCES_HAVE_KOKKOS=1)
endif()

if(MQCES_ENABLE_COVERAGE)
    target_compile_options(mqces PRIVATE --coverage -O0 -g)
    target_link_options(mqces    PRIVATE --coverage)
endif()

set_target_properties(mqces PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Install / find_package(mqces) support is deferred to a later tranche:
# exporting through FetchContent'd Eigen + an INTERFACE warnings target needs
# a hand-written mqcesConfig.cmake.in with find_dependency(Eigen3). Not
# needed for the in-tree test cycle.
