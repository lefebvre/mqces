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
    classify_v3.cpp
    classify_v4.cpp
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
        # Wrap the warnings interface in BUILD_INTERFACE so the install
        # export of mqces doesn't try to drag the local target with it.
        $<BUILD_INTERFACE:mqces::warnings>
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

# ---- Install rules (opt-in) ------------------------------------------------
# MQCES_ENABLE_INSTALL is OFF by default because the FetchContent'd Eigen
# isn't installable as part of our build — its target ("eigen") can't be
# re-exported through the mqcesTargets set. When the user opts in, they're
# expected to either (a) build mqces against a system-installed Eigen
# (--DCMAKE_PREFIX_PATH pointing at it) or (b) post-install Eigen via its
# own install command into the same prefix.
if(MQCES_ENABLE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS mqces
        EXPORT      mqcesTargets
        ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(FILES ${MQCES_PUBLIC_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqces)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/mqces/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqces)

    install(EXPORT mqcesTargets
        FILE        mqcesTargets.cmake
        NAMESPACE   mqces::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mqces)

    configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/mqcesConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/mqcesConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mqces
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/mqcesConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mqcesConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mqcesConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mqces)
endif()
