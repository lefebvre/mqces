# ----------------------------------------------------------------------------
# tests/
#
# Each test file becomes a single executable. gtest_discover_tests then
# registers every TEST() within it with ctest, so individual cases are
# selectable via -R and reported separately.
# ----------------------------------------------------------------------------

include(GoogleTest)

set(MQCES_UNIT_TESTS
    unit/test_spatial_rank.cpp
    unit/test_inverse_rank.cpp
    unit/test_score.cpp
    unit/test_uncertainty.cpp
    unit/test_parallel_shim.cpp
    unit/test_classify_classic.cpp
    unit/test_classify_v2.cpp
    unit/test_nota.cpp
)

foreach(_src IN LISTS MQCES_UNIT_TESTS)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(${_name} ${_src})
    target_link_libraries(${_name}
        PRIVATE
            mqces::mqces
            mqces::warnings
            GTest::gtest_main
    )
    gtest_discover_tests(${_name}
        DISCOVERY_TIMEOUT 30
        PROPERTIES LABELS unit
    )
endforeach()
