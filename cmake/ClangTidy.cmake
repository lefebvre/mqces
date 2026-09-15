# ----------------------------------------------------------------------------
# ClangTidy.cmake
#
# When MQCES_ENABLE_CLANG_TIDY is ON, mqces_target_clang_tidy(<target>) makes
# clang-tidy run as part of compiling that target. It is applied to the
# library and the CLI only: tests, benchmarks and FetchContent dependencies
# don't ride the production bar, and .clang-tidy promotes every finding to an
# error. The .clang-tidy file at the project root is the source of truth for
# enabled checks and the header filter.
# ----------------------------------------------------------------------------

if(MQCES_ENABLE_CLANG_TIDY)
    find_program(MQCES_CLANG_TIDY_EXE
        NAMES clang-tidy clang-tidy-19 clang-tidy-18 clang-tidy-17
        DOC   "clang-tidy executable")

    if(NOT MQCES_CLANG_TIDY_EXE)
        message(FATAL_ERROR
            "MQCES_ENABLE_CLANG_TIDY=ON but clang-tidy was not found on PATH")
    endif()
    message(STATUS "clang-tidy enabled: ${MQCES_CLANG_TIDY_EXE}")
endif()

function(mqces_target_clang_tidy target)
    if(MQCES_ENABLE_CLANG_TIDY)
        set_target_properties(${target} PROPERTIES
            CXX_CLANG_TIDY "${MQCES_CLANG_TIDY_EXE};--quiet")
    endif()
endfunction()
