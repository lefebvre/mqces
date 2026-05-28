# ----------------------------------------------------------------------------
# ClangTidy.cmake
#
# Wires MQCES_ENABLE_CLANG_TIDY to CMAKE_CXX_CLANG_TIDY so tidy runs as part
# of the normal build. The .clang-tidy file at the project root is the
# source of truth for enabled checks.
# ----------------------------------------------------------------------------

if(NOT MQCES_ENABLE_CLANG_TIDY)
    return()
endif()

find_program(MQCES_CLANG_TIDY_EXE
    NAMES clang-tidy clang-tidy-19 clang-tidy-18 clang-tidy-17
    DOC   "clang-tidy executable")

if(NOT MQCES_CLANG_TIDY_EXE)
    message(FATAL_ERROR
        "MQCES_ENABLE_CLANG_TIDY=ON but clang-tidy was not found on PATH")
endif()

# Skip headers from _deps (FetchContent dependencies) to avoid drowning in
# upstream diagnostics.
set(CMAKE_CXX_CLANG_TIDY
    "${MQCES_CLANG_TIDY_EXE}"
    "--header-filter=${CMAKE_SOURCE_DIR}/mqces/.*"
    "--quiet")

message(STATUS "clang-tidy enabled: ${MQCES_CLANG_TIDY_EXE}")
