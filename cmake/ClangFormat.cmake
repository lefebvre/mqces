# ----------------------------------------------------------------------------
# ClangFormat.cmake
#
# Adds two manually-invoked targets when clang-format and git are available:
#
#   cmake --build <dir> --target format        # rewrite sources in place
#   cmake --build <dir> --target format-check  # fail on any deviation
#
# Neither is part of `all`. The file set is the one the lint workflow checks:
# tracked C/C++ sources minus the generated test fixtures, listed with
# `git ls-files` when the target runs, so new files are picked up without
# reconfiguring. .clang-format at the project root is the source of truth for
# style.
#
# CI and the pre-commit hook pin clang-format 19; other major versions can
# format differently, so a mismatch is reported at configure time.
# ----------------------------------------------------------------------------

set(MQCES_CLANG_FORMAT_CI_MAJOR 19)

find_program(MQCES_CLANG_FORMAT_EXE
    NAMES clang-format-${MQCES_CLANG_FORMAT_CI_MAJOR} clang-format
    DOC   "clang-format executable used by the format targets")
find_package(Git QUIET)

if(NOT MQCES_CLANG_FORMAT_EXE OR NOT GIT_FOUND)
    message(STATUS "clang-format or git not found; format targets unavailable")
    return()
endif()

execute_process(
    COMMAND "${MQCES_CLANG_FORMAT_EXE}" --version
    OUTPUT_VARIABLE _mqces_clang_format_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "version ([0-9]+)" _ "${_mqces_clang_format_version}")
if(NOT CMAKE_MATCH_1 STREQUAL MQCES_CLANG_FORMAT_CI_MAJOR)
    message(WARNING
        "clang-format ${CMAKE_MATCH_1} found at ${MQCES_CLANG_FORMAT_EXE}; CI uses "
        "${MQCES_CLANG_FORMAT_CI_MAJOR}. `format` may produce changes CI rejects. "
        "Point MQCES_CLANG_FORMAT_EXE at clang-format ${MQCES_CLANG_FORMAT_CI_MAJOR} "
        "(e.g. `pip install clang-format==${MQCES_CLANG_FORMAT_CI_MAJOR}.*`).")
endif()

foreach(_mode IN ITEMS fix check)
    if(_mode STREQUAL "fix")
        set(_target format)
        set(_comment "Formatting C/C++ sources with clang-format")
    else()
        set(_target format-check)
        set(_comment "Checking C/C++ source formatting with clang-format")
    endif()
    add_custom_target(${_target}
        COMMAND "${CMAKE_COMMAND}"
            "-DCLANG_FORMAT=${MQCES_CLANG_FORMAT_EXE}"
            "-DGIT=${GIT_EXECUTABLE}"
            "-DMODE=${_mode}"
            -P "${CMAKE_CURRENT_LIST_DIR}/RunClangFormat.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "${_comment}"
        VERBATIM
        USES_TERMINAL
    )
endforeach()
