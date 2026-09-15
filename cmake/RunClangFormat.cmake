# ----------------------------------------------------------------------------
# Script-mode driver for the `format` and `format-check` targets (see
# ClangFormat.cmake). Run from the source root:
#
#   cmake -DCLANG_FORMAT=<exe> -DGIT=<exe> -DMODE=fix|check -P RunClangFormat.cmake
#
# The pathspecs match the clang-format job in .github/workflows/lint.yml.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24)

foreach(_var CLANG_FORMAT GIT MODE)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "RunClangFormat.cmake: ${_var} is required")
    endif()
endforeach()

execute_process(
    COMMAND "${GIT}" ls-files
        "*.cpp" "*.hpp" "*.h" "*.c"
        ":!tests/data/fixtures_data.*"
        ":!tests/data/expected_scores.*"
    OUTPUT_VARIABLE _files
    RESULT_VARIABLE _git_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _git_result EQUAL 0)
    message(FATAL_ERROR "git ls-files failed (${_git_result})")
endif()
string(REPLACE "\n" ";" _files "${_files}")
list(LENGTH _files _count)

if(_count EQUAL 0)
    message(STATUS "clang-format: no files to process")
    return()
endif()

if(MODE STREQUAL "fix")
    set(_args -i)
elseif(MODE STREQUAL "check")
    set(_args --dry-run --Werror)
else()
    message(FATAL_ERROR "RunClangFormat.cmake: MODE must be fix or check, got '${MODE}'")
endif()

execute_process(
    COMMAND "${CLANG_FORMAT}" ${_args} ${_files}
    RESULT_VARIABLE _result
)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR
        "clang-format reported formatting differences in the files above; "
        "run the `format` target to apply them")
endif()
message(STATUS "clang-format: ${_count} files ${MODE}ed")
