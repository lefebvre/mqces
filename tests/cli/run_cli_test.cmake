# ----------------------------------------------------------------------------
# End-to-end test of the mqces CLI, run in CMake script mode:
#
#   cmake -DMQCES_CLI=<path> -DINPUT=<json> -DWORK_DIR=<dir> -P run_cli_test.cmake
#
# Checks the output schema and the chosen class on a small well-separated
# input, the v2 variant, and that invalid input, bad usage and an unwritable
# output path all exit non-zero.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.24)

foreach(_var MQCES_CLI INPUT WORK_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "run_cli_test.cmake: ${_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")
file(READ "${INPUT}" _input)

function(expect_equal actual expected what)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${what}: expected '${expected}', got '${actual}'")
    endif()
endfunction()

# run_cli(<expected exit code> <stdout var> <stderr var> args...)
function(run_cli expected_code out_var err_var)
    execute_process(
        COMMAND "${MQCES_CLI}" ${ARGN}
        RESULT_VARIABLE _code
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE  _err
    )
    if(NOT "${_code}" STREQUAL "${expected_code}")
        message(FATAL_ERROR
            "mqces ${ARGN}: expected exit ${expected_code}, got ${_code}\n"
            "stdout: ${_out}\nstderr: ${_err}")
    endif()
    set(${out_var} "${_out}" PARENT_SCOPE)
    set(${err_var} "${_err}" PARENT_SCOPE)
endfunction()

# Validates one result document and checks the chosen class and variant.
function(check_result json expected_variant)
    string(JSON _best GET "${json}" best_class)
    expect_equal("${_best}" "B" "best_class")

    string(JSON _variant GET "${json}" variant)
    expect_equal("${_variant}" "${expected_variant}" "variant")

    string(JSON _n_scores LENGTH "${json}" scores)
    expect_equal("${_n_scores}" "3" "number of scores")
    string(JSON _first GET "${json}" scores 0 class_name)
    expect_equal("${_first}" "B" "scores[0].class_name")

    set(_prev "")
    foreach(_i RANGE 2)
        string(JSON _s GET "${json}" scores ${_i} s_xy)
        if(NOT "${_prev}" STREQUAL "" AND _s LESS _prev)
            message(FATAL_ERROR "scores not sorted ascending: ${_prev} then ${_s}")
        endif()
        set(_prev "${_s}")
    endforeach()

    string(JSON _p GET "${json}" misclassification_prob)
    if(_p LESS 0 OR _p GREATER 1)
        message(FATAL_ERROR "misclassification_prob out of [0, 1]: ${_p}")
    endif()

    string(JSON _nota TYPE "${json}" none_of_the_above)
    expect_equal("${_nota}" "BOOLEAN" "none_of_the_above type")
    string(JSON _nota GET "${json}" none_of_the_above)
    expect_equal("${_nota}" "OFF" "none_of_the_above")
endfunction()

# ---- Success: classic, written to a file -----------------------------------
set(_out_file "${WORK_DIR}/result.json")
run_cli(0 _stdout _stderr -o "${_out_file}" "${INPUT}")
file(READ "${_out_file}" _result)
check_result("${_result}" "classic")

# ---- Success: v2, pretty-printed to stdout ---------------------------------
string(JSON _v2_input SET "${_input}" options variant "\"v2\"")
file(WRITE "${WORK_DIR}/v2_input.json" "${_v2_input}")
run_cli(0 _stdout _stderr --pretty "${WORK_DIR}/v2_input.json")
check_result("${_stdout}" "v2")

# ---- Invalid options are rejected ------------------------------------------
string(JSON _bad SET "${_input}" options variant "\"V2\"")
file(WRITE "${WORK_DIR}/bad_variant.json" "${_bad}")
run_cli(1 _stdout _stderr "${WORK_DIR}/bad_variant.json")
if(NOT _stderr MATCHES "options\\.variant")
    message(FATAL_ERROR "bad variant: unexpected stderr: ${_stderr}")
endif()

string(JSON _bad SET "${_input}" options epsilon "-0.5")
file(WRITE "${WORK_DIR}/bad_epsilon.json" "${_bad}")
run_cli(1 _stdout _stderr "${WORK_DIR}/bad_epsilon.json")

file(WRITE "${WORK_DIR}/not_json.json" "{ this is not json")
run_cli(1 _stdout _stderr "${WORK_DIR}/not_json.json")

# ---- Usage errors ------------------------------------------------------------
run_cli(2 _stdout _stderr)
run_cli(2 _stdout _stderr --no-such-flag "${INPUT}")

# ---- Output that cannot be written -----------------------------------------
run_cli(1 _stdout _stderr -o "${WORK_DIR}/missing-dir/result.json" "${INPUT}")
if(EXISTS "/dev/full")
    # Opens fine, fails on write: exercises the post-write stream check.
    run_cli(1 _stdout _stderr -o /dev/full "${INPUT}")
endif()

message(STATUS "mqces CLI test passed")
