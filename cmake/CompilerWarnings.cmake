# ----------------------------------------------------------------------------
# CompilerWarnings.cmake
#
# Provides an INTERFACE target `mqces_warnings` carrying the project's
# warning flags. Library and app targets link against it PRIVATE-ly so the
# flags don't leak to downstream consumers.
# ----------------------------------------------------------------------------

add_library(mqces_warnings INTERFACE)
add_library(mqces::warnings ALIAS mqces_warnings)

set(_mqces_gnu_warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    # -Wnull-dereference is intentionally omitted: GCC 14 fires false
    # positives inside Eigen's inlined AVX intrinsics (see e.g. redux_impl
    # and findCoeff). The warning is low-signal in general C++ code and
    # specifically broken with the Eigen template surface mqces uses.
)

set(_mqces_gcc_extra
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
)

set(_mqces_msvc_warnings
    /W4
    /permissive-
    /w14242 /w14254 /w14263 /w14265 /w14287 /we4289
    /w14296 /w14311 /w14545 /w14546 /w14547 /w14549
    /w14555 /w14619 /w14640 /w14826 /w14905 /w14906 /w14928
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(mqces_warnings INTERFACE ${_mqces_msvc_warnings})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mqces_warnings INTERFACE ${_mqces_gnu_warnings})
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(mqces_warnings INTERFACE
        ${_mqces_gnu_warnings} ${_mqces_gcc_extra})
endif()

# Warnings are errors when MQCES_WERROR is ON; every CI build job sets it.
if(MQCES_WERROR)
    if(MSVC)
        target_compile_options(mqces_warnings INTERFACE /WX)
    else()
        target_compile_options(mqces_warnings INTERFACE -Werror)
    endif()
endif()
