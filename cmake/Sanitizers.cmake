# ----------------------------------------------------------------------------
# Sanitizers.cmake
#
# Wires MQCES_ENABLE_SANITIZER ∈ {none|asan|ubsan|tsan} to compile/link
# flags applied globally via add_compile_options / add_link_options.
# ----------------------------------------------------------------------------

if(MQCES_ENABLE_SANITIZER STREQUAL "none")
    return()
endif()

if(MSVC)
    if(MQCES_ENABLE_SANITIZER STREQUAL "asan")
        add_compile_options(/fsanitize=address)
    else()
        message(FATAL_ERROR
            "MQCES_ENABLE_SANITIZER=${MQCES_ENABLE_SANITIZER} not supported on MSVC")
    endif()
    return()
endif()

set(_mqces_common_san_flags -fno-omit-frame-pointer -g)

if(MQCES_ENABLE_SANITIZER STREQUAL "asan")
    add_compile_options(${_mqces_common_san_flags} -fsanitize=address)
    add_link_options(-fsanitize=address)
elseif(MQCES_ENABLE_SANITIZER STREQUAL "ubsan")
    add_compile_options(${_mqces_common_san_flags}
        -fsanitize=undefined -fno-sanitize-recover=undefined)
    add_link_options(-fsanitize=undefined)
elseif(MQCES_ENABLE_SANITIZER STREQUAL "tsan")
    add_compile_options(${_mqces_common_san_flags} -fsanitize=thread)
    add_link_options(-fsanitize=thread)
else()
    message(FATAL_ERROR
        "MQCES_ENABLE_SANITIZER='${MQCES_ENABLE_SANITIZER}' is invalid; "
        "expected one of: none asan ubsan tsan")
endif()

message(STATUS "Sanitizer enabled: ${MQCES_ENABLE_SANITIZER}")
