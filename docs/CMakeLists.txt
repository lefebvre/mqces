# Build the Sphinx + Breathe + Doxygen documentation site.
#
# Gated by MQCES_ENABLE_DOCS at the top level. This file is only added
# when that option is ON, so requiring Doxygen / Python here is safe.

find_package(Doxygen REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MQCES_DOXYGEN_INPUT_DIR  "${CMAKE_SOURCE_DIR}/mqces")
set(MQCES_DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(MQCES_DOXYGEN_XML_DIR    "${MQCES_DOXYGEN_OUTPUT_DIR}/xml")
set(MQCES_VERSION            "${PROJECT_VERSION}")

set(_doxyfile_in  "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
set(_doxyfile_out "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
configure_file("${_doxyfile_in}" "${_doxyfile_out}")

file(MAKE_DIRECTORY "${MQCES_DOXYGEN_OUTPUT_DIR}")

add_custom_command(
    OUTPUT "${MQCES_DOXYGEN_XML_DIR}/index.xml"
    COMMAND ${DOXYGEN_EXECUTABLE} "${_doxyfile_out}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${_doxyfile_out}"
    COMMENT "Generating Doxygen XML"
    VERBATIM
)
add_custom_target(mqces_doxygen DEPENDS "${MQCES_DOXYGEN_XML_DIR}/index.xml")

set(MQCES_SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")
set(MQCES_SPHINX_BUILD  "${CMAKE_CURRENT_BINARY_DIR}/html")

add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E env
        "MQCES_DOXYGEN_XML_DIR=${MQCES_DOXYGEN_XML_DIR}"
        ${Python3_EXECUTABLE} -m sphinx -b html
            "${MQCES_SPHINX_SOURCE}" "${MQCES_SPHINX_BUILD}"
    DEPENDS mqces_doxygen
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building Sphinx HTML docs at ${MQCES_SPHINX_BUILD}"
    VERBATIM
)

message(STATUS "  Docs (Sphinx+Doxygen) . ON   -> build with: cmake --build build --target docs")
