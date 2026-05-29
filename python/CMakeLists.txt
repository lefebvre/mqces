# Python bindings via nanobind.
#
# Built when MQCES_ENABLE_PYTHON=ON. Eigen ↔ NumPy conversion uses
# nanobind's bundled <nanobind/eigen/dense.h> headers, which require the
# Eigen target be visible (it is — pulled in by Dependencies.cmake).
#
# nanobind is fetched via FetchContent in cmake/Dependencies.cmake; that
# call exposes `nanobind_add_module()` directly without a separate
# find_package step.

nanobind_add_module(_mqces_core
    NB_STATIC
    bindings.cpp
)
target_link_libraries(_mqces_core
    PRIVATE
        mqces::mqces
        # mqces::warnings intentionally omitted: the bindings TU pulls in
        # nanobind's templates, which trip several of our strict warnings
        # (notably -Wold-style-cast) on Python C-API casts we don't control.
)

# The .so lives inside the Python package directory so that
# `import mqces._mqces_core` works in both editable installs and built
# wheels.
set_target_properties(_mqces_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mqces"
)

# Copy the pure-Python wrapper next to the compiled .so so a developer can
# run `PYTHONPATH=build/python python -m pytest python/tests` against the
# in-tree build. Wired via add_custom_command so edits to the source
# __init__.py trigger a re-copy on the next build (file(COPY ...) only
# runs at configure time and goes stale).
set(_mqces_init_src "${CMAKE_CURRENT_SOURCE_DIR}/mqces/__init__.py")
set(_mqces_init_dst "${CMAKE_CURRENT_BINARY_DIR}/mqces/__init__.py")
add_custom_command(
    OUTPUT  "${_mqces_init_dst}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_mqces_init_src}" "${_mqces_init_dst}"
    DEPENDS "${_mqces_init_src}"
    COMMENT "Syncing python/mqces/__init__.py to build dir"
)
add_custom_target(mqces_python_init ALL DEPENDS "${_mqces_init_dst}")
add_dependencies(_mqces_core mqces_python_init)
