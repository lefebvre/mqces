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
# in-tree build.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mqces/__init__.py
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mqces)
