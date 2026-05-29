# Developer builds

Beyond the standard release build covered in
{doc}`../getting_started/build_cpp`, mqces ships flags for sanitizers,
coverage, and clang-tidy. All are CMake options under the `MQCES_ENABLE_*`
prefix.

## Sanitizers

`MQCES_ENABLE_SANITIZER` is a single-value option; pick one per build
directory:

```bash
cmake -B build-asan  -DMQCES_ENABLE_SANITIZER=asan
cmake -B build-ubsan -DMQCES_ENABLE_SANITIZER=ubsan
cmake -B build-tsan  -DMQCES_ENABLE_SANITIZER=tsan
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

CI runs all three on every push via
[.github/workflows/sanitizers.yml](https://github.com/lefebvre/mqces/blob/master/.github/workflows/sanitizers.yml).

## Coverage

```bash
cmake -B build-cov -DMQCES_ENABLE_COVERAGE=ON
cmake --build build-cov -j
ctest --test-dir build-cov --output-on-failure
gcovr -r . -e 'build/_deps/*' --html --html-details -o coverage.html
```

The
[.github/workflows/coverage.yml](https://github.com/lefebvre/mqces/blob/master/.github/workflows/coverage.yml)
workflow runs this on every push and uploads the HTML report as an
artifact.

## clang-tidy

```bash
cmake -B build-tidy -DMQCES_ENABLE_CLANG_TIDY=ON
cmake --build build-tidy -j
```

The check set is governed by `.clang-tidy` at the repo root. Adding new
checks: extend the `Checks:` glob and run a full build to surface
existing violations before pushing.

## Format

`clang-format` is enforced by
[.github/workflows/lint.yml](https://github.com/lefebvre/mqces/blob/master/.github/workflows/lint.yml).
The style is defined in `.clang-format`; run it locally with:

```bash
find mqces mqces_apps tests benchmarks -name '*.hpp' -o -name '*.cpp' \
  | xargs clang-format -i
```

## Building the docs locally

```bash
cmake -B build -DMQCES_ENABLE_DOCS=ON
cmake --build build --target docs
xdg-open build/docs/html/index.html
```

The `docs` target runs Doxygen → XML, then `sphinx-build` → HTML. See
the {doc}`testing` and {doc}`benchmarks` pages for the surrounding test
and benchmark workflows.
