# mqces

**M**ultivariate **Q**uantile **C**omparison for **E**nvironmental
**S**amples — a C++20 reimplementation and extension of the classification
method of Weber & Dayman (ORNL).

mqces takes a *test* sample (a cloud of multivariate measurements) and a set
of *known classes* (each itself a cloud), and returns the class whose
quantile structure most closely matches the test sample, along with a
misclassification probability and a none-of-the-above flag.

```{toctree}
:caption: Getting started
:maxdepth: 1

getting_started/build_cpp
getting_started/install_python
getting_started/cli_quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 1

guide/variants
guide/sampling
guide/uncertainty
guide/nota
```

```{toctree}
:caption: Algorithm
:maxdepth: 1

algorithm/overview
algorithm/similarity_score
algorithm/spatial_rank
algorithm/solvers
```

```{toctree}
:caption: API reference
:maxdepth: 1

reference/cpp/index
reference/python/index
reference/cli
```

```{toctree}
:caption: Development
:maxdepth: 1

development/building
development/testing
development/benchmarks
```

## At a glance

mqces ships **four classifier variants** along two orthogonal axes:

| Variant            | t-statistic              | Inner solver                | Use when                                                                |
|--------------------|--------------------------|-----------------------------|-------------------------------------------------------------------------|
| `mqces::classic`   | independent-samples      | Weiszfeld (loose)           | Reproducing the published paper exactly.                                |
| `mqces::v2`        | paired-difference        | Weiszfeld (loose)           | You want corrected statistics with no other changes.                    |
| `mqces::v3`        | paired-difference        | **Vardi-Zhang**             | You need reproducible scores at `~1e-7` tolerance.                      |
| `mqces::v4`        | paired-difference        | **VZ + Anderson**           | Production scale (N ≈ 10⁶); fastest of the four.                        |

A `SamplingConfig` with `reference_size = R` is honored by every variant.
When `R > 0`, both `spatial_rank` and `inverse_spatial_rank` switch from
O(N²) to O(N·R) cost using a deterministic uniform subsample. At N = 10⁶
and R = 10⁴ this delivers the ~100× wall-time win that makes million-row
inputs tractable.

## Citation

> C. Weber and K. Dayman, *Using Quantile Comparisons to Classify
> Environmental Samples*, Oak Ridge National Laboratory.

The paper PDF is **not committed** to the repository. mqces is a
clean-room reimplementation; no code or data from the original work is
incorporated.
