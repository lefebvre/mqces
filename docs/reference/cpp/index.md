# C++ API reference

This page collects the full public C++ surface of mqces. All symbols
live in the `mqces` namespace (variant entry points are in nested
namespaces: `mqces::classic`, `mqces::v2`, `mqces::v3`, `mqces::v4`).

The umbrella include is:

```cpp
#include <mqces/mqces.hpp>
```

## Types — `mqces/types.hpp`

```{eval-rst}
.. doxygentypedef:: mqces::Sample
   :project: mqces

.. doxygentypedef:: mqces::FeatureWeights
   :project: mqces

.. doxygenstruct:: mqces::Class
   :project: mqces
   :members:

.. doxygenstruct:: mqces::Score
   :project: mqces
   :members:

.. doxygenstruct:: mqces::ClassificationResult
   :project: mqces
   :members:

.. doxygenstruct:: mqces::UncertaintyConfig
   :project: mqces
   :members:

.. doxygenenum:: mqces::SolverKind
   :project: mqces

.. doxygenstruct:: mqces::SolverConfig
   :project: mqces
   :members:

.. doxygenstruct:: mqces::SamplingConfig
   :project: mqces
   :members:

.. doxygenstruct:: mqces::ClassifierOptions
   :project: mqces
   :members:
```

## Classification — `mqces/classify.hpp`

```{eval-rst}
.. doxygenfunction:: mqces::classic::classify
   :project: mqces

.. doxygenfunction:: mqces::v2::classify
   :project: mqces

.. doxygenfunction:: mqces::v3::classify
   :project: mqces

.. doxygenfunction:: mqces::v4::classify
   :project: mqces
```

## Score — `mqces/score.hpp`

See {doc}`../../algorithm/similarity_score` for the equation and
overload guidance.

```{eval-rst}
.. doxygenfunction:: mqces::similarity_score(const Sample&, const Sample&, const FeatureWeights&)
   :project: mqces

.. doxygenfunction:: mqces::similarity_score(const Sample&, const Sample&, const FeatureWeights&, const SolverConfig&)
   :project: mqces

.. doxygenfunction:: mqces::similarity_score(const Sample&, const Sample&, const FeatureWeights&, const SolverConfig&, const SamplingConfig&)
   :project: mqces
```

## Spatial rank — `mqces/quantile.hpp`

```{eval-rst}
.. doxygenfunction:: mqces::spatial_rank(const Sample&, double)
   :project: mqces

.. doxygenfunction:: mqces::spatial_rank(const Sample&, const SamplingConfig&, double)
   :project: mqces

.. doxygenstruct:: mqces::InverseRankResult
   :project: mqces
   :members:

.. doxygenfunction:: mqces::inverse_spatial_rank(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&, const Sample&, const SolverConfig&)
   :project: mqces

.. doxygenfunction:: mqces::inverse_spatial_rank(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&, const Sample&, const SolverConfig&, const SamplingConfig&)
   :project: mqces

.. doxygenfunction:: mqces::inverse_spatial_rank(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&, const Sample&, double, std::size_t)
   :project: mqces
```

## Uncertainty — `mqces/uncertainty.hpp`

```{eval-rst}
.. doxygenfunction:: mqces::perturb
   :project: mqces
```

## None-of-the-above — `mqces/nota.hpp`

```{eval-rst}
.. doxygenfunction:: mqces::is_none_of_the_above
   :project: mqces
```
