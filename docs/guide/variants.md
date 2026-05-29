# Choosing a variant

mqces ships four classifier variants in parallel. They differ along two
axes — *t-statistic definition* and *inner solver* — but share the same
input/output shape, so swapping between them is a one-token change.

## Decision matrix

| Variant            | t-statistic              | Inner solver                | Tolerance | Notes                                                     |
|--------------------|--------------------------|-----------------------------|-----------|-----------------------------------------------------------|
| `mqces::classic`   | independent-samples Eq. 6 | Weiszfeld (loose)           | ~1e-4     | Paper-faithful reference; bit-exact reproduction goal.    |
| `mqces::v2`        | paired-difference         | Weiszfeld (loose)           | ~1e-4     | Corrected statistics, same solver.                        |
| `mqces::v3`        | paired-difference         | Vardi-Zhang                 | ~1e-7     | Closes Weiszfeld's vertex-plateau pathology.              |
| `mqces::v4`        | paired-difference         | VZ + Type-II Anderson       | ~1e-7     | Production target at N = 10⁶ scale; fastest of the four.  |

## Picking one

**Choose `classic`** when reproducing the published method exactly is the
goal. Its statistics match Weber & Dayman Eqs. 5-7 verbatim; bit-exact
agreement with the reference NumPy implementation is part of the test
suite.

**Choose `v2`** when the corrected paired-difference t-statistic matters
but you don't want any solver change relative to `classic`. The scores
themselves are identical to `classic` for the same RNG seed.

**Choose `v3`** when reproducibility at tight tolerance matters — for
example, when comparing across machine architectures or compiler
versions. The Vardi-Zhang subgradient correction at vertices closes the
residual plateau that forces v1/v2 to a loose Weiszfeld tolerance.

**Choose `v4`** for production-scale runs. Type-II Anderson acceleration
over a rolling window converts VZ's linear convergence into superlinear
when the problem is amenable; a safeguarded fallback (Toth-Kelley 2015)
reverts to plain VZ whenever an accelerated step inflates the residual,
so divergence is impossible.

## Switching

```{eval-rst}
.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            #include <mqces/classify.hpp>

            auto result = mqces::v4::classify(test, classes, options);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            result = mqces.classify(test, classes, weights, variant="v4")

    .. tab-item:: CLI
        :sync: cli

        .. code-block:: bash

            mqces --variant v4 input.json
```

All four variants honor the same {doc}`SamplingConfig <sampling>`,
{doc}`UncertaintyConfig <uncertainty>`, and {doc}`NOTA threshold <nota>`.
