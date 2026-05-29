# `similarity_score` (Eq. 3)

The similarity score is the single number `classify()` aggregates over
Monte-Carlo replicates and uses to rank classes. Every variant —
`classic`, `v2`, `v3`, `v4` — ultimately calls this primitive.

Header: [mqces/score.hpp](https://github.com/lefebvre/mqces/blob/master/mqces/score.hpp).

## Definition

For two samples $X \in \mathbb{R}^{N_X \times d}$ and
$Y \in \mathbb{R}^{N_Y \times d}$ with diagonal feature-weight matrix
$W = \mathrm{diag}(w_0, \ldots, w_{d-1})$:

$$ S_{XY} = \sum_{i=1}^{N_X} (\tilde{x}_i - x_i)^\top W (\tilde{x}_i - x_i)
         + \sum_{j=1}^{N_Y} (\tilde{y}_j - y_j)^\top W (\tilde{y}_j - y_j). $$

Here $\tilde{x}_i$ is the **inverse-spatial-rank reconstruction** of $x_i$:
the point whose spatial rank against the *Y* cloud equals the spatial rank
of $x_i$ against the *X* cloud. $\tilde{y}_j$ is the symmetric
reconstruction of $y_j$ against $X$.

The score is symmetric in $X$ and $Y$ and non-negative. $S_{XY} = 0$ means
the two clouds are indistinguishable under the rank metric.

## Intuition

Spatial rank summarizes a point's position in its cloud as a unit-vector
sum (Eq. 1). The inverse rank asks: *if I drop $x_i$ into the* $Y$
*cloud and demand the same position, where does it land?* If the two
clouds have the same quantile structure, $\tilde{x}_i \approx x_i$ and
the squared distance is small. If they don't, $\tilde{x}_i$ drifts away
from $x_i$ and the score grows.

The diagonal $W$ lets callers weight features unequally — useful when
some isotope ratios carry more discriminative power than others.

## C++ overloads

Three overloads expose progressively more control. Full signatures are
in the {doc}`C++ API reference <../reference/cpp/index>` under
`mqces/score.hpp`.

- **Single-argument** — uses an internal `SolverConfig` tuned for the
  score's accumulated-error budget (loose tolerance, generous iteration
  cap). Use when you just want the score.
- **With `SolverConfig`** — caller overrides the inner solver. Used by
  `mqces::v3` (Vardi-Zhang) and the v3-equivalent variants.
- **With `SolverConfig` + `SamplingConfig`** — full surface. Caller
  controls both the inner solver and the reference-subset sampling.
  Used by `mqces::v4` and any caller that needs the O(N·R) approximation
  end-to-end.

```{important}
In the three-argument overload, the same `SamplingConfig` drives **both**
`spatial_rank(y, sampling)` and `inverse_spatial_rank(_, y, _, sampling)`,
so the R-subset of each cloud is consistent across the two reconstruction
directions.
```

## Errors

Throws `std::invalid_argument` if:

- `weights.size() != x.cols()`, or
- `x.cols() != y.cols()` (different feature counts between the two
  samples).

## Cost

Per call, in terms of $N = \max(N_X, N_Y)$, $d$, and the configured
sampling reference size $R$:

| Path                     | Cost            |
|--------------------------|-----------------|
| Exact (`R = 0`)          | $O(N^2 \cdot d)$ |
| Subsample (`R > 0`)      | $O(N \cdot R \cdot d)$ |

At $N = 10^6$ and $R = 10^4$ the subsample path is roughly 100× faster.
See {doc}`../guide/sampling` for guidance on picking $R$.

## Tolerance choice and the inner solver

`similarity_score` accumulates squared distances across $N$ rows, so the
inner solver's per-row error budget is *not* the score's error budget — it
scales as $\sqrt{N}$ in the worst case. The single-argument overload
picks tolerances accordingly. If you override the solver via the
two/three-argument forms, choose `solver.tol` with the accumulation in
mind:

| $N$            | Recommended `solver.tol`  |
|----------------|---------------------------|
| $10^3$         | $10^{-9}$                 |
| $10^4$         | $10^{-8}$                 |
| $10^5$         | $10^{-7}$                 |
| $10^6$         | $10^{-7}$ with VZ + AA    |

See {doc}`solvers` for the comparison between Weiszfeld, Vardi-Zhang, and
the Anderson-accelerated variant.

## Python equivalent

```python
import mqces
s = mqces.similarity_score(x, y, weights)
```

The Python binding exposes only the single-argument form; the
solver/sampling overrides are reached through the higher-level
{py:func}`mqces.classify` entry point with `variant="v3"` or `"v4"` and a
`reference_size` argument.

## See also

- {doc}`spatial_rank` — the forward + inverse rank primitive `similarity_score` calls.
- {doc}`solvers` — solver trade-offs (Weiszfeld vs Vardi-Zhang vs VZ + AA).
- {doc}`../guide/sampling` — picking `reference_size`.
