# Reference-subset sampling

The bottleneck in mqces is `spatial_rank` and its inverse, both of which
naively cost O(N²) per cloud. For N = 10⁶ that's prohibitive. `SamplingConfig`
swaps the inner sums to an O(N·R) approximation by drawing a deterministic
uniform subsample of size `R = reference_size` from each cloud.

## How it works

For each row `j`, the exact spatial rank is

$$ U_j = \frac{1}{N} \sum_{i \ne j} \frac{x_j - x_i}{\|x_j - x_i\|}. $$

When `reference_size = R` with $R < N$, the sum runs over an
$R$-element uniform subset $\text{Refs}$ instead of all $N$ rows:

$$ U_j \approx \frac{1}{R} \sum_{i \in \text{Refs},\, i \ne j} \frac{x_j - x_i}{\|x_j - x_i\|}. $$

The estimator is unbiased; its per-coordinate variance scales as $O(1/R)$.
The same subset is reused for `inverse_spatial_rank` so both directions of
the score reconstruction stay consistent.

## Picking R

| N                 | R           | Approximate cost ratio | Notes                                                                |
|-------------------|-------------|------------------------|----------------------------------------------------------------------|
| 10³               | 0 (exact)   | 1.0                    | Exact path is already cheap; subsampling adds variance for no win.   |
| 10⁴               | 10³ – 10⁴   | 0.1 – 1.0              | Optional; profile and decide.                                        |
| 10⁵               | 5·10³       | ~0.05                  | Sweet spot for many problems.                                        |
| 10⁶               | 10⁴         | ~0.01                  | The ~100× win that makes million-row inputs tractable.               |

`reference_size = 0` is the special "exact" sentinel — both kernels
dispatch back to the O(N²) path. `reference_size >= N` also dispatches to
exact (no point subsampling more than the population).

## Configuring

```{eval-rst}
.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. code-block:: cpp

            mqces::ClassifierOptions opts;
            opts.weights        = weights;
            opts.sampling.reference_size = 10000;       // R
            opts.sampling.strategy       = mqces::SamplingConfig::Strategy::Uniform;
            opts.sampling.seed           = 0xACEBEEF;   // determinism
            auto result = mqces::v4::classify(test, classes, opts);

    .. tab-item:: Python
        :sync: python

        .. code-block:: python

            result = mqces.classify(
                test, classes, weights,
                variant="v4",
                reference_size=10000,
                sampling_seed=0xACEBEEF,
            )

    .. tab-item:: CLI
        :sync: cli

        .. code-block:: bash

            mqces --variant v4 --reference-size 10000 input.json
```

## Sampling strategies

`SamplingConfig::strategy` selects how the subset is drawn:

- **`Uniform`** *(default)* — random R-subset per cloud, O(N·R) cost. The
  recommended production strategy; combines well with Vardi-Zhang +
  Anderson to give an ~20× wall-time win at the N = 2k proxy scale vs
  exact Weiszfeld.
- **`Stratified`** — reserved for a future refinement; currently behaves
  as `Uniform`.
- **`KdTreeLocalExact`** — Barnes-Hut traversal over a balanced k-d tree
  of the cloud. Nearby points contribute exactly; subtrees that subtend a
  small angle from the query (controlled by `kd_opening_theta`)
  contribute via their centroid + count (multipole-zero approximation).

```{warning}
The `KdTreeLocalExact` strategy is **production-safe only for
`spatial_rank`**. For `inverse_spatial_rank`, the Barnes-Hut approximation
introduces a per-step bias in the Weiszfeld fixed-point map: at tight
`kd_opening_theta` it matches the matrix path to machine precision but
pays traversal overhead, and at loose `kd_opening_theta` the iteration
stalls and throws non-convergence. Treat it as a correctness-equivalent
of the exact path until a contractivity-preserving variant lands.
```
