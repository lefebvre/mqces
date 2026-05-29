# `spatial_rank` (Eq. 1) and inverse

## Forward: spatial rank

For a cloud $X \in \mathbb{R}^{N \times d}$, the spatial rank of row $j$ is
the average unit-vector offset from every other row:

$$ U_j = \frac{1}{N} \sum_{i \ne j} \frac{x_j - x_i}{\|x_j - x_i\|}. $$

$U_j$ lies inside the closed unit ball. It's 0 at the cloud's geometric
median and approaches the unit sphere as $x_j$ moves out to the
periphery.

Coincident points (distance below `eps`) are skipped — their contribution
would be NaN otherwise.

Full signatures are in the {doc}`C++ API reference <../reference/cpp/index>`
under `mqces/quantile.hpp`.

## Inverse: solve for the point that produces a given rank

Given a target rank vector $u$ and a cloud $Y$, find the point $\tilde{x}$
whose spatial rank against $Y$ equals $u$. This is the fixed-point problem
the inner solver attacks:

$$ \tilde{x} = \frac{\sum_{i} y_i / \|\tilde{x} - y_i\|}{\sum_{i} 1 / \|\tilde{x} - y_i\|} + u \cdot \left( \sum_i \frac{1}{\|\tilde{x} - y_i\|} \right)^{-1}. $$

The plain Weiszfeld iteration is the unweighted version; Vardi-Zhang
adds a subgradient correction at the vertices $y_i$ (without it,
iterates near a vertex hit a residual plateau). See {doc}`solvers` for
the full comparison.

The reconstruction returns an `InverseRankResult` carrying both the
solved point cloud and convergence diagnostics (max iterations, max
residual, Anderson-acceleration counters). See the
{doc}`C++ API reference <../reference/cpp/index>` for the full struct.

## Costs

| Operation                            | Exact          | Subsample (R)        |
|--------------------------------------|----------------|----------------------|
| `spatial_rank(X)`                    | $O(N^2 d)$     | $O(N R d)$           |
| `inverse_spatial_rank(u, Y)` per row | $O(N d)$ per iter | $O(R d)$ per iter |

For per-call totals, multiply the inverse cost by the number of iterations
the solver takes — typically 20–200 for Weiszfeld, 10–50 for Vardi-Zhang,
5–20 for VZ + Anderson.
