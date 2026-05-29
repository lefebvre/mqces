# Inner solvers

`inverse_spatial_rank` is a fixed-point problem; the inner solver decides
how fast and how accurately the iteration converges. mqces ships three
solvers behind `SolverKind`:

## Weiszfeld (classic, v2)

The original spatial-median iteration:

$$ x^{(k+1)} = \frac{\sum_i y_i / \|x^{(k)} - y_i\|}{\sum_i 1 / \|x^{(k)} - y_i\|}. $$

Linear convergence. Cheap per step. The catch: when an iterate lands
close to one of the $y_i$ vertices, the denominator blows up and the
iteration hits a residual plateau that stalls progress around $10^{-4}$.
Loose tolerances mask the problem in `classic` and `v2`; tight
tolerances expose it.

## Vardi-Zhang (v3)

Vardi & Zhang (2000) add a subgradient correction whenever an iterate
falls within radius `vz_vertex_eps` of any $y_i$. The correction
effectively "passes through" the vertex instead of stalling on it, so
the iteration converges to machine precision regardless of geometry.

Cost per step: same as Weiszfeld plus a vertex-proximity check. The
typical iteration count drops by 2–4× vs Weiszfeld at equivalent
tolerance, so wall time also drops.

`vz_vertex_eps` defaults to `1e-8`; loosen it if the input cloud has
many near-duplicates.

## Vardi-Zhang + Anderson (v4)

Type-II Anderson acceleration (AA) maintains a rolling window of recent
iterates and solves a small least-squares problem to extrapolate the
next step. On well-conditioned problems this converts VZ's linear
convergence into superlinear.

Two safety nets:

- **Tikhonov regularization** (`aa_reg`, default `1e-12`) keeps the
  least-squares problem well-posed when iterates become nearly
  collinear.
- **Toth-Kelley safeguard** (`aa_safeguard = true`, the default) reverts
  to a plain VZ step whenever an accelerated step inflates the
  fixed-point residual. Without this, AA can diverge on pathological
  inputs.

```{note}
AA judges convergence on the **fixed-point residual** $\|G(x) - x\|$,
not on the step length $\|x^{(k+1)} - x^{(k)}\|$. AA's history mixing
makes step length a poor proxy for proximity to the solution.
```

## Comparison

| Solver                   | `SolverKind`            | Typical iters | Best tol  | When to pick                                     |
|--------------------------|-------------------------|---------------|-----------|--------------------------------------------------|
| Weiszfeld (loose)        | `Weiszfeld`             | 50 – 200      | ~10⁻⁴     | `classic`/`v2` reproduction.                     |
| Vardi-Zhang              | `VardiZhang`            | 20 – 80       | ~10⁻⁹     | Reproducible scores, no acceleration.            |
| Vardi-Zhang + Anderson   | `VardiZhangAA`          | 5 – 30        | ~10⁻⁹     | Production scale; fastest of the three.          |

## Configuring

`SolverKind` and `SolverConfig` are defined in `mqces/types.hpp`; see
the {doc}`C++ API reference <../reference/cpp/index>` for the field
listing. Drop a `SolverConfig` into `ClassifierOptions::solver` to
override the variant default:

```cpp
mqces::ClassifierOptions opts;
opts.weights        = weights;
opts.solver.kind    = mqces::SolverKind::VardiZhangAA;
opts.solver.tol     = 1e-9;
opts.solver.max_iters = 80;
opts.solver.aa_window = 7;
```

The variant default is set inside the `classify()` implementation when
`solver.kind` is left at its default; explicit overrides win.
