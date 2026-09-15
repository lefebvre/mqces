#include <mqces/types.hpp>
#include <mqces/uncertainty.hpp>

#include <Eigen/Core>
#include <gtest/gtest.h>

using mqces::perturb;
using mqces::Sample;

namespace {

Sample constant_sample(int n, int d, double value) {
  Sample s(n, d);
  s.setConstant(value);
  return s;
}

}  // namespace

// epsilon == 0 is a no-op: output equals input element-wise.
TEST(Uncertainty, ZeroEpsilonIsNoOp) {
  Sample x = constant_sample(10, 3, 2.5);
  Sample y = perturb(x, 0.0, 42);
  EXPECT_EQ(y.rows(), x.rows());
  EXPECT_EQ(y.cols(), x.cols());
  EXPECT_TRUE((y.array() == x.array()).all());
}

// Same seed → same result. Different seed → different result.
TEST(Uncertainty, DeterministicAndSeedDependent) {
  Sample x = constant_sample(10, 3, 1.0);
  Sample a = perturb(x, 0.05, 123);
  Sample b = perturb(x, 0.05, 123);
  Sample c = perturb(x, 0.05, 124);
  EXPECT_TRUE((a.array() == b.array()).all());
  EXPECT_FALSE((a.array() == c.array()).all());
}

// Element-wise perturbation form: y_ij = x_ij * (1 + eps * z),
// so when x is a nonzero constant the empirical mean of (y/x - 1) over many
// independent draws should converge to 0 and the variance to eps^2.
TEST(Uncertainty, EmpiricalMeanAndVarianceMatchModel) {
  constexpr int n = 1000;
  constexpr int d = 1;
  constexpr double val = 7.5;
  constexpr double eps = 0.1;

  Sample x = constant_sample(n, d, val);
  Sample y = perturb(x, eps, 99);
  auto noise = (y.array() / val - 1.0).eval();

  const double mean = noise.mean();
  const double var = (noise - mean).square().sum() / static_cast<double>(noise.size() - 1);

  EXPECT_NEAR(mean, 0.0, 0.02);        // ~1/sqrt(1000) ~ 0.03; loose bound
  EXPECT_NEAR(var, eps * eps, 0.005);  // var(eps * N(0,1)) = eps^2
}

// Zero entries stay zero (multiplicative model: x = 0 → output = 0 regardless
// of the noise draw). Documents a real edge: if an input feature is exactly
// zero, the QC perturbation cannot move it.
TEST(Uncertainty, ZeroEntriesArePreserved) {
  Sample x(3, 2);
  x << 0.0, 1.0, 0.0, 2.0, 0.0, 3.0;
  Sample y = perturb(x, 0.5, 7);
  EXPECT_TRUE((y.col(0).array() == 0.0).all());
  EXPECT_FALSE((y.col(1).array() == x.col(1).array()).all());
}
