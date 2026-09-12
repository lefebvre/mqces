#pragma once

#include <cmath>
#include <stdexcept>

// Lightweight statistical helpers used by the classifier. Header-only;
// short enough to inline without a dedicated TU.
//
// The Student's t CDF is implemented through the regularized incomplete
// beta function (Numerical Recipes §6.4 continued-fraction form). We need
// it for Eq. 7 with N = 10 (df = 9), where the normal approximation is
// inadequate.

namespace mqces::detail {

// log-gamma via std::lgamma is fine — the C++ standard guarantees correctness.
inline double log_beta(double a, double b) noexcept {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

// Continued-fraction expansion of the incomplete beta function (Lentz's
// method). Pre-condition: 0 < x < 1, a > 0, b > 0.
inline double betacf(double a, double b, double x) {
  constexpr int kMaxIter = 200;
  constexpr double kEps = 3.0e-12;
  constexpr double kTiny = 1.0e-300;

  double qab = a + b;
  double qap = a + 1.0;
  double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kTiny) {
    d = kTiny;
  }
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= kMaxIter; ++m) {
    int m2 = 2 * m;
    double aa = static_cast<double>(m) * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) {
      d = kTiny;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) {
      c = kTiny;
    }
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kTiny) {
      d = kTiny;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kTiny) {
      c = kTiny;
    }
    d = 1.0 / d;
    double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1.0) < kEps) {
      return h;
    }
  }
  throw std::runtime_error("betacf: continued-fraction did not converge");
}

// Regularized incomplete beta function I_x(a, b).
inline double regularized_incomplete_beta(double a, double b, double x) {
  if (x < 0.0 || x > 1.0) {
    throw std::invalid_argument("regularized_incomplete_beta: x must be in [0, 1]");
  }
  if (x == 0.0 || x == 1.0) {
    return x;
  }
  double bt = std::exp(-log_beta(a, b) + a * std::log(x) + b * std::log(1.0 - x));
  // Continued fraction converges quickly in different domains depending on x.
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return bt * betacf(a, b, x) / a;
  }
  return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Student's t CDF: P(T ≤ t) for df degrees of freedom.
//
// Identity:
//   P(T ≤ t) = 1 - 0.5 * I(df/(df + t²), df/2, 1/2)  for t > 0
//   P(T ≤ t) =     0.5 * I(df/(df + t²), df/2, 1/2)  for t < 0
//   P(T ≤ 0) = 0.5
inline double students_t_cdf(double t, double df) {
  if (df <= 0.0) {
    throw std::invalid_argument("students_t_cdf: df must be positive");
  }
  if (t == 0.0) {
    return 0.5;
  }
  double x = df / (df + t * t);
  double tail = 0.5 * regularized_incomplete_beta(0.5 * df, 0.5, x);
  return (t > 0.0) ? (1.0 - tail) : tail;
}

}  // namespace mqces::detail
