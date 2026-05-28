#include <mqces/uncertainty.hpp>

#include <mqces/detail/rng.hpp>

#include <random>

namespace mqces {

Sample perturb(const Sample& x0, double epsilon, std::uint64_t seed)
{
    if (epsilon <= 0.0) {
        return x0;
    }
    auto                       rng = detail::make_stream(seed, 0);
    std::normal_distribution<> n01(0.0, 1.0);
    Sample                     out(x0.rows(), x0.cols());
    for (Eigen::Index i = 0; i < x0.rows(); ++i) {
        for (Eigen::Index j = 0; j < x0.cols(); ++j) {
            out(i, j) = x0(i, j) * (1.0 + epsilon * n01(rng));
        }
    }
    return out;
}

}  // namespace mqces
