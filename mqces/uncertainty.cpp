#include <mqces/uncertainty.hpp>

#include <mqces/detail/rng.hpp>

namespace mqces {

Sample perturb(const Sample& x0, double epsilon, std::uint64_t seed)
{
    if (epsilon <= 0.0) {
        return x0;
    }
    auto   rng = detail::make_stream(seed, 0);
    Sample out(x0.rows(), x0.cols());
    for (Eigen::Index i = 0; i < x0.rows(); ++i) {
        for (Eigen::Index j = 0; j < x0.cols(); ++j) {
            out(i, j) = x0(i, j) * (1.0 + epsilon * detail::standard_normal(rng));
        }
    }
    return out;
}

}  // namespace mqces
