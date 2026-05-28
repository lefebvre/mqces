// nanobind bindings for the mqces public API.
//
// Exposes a single `_mqces_core` extension module with two free functions:
//
//   classify_classic(test, classes, weights, **opts) -> ClassificationResult
//   classify_v2(test, classes, weights, **opts) -> ClassificationResult
//
// plus three primitives (spatial_rank, similarity_score, perturb) that the
// pytest suite uses to verify the round-trip with the Python helpers in
// tests/data/generate_fixtures.py.
//
// NumPy arrays are passed as nb::ndarray<double, nb::c_contig> and copied
// into Eigen matrices. The fixture sizes are small (≤ 19 × 50 × 9 doubles)
// so the copy is negligible; a zero-copy Eigen::Map binding can be added
// later if a profiler ever flags it.

#include <mqces/classify.hpp>
#include <mqces/quantile.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>
#include <mqces/uncertainty.hpp>
#include <mqces/version.hpp>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

// Convert a Python list of (name: str, specimens: ndarray) tuples into
// std::vector<mqces::Class>. The ndarray is copied into a row-major Eigen
// matrix to satisfy mqces::Sample's storage order.
std::vector<mqces::Class> to_classes(
    const std::vector<std::pair<std::string, nb::ndarray<double, nb::c_contig>>>& py_classes)
{
    std::vector<mqces::Class> out;
    out.reserve(py_classes.size());
    for (const auto& [name, arr] : py_classes) {
        if (arr.ndim() != 2) {
            throw std::invalid_argument(
                "class '" + name + "' specimens must be a 2-D array");
        }
        const auto n = static_cast<Eigen::Index>(arr.shape(0));
        const auto d = static_cast<Eigen::Index>(arr.shape(1));
        mqces::Sample s(n, d);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = 0; j < d; ++j) {
                s(i, j) = arr.data()[static_cast<std::size_t>(i * d + j)];
            }
        }
        out.push_back({name, std::move(s)});
    }
    return out;
}

mqces::Sample to_sample(
    const nb::ndarray<double, nb::c_contig>& arr, const char* what)
{
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(what) + " must be a 2-D array");
    }
    const auto n = static_cast<Eigen::Index>(arr.shape(0));
    const auto d = static_cast<Eigen::Index>(arr.shape(1));
    mqces::Sample s(n, d);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < d; ++j) {
            s(i, j) = arr.data()[static_cast<std::size_t>(i * d + j)];
        }
    }
    return s;
}

mqces::FeatureWeights to_weights(
    const nb::ndarray<double, nb::c_contig>& arr, Eigen::Index expected_d)
{
    if (arr.ndim() != 1) {
        throw std::invalid_argument("weights must be a 1-D array");
    }
    if (static_cast<Eigen::Index>(arr.shape(0)) != expected_d) {
        throw std::invalid_argument("weights length must equal test.shape[1]");
    }
    mqces::FeatureWeights w(expected_d);
    for (Eigen::Index k = 0; k < expected_d; ++k) {
        w(k) = arr.data()[static_cast<std::size_t>(k)];
    }
    return w;
}

mqces::ClassifierOptions build_options(
    const nb::ndarray<double, nb::c_contig>& weights_arr, Eigen::Index expected_d,
    double epsilon, std::size_t mc_samples, std::uint64_t seed,
    double nota_threshold, int n_threads)
{
    mqces::ClassifierOptions o;
    o.weights                   = to_weights(weights_arr, expected_d);
    o.uncertainty.epsilon       = epsilon;
    o.uncertainty.mc_samples    = mc_samples;
    o.uncertainty.seed          = seed;
    o.nota_threshold            = nota_threshold;
    o.n_threads                 = n_threads;
    return o;
}

nb::dict result_to_dict(const mqces::ClassificationResult& r)
{
    nb::dict d;
    d["best_class"]             = r.best_class;
    d["misclassification_prob"] = r.misclassification_prob;
    d["none_of_the_above"]      = r.none_of_the_above;
    std::vector<std::pair<std::string, double>> scores;
    scores.reserve(r.scores.size());
    for (const auto& s : r.scores) {
        scores.emplace_back(s.class_name, s.s_xy);
    }
    d["scores"] = nb::cast(std::move(scores));
    return d;
}

template <auto Classify>
nb::dict classify_wrapper(
    const nb::ndarray<double, nb::c_contig>&                                       test_arr,
    const std::vector<std::pair<std::string, nb::ndarray<double, nb::c_contig>>>&  classes,
    const nb::ndarray<double, nb::c_contig>&                                       weights_arr,
    double                                                                         epsilon,
    std::size_t                                                                    mc_samples,
    std::uint64_t                                                                  seed,
    double                                                                         nota_threshold,
    int                                                                            n_threads)
{
    mqces::Sample test = to_sample(test_arr, "test");
    auto          opts = build_options(weights_arr, test.cols(), epsilon,
                                       mc_samples, seed, nota_threshold, n_threads);
    auto          cs   = to_classes(classes);
    std::span<const mqces::Class> known(cs);
    return result_to_dict(Classify(test, known, opts));
}

}  // namespace

NB_MODULE(_mqces_core, m)
{
    m.doc()                  = "nanobind bindings for mqces";
    m.attr("__version__")    = mqces::version_string;

    m.def("classify_classic", &classify_wrapper<mqces::classic::classify>,
          nb::arg("test"), nb::arg("classes"), nb::arg("weights"),
          nb::arg("epsilon")        = 0.0,
          nb::arg("mc_samples")     = 10,
          nb::arg("seed")           = std::uint64_t{0xC0FFEE},
          nb::arg("nota_threshold") = 0.05,
          nb::arg("n_threads")      = 0,
          "Run mqces::classic::classify and return a dict result.");

    m.def("classify_v2", &classify_wrapper<mqces::v2::classify>,
          nb::arg("test"), nb::arg("classes"), nb::arg("weights"),
          nb::arg("epsilon")        = 0.0,
          nb::arg("mc_samples")     = 10,
          nb::arg("seed")           = std::uint64_t{0xC0FFEE},
          nb::arg("nota_threshold") = 0.05,
          nb::arg("n_threads")      = 0,
          "Run mqces::v2::classify and return a dict result.");

    m.def("similarity_score",
          [](const nb::ndarray<double, nb::c_contig>& x,
             const nb::ndarray<double, nb::c_contig>& y,
             const nb::ndarray<double, nb::c_contig>& w) {
              auto xs = to_sample(x, "x");
              auto ys = to_sample(y, "y");
              auto ws = to_weights(w, xs.cols());
              return mqces::similarity_score(xs, ys, ws);
          },
          nb::arg("x"), nb::arg("y"), nb::arg("weights"),
          "Compute Eq. 3 similarity score between two samples.");

    m.def("spatial_rank",
          [](const nb::ndarray<double, nb::c_contig>& x) {
              auto                                                                   xs = to_sample(x, "x");
              Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> u
                  = mqces::spatial_rank(xs);
              return u;
          },
          nb::arg("x"), "Compute Eq. 1 spatial rank vectors.");

    m.def("perturb",
          [](const nb::ndarray<double, nb::c_contig>& x, double eps,
             std::uint64_t seed) {
              return mqces::perturb(to_sample(x, "x"), eps, seed);
          },
          nb::arg("x"), nb::arg("epsilon"), nb::arg("seed"),
          "Multiplicative Gaussian perturbation (Eq. 10).");
}
