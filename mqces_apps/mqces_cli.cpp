// mqces command-line interface.
//
// Reads a JSON input file describing the test sample, known classes, and
// classifier options; writes a JSON classification result to stdout (or to
// an output file via -o).
//
// Input schema:
//   {
//     "weights":  [w_0, w_1, ...],                     // length = n_features
//     "test":     [[r00, r01, ...], [r10, r11, ...]],  // (n_specimens, n_features)
//     "classes": [
//       { "name": "...", "specimens": [[...], ...] },  // each (n, n_features)
//       ...
//     ],
//     "options": {
//       "epsilon":         0.05,                       // optional, default 0
//       "mc_samples":      10,                         // optional, default 10
//       "seed":            42,                         // optional
//       "nota_threshold":  0.05,                       // optional, default 0.05
//       "n_threads":       0,                          // optional, default auto
//       "variant":         "classic"|"v2"|"v3"|"v4",   // optional, default classic
//       "reference_size":  0,                          // optional, 0 == exact
//       "sampling_seed":   0xACEBEEF,                  // optional
//       "solver_tol":      1e-7,                       // optional, solver-dep
//       "solver_max_iters": 500,                       // optional
//       "vz_vertex_eps":   1e-6,                       // optional, v3/v4 only
//       "aa_window":       5,                          // optional, v4 only
//       "aa_reg":          1e-12                       // optional, v4 only
//     }
//   }
//
// Command-line flags override the corresponding JSON option:
//   --variant <name>           overrides options.variant
//   --reference-size <N>       overrides options.reference_size
//
// Output schema:
//   {
//     "best_class":             "...",
//     "scores":                 [{ "class_name": "...", "s_xy": 0.001 }, ...],
//     "misclassification_prob": 0.018,
//     "none_of_the_above":      false,
//     "variant":                "classic"|"v2"|"v3"|"v4"
//   }

#include <mqces/classify.hpp>
#include <mqces/types.hpp>
#include <mqces/version.hpp>

#include <nlohmann/json.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

void print_usage(std::ostream& os, const char* argv0)
{
    os << "usage: " << argv0 << " [options] <input.json>\n"
       << "\n"
       << "Options:\n"
       << "  -o, --output <file>          Write JSON result to <file> (default: stdout).\n"
       << "      --pretty                 Pretty-print the JSON output (indent=2).\n"
       << "      --variant <name>         Override input.options.variant\n"
       << "                               (classic|v2|v3|v4).\n"
       << "      --reference-size <N>     Override input.options.reference_size\n"
       << "                               (0 == exact).\n"
       << "  -h, --help                   Show this message and exit.\n"
       << "  -V, --version                Print mqces version and exit.\n";
}

mqces::Sample parse_sample(const json& j, const char* what)
{
    if (!j.is_array() || j.empty() || !j.front().is_array()) {
        throw std::runtime_error(std::string(what) + ": expected a non-empty 2-D array");
    }
    const auto n = static_cast<Eigen::Index>(j.size());
    const auto d = static_cast<Eigen::Index>(j.front().size());
    mqces::Sample s(n, d);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (!j[static_cast<std::size_t>(i)].is_array()
            || static_cast<Eigen::Index>(j[static_cast<std::size_t>(i)].size()) != d) {
            throw std::runtime_error(std::string(what) + ": ragged rows");
        }
        for (Eigen::Index k = 0; k < d; ++k) {
            s(i, k) = j[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]
                          .get<double>();
        }
    }
    return s;
}

mqces::FeatureWeights parse_weights(const json& j, Eigen::Index expected_d)
{
    if (!j.is_array() || static_cast<Eigen::Index>(j.size()) != expected_d) {
        throw std::runtime_error("weights: expected array of length n_features");
    }
    mqces::FeatureWeights w(expected_d);
    for (Eigen::Index k = 0; k < expected_d; ++k) {
        w(k) = j[static_cast<std::size_t>(k)].get<double>();
    }
    return w;
}

struct ParsedInput {
    mqces::Sample             test;
    std::vector<mqces::Class> classes;
    mqces::ClassifierOptions  options;
    std::string               variant;  // "classic" | "v2" | "v3" | "v4"
};

bool is_known_variant(const std::string& v)
{
    return v == "classic" || v == "v2" || v == "v3" || v == "v4";
}

ParsedInput parse_input(const json& j)
{
    ParsedInput in;
    in.test = parse_sample(j.at("test"), "test");

    if (!j.contains("classes") || !j["classes"].is_array() || j["classes"].empty()) {
        throw std::runtime_error("classes: expected non-empty array");
    }
    for (const auto& cj : j["classes"]) {
        mqces::Class c;
        c.name      = cj.at("name").get<std::string>();
        c.specimens = parse_sample(cj.at("specimens"), c.name.c_str());
        in.classes.push_back(std::move(c));
    }

    in.options.weights = parse_weights(j.at("weights"), in.test.cols());

    const auto& opt = j.value("options", json::object());
    in.options.uncertainty.epsilon
        = opt.value("epsilon", in.options.uncertainty.epsilon);
    in.options.uncertainty.mc_samples
        = opt.value("mc_samples", in.options.uncertainty.mc_samples);
    in.options.uncertainty.seed
        = opt.value("seed", in.options.uncertainty.seed);
    in.options.nota_threshold = opt.value("nota_threshold", in.options.nota_threshold);
    in.options.n_threads      = opt.value("n_threads", in.options.n_threads);
    in.variant                = opt.value("variant", std::string{"classic"});

    // SamplingConfig.
    in.options.sampling.reference_size
        = opt.value("reference_size", in.options.sampling.reference_size);
    in.options.sampling.seed = opt.value("sampling_seed", in.options.sampling.seed);

    // SolverConfig — leave kind alone (each classifier variant picks its
    // own default if not overridden by the user). Other fields apply
    // uniformly across variants.
    in.options.solver.tol          = opt.value("solver_tol", in.options.solver.tol);
    in.options.solver.max_iters    = opt.value("solver_max_iters", in.options.solver.max_iters);
    in.options.solver.vz_vertex_eps
        = opt.value("vz_vertex_eps", in.options.solver.vz_vertex_eps);
    in.options.solver.aa_window    = opt.value("aa_window", in.options.solver.aa_window);
    in.options.solver.aa_reg       = opt.value("aa_reg", in.options.solver.aa_reg);

    if (!is_known_variant(in.variant)) {
        throw std::runtime_error(
            "options.variant: must be one of classic|v2|v3|v4 (got \""
            + in.variant + "\")");
    }
    return in;
}

json result_to_json(const mqces::ClassificationResult& r, const std::string& variant)
{
    json out;
    out["best_class"]             = r.best_class;
    out["misclassification_prob"] = r.misclassification_prob;
    out["none_of_the_above"]      = r.none_of_the_above;
    out["variant"]                = variant;
    out["scores"]                 = json::array();
    for (const auto& s : r.scores) {
        out["scores"].push_back({{"class_name", s.class_name}, {"s_xy", s.s_xy}});
    }
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string                input_path;
    std::string                output_path;
    bool                       pretty = false;
    std::optional<std::string> variant_override;
    std::optional<std::size_t> reference_size_override;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(std::cout, argv[0]);
            return 0;
        }
        if (a == "-V" || a == "--version") {
            std::cout << "mqces " << mqces::version_string << "\n";
            return 0;
        }
        if (a == "--pretty") {
            pretty = true;
        } else if (a == "-o" || a == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << a << " requires a path\n";
                return 2;
            }
            output_path = argv[++i];
        } else if (a == "--variant") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << a << " requires a name\n";
                return 2;
            }
            variant_override = argv[++i];
        } else if (a == "--reference-size") {
            if (i + 1 >= argc) {
                std::cerr << "error: " << a << " requires an integer\n";
                return 2;
            }
            try {
                const long long parsed = std::stoll(argv[++i]);
                if (parsed < 0) {
                    std::cerr << "error: --reference-size must be non-negative "
                                 "(got " << parsed << ")\n";
                    return 2;
                }
                reference_size_override = static_cast<std::size_t>(parsed);
            } catch (const std::exception&) {
                std::cerr << "error: --reference-size value is not an integer\n";
                return 2;
            }
        } else if (!a.empty() && a.front() == '-') {
            std::cerr << "error: unknown flag '" << a << "'\n";
            print_usage(std::cerr, argv[0]);
            return 2;
        } else if (input_path.empty()) {
            input_path = std::move(a);
        } else {
            std::cerr << "error: extra positional argument '" << a << "'\n";
            return 2;
        }
    }
    if (input_path.empty()) {
        std::cerr << "error: missing input.json\n";
        print_usage(std::cerr, argv[0]);
        return 2;
    }

    json input_json;
    try {
        std::ifstream in(input_path);
        if (!in) {
            std::cerr << "error: cannot open " << input_path << "\n";
            return 1;
        }
        in >> input_json;
    } catch (const std::exception& e) {
        std::cerr << "error: failed to parse JSON from " << input_path << ": "
                  << e.what() << "\n";
        return 1;
    }

    ParsedInput parsed;
    try {
        parsed = parse_input(input_json);
    } catch (const std::exception& e) {
        std::cerr << "error: invalid input: " << e.what() << "\n";
        return 1;
    }

    // Apply CLI overrides on top of the JSON-derived options.
    if (variant_override) {
        if (!is_known_variant(*variant_override)) {
            std::cerr << "error: --variant must be classic|v2|v3|v4 (got \""
                      << *variant_override << "\")\n";
            return 2;
        }
        parsed.variant = *variant_override;
    }
    if (reference_size_override) {
        parsed.options.sampling.reference_size = *reference_size_override;
    }

    mqces::ClassificationResult result;
    try {
        const std::span<const mqces::Class> known(parsed.classes);
        if (parsed.variant == "classic") {
            result = mqces::classic::classify(parsed.test, known, parsed.options);
        } else if (parsed.variant == "v2") {
            result = mqces::v2::classify(parsed.test, known, parsed.options);
        } else if (parsed.variant == "v3") {
            result = mqces::v3::classify(parsed.test, known, parsed.options);
        } else {  // v4 (validated above)
            result = mqces::v4::classify(parsed.test, known, parsed.options);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: classification failed: " << e.what() << "\n";
        return 1;
    }

    const auto out_json = result_to_json(result, parsed.variant);
    const auto serialized = pretty ? out_json.dump(2) : out_json.dump();

    if (output_path.empty()) {
        std::cout << serialized << "\n";
    } else {
        std::ofstream f(output_path);
        if (!f) {
            std::cerr << "error: cannot open " << output_path << " for writing\n";
            return 1;
        }
        f << serialized << "\n";
    }
    return 0;
}
