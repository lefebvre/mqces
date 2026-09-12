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
//       "epsilon":           0.05,                     // optional, default 0 (no measurement error)
//       "mc_samples":        10,                       // optional, default 10; >= 2, used when epsilon > 0
//       "seed":              42,                       // optional
//       "nota_threshold":    0.05,                     // optional, default 0.05; in (0, 1)
//       "nota_permutations": 199,                      // optional, default 199; 0 skips NOTA
//       "n_threads":         0,                        // optional, default auto
//       "variant":           "classic" | "v2"          // optional, default classic
//     }
//   }
//
// Exit status: 0 on success, 1 on unreadable/invalid input or a failed
// classification or write, 2 on bad command-line usage.
//
// Output schema:
//   {
//     "best_class":             "...",
//     "scores":                 [{ "class_name": "...", "s_xy": 0.001 }, ...],
//     "misclassification_prob": 0.018,
//     "none_of_the_above":      false,
//     "variant":                "classic" | "v2"
//   }

#include <mqces/classify.hpp>
#include <mqces/types.hpp>
#include <mqces/version.hpp>

#include <nlohmann/json.hpp>

#include <Eigen/Core>

#include <cstdlib>
#include <fstream>
#include <iostream>
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
       << "  -o, --output <file>   Write JSON result to <file> (default: stdout).\n"
       << "      --pretty          Pretty-print the JSON output (indent=2).\n"
       << "  -h, --help            Show this message and exit.\n"
       << "  -V, --version         Print mqces version and exit.\n";
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
    std::string               variant;  // "classic" | "v2"
};

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
    in.options.nota_permutations
        = opt.value("nota_permutations", in.options.nota_permutations);
    in.options.n_threads      = opt.value("n_threads", in.options.n_threads);
    in.variant                = opt.value("variant", std::string{"classic"});

    if (in.variant != "classic" && in.variant != "v2") {
        throw std::runtime_error(
            R"(options.variant: must be "classic" or "v2" (got ")" + in.variant + "\")");
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

int run(int argc, char** argv)
{
    std::string input_path;
    std::string output_path;
    bool        pretty = false;

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

    mqces::ClassificationResult result;
    try {
        const std::span<const mqces::Class> known(parsed.classes);
        if (parsed.variant == "classic") {
            result = mqces::classic::classify(parsed.test, known, parsed.options);
        } else {
            result = mqces::v2::classify(parsed.test, known, parsed.options);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: classification failed: " << e.what() << "\n";
        return 1;
    }

    const auto out_json = result_to_json(result, parsed.variant);
    const auto serialized = pretty ? out_json.dump(2) : out_json.dump();

    if (output_path.empty()) {
        std::cout << serialized << "\n" << std::flush;
        if (!std::cout) {
            std::cerr << "error: failed to write result to stdout\n";
            return 1;
        }
    } else {
        std::ofstream f(output_path);
        if (!f) {
            std::cerr << "error: cannot open " << output_path << " for writing\n";
            return 1;
        }
        f << serialized << "\n";
        f.close();
        if (!f) {
            std::cerr << "error: failed to write result to " << output_path << "\n";
            return 1;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    // run() reports expected failures itself; this catches the rest (e.g.
    // std::bad_alloc) so no exception escapes main.
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "error: unknown exception\n";
    }
    return 1;
}
