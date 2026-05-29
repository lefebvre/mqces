#include <mqces/classify.hpp>
#include <mqces/types.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <random>
#include <span>
#include <vector>

using mqces::Class;
using mqces::ClassifierOptions;
using mqces::FeatureWeights;
using mqces::Sample;

namespace {

Class make_class(const std::string& name, int n, int d, double center,
                 double sigma, std::uint64_t seed)
{
    std::mt19937_64            gen(seed);
    std::normal_distribution<> dist(center, sigma);
    Class                      c;
    c.name      = name;
    c.specimens = Sample(n, d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            c.specimens(i, j) = dist(gen);
        }
    }
    return c;
}

ClassifierOptions default_options(int d)
{
    ClassifierOptions opts;
    opts.weights                 = FeatureWeights::Ones(d);
    opts.uncertainty.epsilon     = 0.0;
    opts.uncertainty.mc_samples  = 5;
    opts.uncertainty.seed        = 7777;
    opts.n_threads               = 1;
    return opts;
}

}  // namespace

// v4 must agree with v3 on best class — they share the same selection rule
// (argmin of mean score) and the same statistics. Only the inner solver
// changes, and AA converges to the same fixed point as plain VZ.
TEST(ClassifyV4, BestClassAgreesWithV3)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, -5.0, 0.5, 11),
        make_class("B", 20, d,  0.0, 0.5, 22),
        make_class("C", 20, d,  5.0, 0.5, 33),
    };
    Sample test = make_class("test", 20, d, 0.0, 0.5, 44).specimens;
    auto   opts = default_options(d);
    auto   r_v3 = mqces::v3::classify(test, std::span<const Class>{known}, opts);
    auto   r_v4 = mqces::v4::classify(test, std::span<const Class>{known}, opts);
    EXPECT_EQ(r_v3.best_class, r_v4.best_class);
    EXPECT_EQ(r_v4.best_class, "B");
}

// Misclass prob in [0, 1].
TEST(ClassifyV4, MisclassProbInUnitInterval)
{
    constexpr int d = 2;
    std::vector<Class> known{
        make_class("A", 15, d, 0.0, 0.5, 5),
        make_class("B", 15, d, 0.6, 0.5, 6),
    };
    Sample test = make_class("test", 15, d, 0.0, 0.5, 7).specimens;
    auto   res  = mqces::v4::classify(test, std::span<const Class>{known}, default_options(d));
    EXPECT_GE(res.misclassification_prob, 0.0);
    EXPECT_LE(res.misclassification_prob, 1.0);
}

// Single-class case has nothing to misclassify into.
TEST(ClassifyV4, SingleClassReportsZeroMisclassProb)
{
    constexpr int d = 2;
    std::vector<Class> known{make_class("only", 15, d, 0.0, 1.0, 8)};
    Sample             test = make_class("test", 15, d, 0.0, 1.0, 9).specimens;
    auto               res  = mqces::v4::classify(
        test, std::span<const Class>{known}, default_options(d));
    EXPECT_EQ(res.best_class, "only");
    EXPECT_EQ(res.misclassification_prob, 0.0);
}

// Determinism.
TEST(ClassifyV4, IsDeterministic)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, -3.0, 0.5, 1),
        make_class("B", 20, d,  0.0, 0.5, 2),
    };
    Sample test = make_class("test", 20, d, 0.0, 0.5, 3).specimens;
    auto   opts = default_options(d);
    auto   r1   = mqces::v4::classify(test, std::span<const Class>{known}, opts);
    auto   r2   = mqces::v4::classify(test, std::span<const Class>{known}, opts);
    EXPECT_EQ(r1.best_class, r2.best_class);
    EXPECT_EQ(r1.misclassification_prob, r2.misclassification_prob);
}

// Explicit caller-provided solver override is honored.
TEST(ClassifyV4, RespectsExplicitSolverOverride)
{
    constexpr int d = 3;
    std::vector<Class> known{
        make_class("A", 20, d, -2.0, 0.5, 1),
        make_class("B", 20, d,  2.0, 0.5, 2),
    };
    Sample test = make_class("test", 20, d, 0.0, 0.5, 3).specimens;

    auto opts                = default_options(d);
    opts.solver.kind         = mqces::SolverKind::VardiZhangAA;
    opts.solver.tol          = 1e-8;
    opts.solver.max_iters    = 300;
    opts.solver.aa_window    = 4;
    opts.solver.aa_reg       = 1e-11;
    opts.solver.aa_safeguard = true;

    auto res = mqces::v4::classify(test, std::span<const Class>{known}, opts);
    EXPECT_FALSE(res.best_class.empty());
}
