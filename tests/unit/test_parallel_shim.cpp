#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

using mqces::detail::current_thread_id;
using mqces::detail::make_stream;
using mqces::detail::max_threads;
using mqces::detail::parallel_for;
using mqces::detail::parallel_reduce;
using mqces::detail::splitmix64;

// parallel_for visits every index exactly once.
TEST(ParallelShim, ParallelForVisitsAllIndicesOnce)
{
    constexpr std::size_t n = 10'000;
    std::vector<int>      hits(n, 0);
    std::atomic<int>      races{0};
    parallel_for(n, [&](std::size_t i) {
        // Race detection: if two threads write the same slot we'll see > 1
        // after the join. We don't synchronize the write — that's the point.
        if (hits[i] != 0) {
            races.fetch_add(1);
        }
        hits[i] = 1;
    });
    EXPECT_EQ(races.load(), 0);
    int total = std::accumulate(hits.begin(), hits.end(), 0);
    EXPECT_EQ(total, static_cast<int>(n));
}

// parallel_reduce sums to the correct total.
TEST(ParallelShim, ParallelReduceSumsCorrectly)
{
    constexpr std::size_t n = 10'000;
    auto                  sum = parallel_reduce(
        n, 0LL,
        [](std::size_t i) { return static_cast<long long>(i); },
        [](long long a, long long b) { return a + b; });
    const long long expected = static_cast<long long>(n) * (n - 1) / 2;
    EXPECT_EQ(sum, expected);
}

// n == 0 short-circuits to init.
TEST(ParallelShim, ParallelReduceZeroLengthReturnsInit)
{
    auto sum = parallel_reduce(
        0, 42LL, [](std::size_t) { return 1LL; }, [](long long a, long long b) { return a + b; });
    EXPECT_EQ(sum, 42);
}

// Thread-id and max-threads return sensible values.
TEST(ParallelShim, ThreadIdInRange)
{
    const int               max = max_threads();
    std::atomic<int>        worst{-1};
    constexpr std::size_t   n = 1'000;
    parallel_for(n, [&](std::size_t) {
        int tid = current_thread_id();
        int prev = worst.load();
        while (tid > prev && !worst.compare_exchange_weak(prev, tid)) {}
    });
    EXPECT_GE(worst.load(), 0);
    EXPECT_LT(worst.load(), max);
}

// splitmix64 is deterministic and avalanches: differing inputs give very
// different outputs (popcount of XOR ≥ 16 on average for unrelated inputs).
TEST(ParallelShim, SplitMix64IsDeterministicAndMixes)
{
    EXPECT_EQ(splitmix64(0), splitmix64(0));
    EXPECT_NE(splitmix64(0), splitmix64(1));

    const auto a = splitmix64(12345);
    const auto b = splitmix64(12346);
    int        diff_bits = __builtin_popcountll(a ^ b);
    EXPECT_GE(diff_bits, 16);  // very loose avalanche bound
}

// Different stream indices yield uncorrelated mt19937_64 sequences.
TEST(ParallelShim, MakeStreamDifferentIndicesGiveDifferentSequences)
{
    auto r1 = make_stream(0xABCDEF, 0);
    auto r2 = make_stream(0xABCDEF, 1);
    EXPECT_NE(r1(), r2());
}
