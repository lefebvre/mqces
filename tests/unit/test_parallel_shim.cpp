#include <mqces/detail/parallel.hpp>
#include <mqces/detail/rng.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using mqces::detail::current_thread_id;
using mqces::detail::make_stream;
using mqces::detail::max_threads;
using mqces::detail::parallel_for;
using mqces::detail::shuffle;
using mqces::detail::splitmix64;
using mqces::detail::uniform_index;

// parallel_for visits every index exactly once.
TEST(ParallelShim, ParallelForVisitsAllIndicesOnce) {
  constexpr std::size_t n = 10'000;
  std::vector<int> hits(n, 0);
  std::atomic<int> races{0};
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

// An exception thrown by the body reaches the caller as that exception
// instead of escaping the parallel region (which would terminate).
TEST(ParallelShim, ParallelForPropagatesExceptions) {
  constexpr std::size_t n = 1'000;
  EXPECT_THROW(parallel_for(n,
                            [](std::size_t i) {
                              if (i == 517) {
                                throw std::runtime_error("boom");
                              }
                            }),
               std::runtime_error);
}

// A per-call thread count must not leak into the process-wide OpenMP
// setting seen by later, unrelated parallel regions.
TEST(ParallelShim, ThreadCountDoesNotChangeGlobalDefault) {
  const int before = max_threads();
  parallel_for(16, 1, [](std::size_t) {});
  parallel_for(16, before + 3, [](std::size_t) {});
  EXPECT_EQ(max_threads(), before);
}

// A parallel_for nested inside another runs serially on the calling thread,
// so the outer loop's team size governs. With n_threads = 1 the outer region
// is inactive in OpenMP terms, and without this guard an inner region would
// become the first active level and use every core, defeating the cap that
// classify's n_threads option promises.
TEST(ParallelShim, NestedLoopRespectsOuterThreadCap) {
  std::mutex mutex;
  std::set<std::thread::id> threads;
  parallel_for(4, 1, [&](std::size_t) {
    parallel_for(256, [&](std::size_t) {
      const std::lock_guard<std::mutex> lock(mutex);
      threads.insert(std::this_thread::get_id());
    });
  });
  EXPECT_EQ(threads.size(), 1u);
}

// Nesting depth is restored after an exception, so a later top-level loop is
// parallel again rather than stuck on the serial path.
TEST(ParallelShim, NestingDepthRestoredAfterException) {
  EXPECT_THROW(parallel_for(8, 1, [](std::size_t) { throw std::runtime_error("boom"); }),
               std::runtime_error);
  if (max_threads() < 2) {
    GTEST_SKIP() << "needs at least two OpenMP threads";
  }
  std::mutex mutex;
  std::set<std::thread::id> threads;
  parallel_for(4096, [&](std::size_t) {
    const std::lock_guard<std::mutex> lock(mutex);
    threads.insert(std::this_thread::get_id());
  });
  EXPECT_GT(threads.size(), 1u);
}

// Thread-id and max-threads return sensible values.
TEST(ParallelShim, ThreadIdInRange) {
  const int max = max_threads();
  std::atomic<int> worst{-1};
  constexpr std::size_t n = 1'000;
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
TEST(ParallelShim, SplitMix64IsDeterministicAndMixes) {
  EXPECT_EQ(splitmix64(0), splitmix64(0));
  EXPECT_NE(splitmix64(0), splitmix64(1));

  const auto a = splitmix64(12345);
  const auto b = splitmix64(12346);
  int diff_bits = std::popcount(a ^ b);
  EXPECT_GE(diff_bits, 16);  // very loose avalanche bound
}

// Different stream indices yield uncorrelated mt19937_64 sequences.
TEST(ParallelShim, MakeStreamDifferentIndicesGiveDifferentSequences) {
  auto r1 = make_stream(0xABCDEF, 0);
  auto r2 = make_stream(0xABCDEF, 1);
  EXPECT_NE(r1(), r2());
}

// The in-tree distributions are defined by raw mt19937_64 output, which the
// standard fixes, so these values hold on every standard library.
TEST(ParallelShim, PortableDrawsArePinned) {
  std::mt19937_64 rng(5489u);
  EXPECT_EQ(uniform_index(rng, 10), 0u);  // first output 14514284786278117030 % 10

  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};
  std::mt19937_64 rng2(1);
  shuffle(v.begin(), v.size(), rng2);
  EXPECT_EQ(v, (std::vector<int>{4, 6, 3, 5, 1, 7, 2, 0}));

  // log/cos may round differently in the last place across libm builds.
  std::mt19937_64 rng3(2);
  EXPECT_NEAR(mqces::detail::standard_normal(rng3), 0.26519244583197793, 1e-12);
}

// uniform_index covers [0, n) and stays in range.
TEST(ParallelShim, UniformIndexCoversRange) {
  std::mt19937_64 rng(99);
  std::vector<int> hits(7, 0);
  for (int i = 0; i < 7'000; ++i) {
    const auto k = uniform_index(rng, 7);
    ASSERT_LT(k, 7u);
    ++hits[static_cast<std::size_t>(k)];
  }
  for (int h : hits) {
    EXPECT_GT(h, 800);
  }
}
