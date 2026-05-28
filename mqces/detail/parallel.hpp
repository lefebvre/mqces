#pragma once

#include <cstddef>
#include <utility>

#ifdef MQCES_HAVE_OPENMP
#include <omp.h>
#endif

// Tiny parallel shim. OpenMP backend when MQCES_HAVE_OPENMP is defined,
// sequential fallback otherwise. Kokkos backend will live in a separate
// translation unit (parallel_kokkos.cpp) when MQCES_ENABLE_KOKKOS=ON.

namespace mqces::detail {

template <class F>
inline void parallel_for(std::size_t n, F&& f)
{
#ifdef MQCES_HAVE_OPENMP
    const auto n_signed = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n_signed; ++i) {
        f(static_cast<std::size_t>(i));
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        f(i);
    }
#endif
}

// parallel_reduce(n, init, map, combine): apply `map(i)` for i in [0, n),
// combine results with `combine(acc, value)` starting from `init`. The
// combine operation must be associative and commutative.
template <class T, class Map, class Combine>
inline T parallel_reduce(std::size_t n, T init, Map&& map, Combine&& combine)
{
    if (n == 0) {
        return init;
    }
#ifdef MQCES_HAVE_OPENMP
    T result = init;
#pragma omp parallel
    {
        T local = init;
#pragma omp for nowait schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            local = combine(local, map(static_cast<std::size_t>(i)));
        }
#pragma omp critical
        {
            result = combine(result, local);
        }
    }
    return result;
#else
    T result = init;
    for (std::size_t i = 0; i < n; ++i) {
        result = combine(result, map(i));
    }
    return result;
#endif
}

inline int max_threads()
{
#ifdef MQCES_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int current_thread_id()
{
#ifdef MQCES_HAVE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline void set_max_threads(int n)
{
#ifdef MQCES_HAVE_OPENMP
    if (n > 0) {
        omp_set_num_threads(n);
    }
#else
    (void)n;
#endif
}

}  // namespace mqces::detail
