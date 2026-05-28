#pragma once

#include <cstdint>
#include <random>

namespace mqces::detail {

// splitmix64 mixer — used to derive uncorrelated 64-bit seeds from an
// integer index.
constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Derive a per-stream seed from a base seed and a stream index. Two
// different stream indices give uncorrelated mt19937_64 sequences,
// which is what we want for the Monte-Carlo replicate loop.
inline std::mt19937_64 make_stream(std::uint64_t base_seed, std::uint64_t stream_index)
{
    return std::mt19937_64(splitmix64(base_seed ^ splitmix64(stream_index)));
}

}  // namespace mqces::detail
