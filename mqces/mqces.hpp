#pragma once

/**
 * @file mqces.hpp
 * @brief Umbrella include for the mqces public API.
 *
 * Pulls in the classifier entry points, score primitive, spatial-rank
 * primitives, Monte-Carlo sampler, NOTA decision, and version macros.
 * Consumers can include this single header to reach everything public.
 */

#include <mqces/classify.hpp>
#include <mqces/nota.hpp>
#include <mqces/quantile.hpp>
#include <mqces/score.hpp>
#include <mqces/types.hpp>
#include <mqces/uncertainty.hpp>
#include <mqces/version.hpp>
