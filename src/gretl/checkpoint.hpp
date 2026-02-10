// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file checkpoint.hpp
 */

#pragma once

#include <map>
#include <ostream>
#include <iostream>
#include <cassert>
#include <limits>
#include <functional>
#include <memory>

#include "checkpoint_strategy.hpp"

/// @brief gretl_assert that prints line and file info before throwing in release and halting in debug
#define gretl_assert(x)                                                                                          \
  if (!(x))                                                                                                      \
    throw std::runtime_error{"Error on line " + std::to_string(__LINE__) + " in file " + std::string(__FILE__)}; \
  assert(x);

/// @brief gretl_assert_msg that prints message, line and file info before throwing in release and halting in debug
#define gretl_assert_msg(x, msg_name_)                                                                           \
  if (!(x))                                                                                                      \
    throw std::runtime_error{"Error on line " + std::to_string(__LINE__) + " in file " + std::string(__FILE__) + \
                             std::string(", ") + std::string(msg_name_)};                                        \
  assert(x);

namespace gretl {

/// @brief interface to run forward with a linear graph, checkpoint, then automatically backpropagate the sensitivities
/// given the reverse_callback vjp.
/// @tparam T type of each state's data
/// @param numSteps number of forward iterations
/// @param x initial condition
/// @param update_func function which evaluates the forward response
/// @param reverse_callback vjp function (action of Jacobian-transposed) to back propagate sensitivities
/// @param strategy checkpoint strategy (required)
/// @return
template <typename T>
T advance_and_reverse_steps(size_t numSteps, T x, std::function<T(size_t n, const T&)> update_func,
                            std::function<void(size_t n, const T&)> reverse_callback,
                            std::unique_ptr<CheckpointStrategy> strategy)
{
  CheckpointStrategy& cps = *strategy;
  std::map<size_t, T> savedCps;
  savedCps[0] = x;

  cps.add_checkpoint_and_get_index_to_remove(0, true);
  for (size_t i = 0; i < numSteps; ++i) {
    x = update_func(i, savedCps[i]);
    size_t eraseStep = cps.add_checkpoint_and_get_index_to_remove(i + 1, false);
    if (cps.valid_checkpoint_index(eraseStep)) {
      savedCps.erase(eraseStep);
    }

    savedCps[i + 1] = x;
  }

  double xf = x;

  for (size_t i = numSteps; i + 1 > 0; --i) {
    while (cps.last_checkpoint_step() < i) {
      size_t lastCp = cps.last_checkpoint_step();
      x = update_func(lastCp, savedCps[lastCp]);
      size_t eraseStep = cps.add_checkpoint_and_get_index_to_remove(lastCp + 1, false);
      if (cps.valid_checkpoint_index(eraseStep)) {
        savedCps.erase(eraseStep);
      }
      savedCps[lastCp + 1] = x;
      cps.record_recomputation();
    }
    reverse_callback(i, savedCps[i]);

    cps.erase_step(i);
    savedCps.erase(i);
  }

  return xf;
}

}  // namespace gretl
