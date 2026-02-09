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
#include "wang_checkpoint_strategy.hpp"

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

/// @brief Backward-compatible CheckpointManager wrapper.
/// Delegates to WangCheckpointStrategy under the hood.
class CheckpointManager {
 public:
  static constexpr size_t invalidCheckpointIndex = CheckpointStrategy::invalidCheckpointIndex;

  explicit CheckpointManager(size_t maxStates = 20) : maxNumStates(maxStates) {}

  static bool valid_checkpoint_index(size_t i) { return CheckpointStrategy::valid_checkpoint_index(i); }

  size_t add_checkpoint_and_get_index_to_remove(size_t step, bool persistent = false)
  {
    return impl().add_checkpoint_and_get_index_to_remove(step, persistent);
  }

  size_t last_checkpoint_step() const { return impl().last_checkpoint_step(); }

  bool erase_step(size_t stepIndex) { return impl().erase_step(stepIndex); }

  bool contains_step(size_t stepIndex) const { return impl().contains_step(stepIndex); }

  void reset()
  {
    if (impl_) impl_->reset();
  }

  size_t maxNumStates;

 private:
  CheckpointStrategy& impl()
  {
    if (!impl_) impl_ = std::make_unique<WangCheckpointStrategy>(maxNumStates);
    return *impl_;
  }

  const CheckpointStrategy& impl() const
  {
    if (!impl_) impl_ = std::make_unique<WangCheckpointStrategy>(maxNumStates);
    return *impl_;
  }

  mutable std::unique_ptr<CheckpointStrategy> impl_;
};

/// @brief ostream operator for backward-compatible CheckpointManager
inline std::ostream& operator<<(std::ostream& stream, const CheckpointManager& mgr)
{
  // Delegate to the underlying strategy if available
  stream << "CHECKPOINTS: capacity = " << mgr.maxNumStates << std::endl;
  return stream;
}

/// @brief interface to run forward with a linear graph, checkpoint, then automatically backpropagate the sensitivities
/// given the reverse_callback vjp.
/// @tparam T type of each state's data
/// @param numSteps number of forward iterations
/// @param storageSize maximum states to save in memory at a time
/// @param x initial condition
/// @param update_func function which evaluates the forward response
/// @param reverse_callback vjp function (action of Jacobian-transposed) to back propagate sensitivities
/// @param strategy optional checkpoint strategy (defaults to WangCheckpointStrategy)
/// @return
template <typename T>
T advance_and_reverse_steps(size_t numSteps, size_t storageSize, T x, std::function<T(size_t n, const T&)> update_func,
                            std::function<void(size_t n, const T&)> reverse_callback,
                            std::unique_ptr<CheckpointStrategy> strategy = nullptr)
{
  if (!strategy) {
    strategy = std::make_unique<WangCheckpointStrategy>(storageSize);
  }
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
