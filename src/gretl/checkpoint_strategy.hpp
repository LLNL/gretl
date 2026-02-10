// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file checkpoint_strategy.hpp
 * @brief Abstract interface for checkpoint eviction strategies.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>

namespace gretl {

/// @brief Performance counters for comparing checkpoint algorithms.
struct CheckpointMetrics {
  size_t stores = 0;          ///< Number of checkpoint store operations
  size_t evictions = 0;       ///< Number of checkpoint evictions
  size_t recomputations = 0;  ///< Forward re-evaluations triggered during reverse
};

/// @brief Abstract interface for checkpoint eviction strategies.
///
/// Implementations decide which step to evict when checkpoint capacity is
/// exceeded. The interface exposes only the operations that DataStore
/// requires, hiding all algorithm-specific data structures.
class CheckpointStrategy {
 public:
  static constexpr size_t invalidCheckpointIndex =
      std::numeric_limits<size_t>::max();  ///< Magic number for invalid checkpoint

  /// @brief Check if a checkpoint index is valid
  static bool valid_checkpoint_index(size_t i) { return i != invalidCheckpointIndex; }

  virtual ~CheckpointStrategy() = default;

  /// @brief Add a checkpoint for the given step.
  /// @param step The step index to checkpoint.
  /// @param persistent If true, this checkpoint cannot be evicted.
  /// @return The step index to evict, or invalidCheckpointIndex if none.
  virtual size_t add_checkpoint_and_get_index_to_remove(size_t step, bool persistent = false) = 0;

  /// @brief Return the step index of the earliest currently stored checkpoint.
  virtual size_t last_checkpoint_step() const = 0;

  /// @brief Remove the checkpoint at the given step.
  /// @return true if a checkpoint was found and removed.
  virtual bool erase_step(size_t stepIndex) = 0;

  /// @brief Check if a checkpoint exists for the given step.
  virtual bool contains_step(size_t stepIndex) const = 0;

  /// @brief Clear all non-persistent checkpoints.
  virtual void reset() = 0;

  /// @brief Return the maximum number of non-persistent checkpoint slots.
  virtual size_t capacity() const = 0;

  /// @brief Return the current number of checkpoints (persistent + non-persistent).
  virtual size_t size() const = 0;

  /// @brief Print checkpoint state to the output stream.
  virtual void print(std::ostream& os) const = 0;

  /// @brief Return accumulated performance metrics.
  virtual CheckpointMetrics metrics() const = 0;

  /// @brief Reset accumulated performance metrics to zero.
  virtual void reset_metrics() = 0;

  /// @brief Record a forward recomputation (called by DataStore during fetch).
  virtual void record_recomputation() = 0;
};

/// @brief ostream operator for CheckpointStrategy
inline std::ostream& operator<<(std::ostream& os, const CheckpointStrategy& s)
{
  s.print(os);
  return os;
}

}  // namespace gretl
