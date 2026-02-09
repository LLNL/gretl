// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file wang_checkpoint_strategy.hpp
 * @brief Wang et al. 2009 "Minimal Repetition Dynamic Checkpointing" strategy.
 */

#pragma once

#include "checkpoint_strategy.hpp"
#include <set>

namespace gretl {

/// @brief Wang et al. 2009 "Minimal Repetition Dynamic Checkpointing"
///
/// Uses a level-based priority scheme where each checkpoint has a level
/// that determines its dispensability. The "most dispensable" checkpoint
/// is the one whose level drops below a previously seen higher level
/// when iterating the ordered set.
class WangCheckpointStrategy final : public CheckpointStrategy {
 public:
  /// @brief Construct with a given number of non-persistent checkpoint slots.
  explicit WangCheckpointStrategy(size_t maxStates);

  size_t add_checkpoint_and_get_index_to_remove(size_t step, bool persistent = false) override;
  size_t last_checkpoint_step() const override;
  bool erase_step(size_t stepIndex) override;
  bool contains_step(size_t stepIndex) const override;
  void reset() override;
  size_t capacity() const override;
  size_t size() const override;
  void print(std::ostream& os) const override;
  CheckpointMetrics metrics() const override;
  void reset_metrics() override;
  void record_recomputation() override;

 private:
  /// @brief Checkpoint with level for eviction priority (Wang-specific).
  struct Checkpoint {
    size_t level;  ///< level
    size_t step;   ///< step
    static constexpr size_t infinity() { return std::numeric_limits<size_t>::max(); }
  };

  /// @brief Comparison operator for ordering checkpoints in the set.
  /// Persistent checkpoints (infinity level) sort last; among others, higher step first.
  struct CheckpointCompare {
    bool operator()(const Checkpoint& a, const Checkpoint& b) const
    {
      if (a.level == Checkpoint::infinity() && b.level == Checkpoint::infinity()) {
        return a.step > b.step;
      }
      if (a.level == Checkpoint::infinity()) return false;
      if (b.level == Checkpoint::infinity()) return true;
      return a.step > b.step;
    }
  };

  /// @brief Find the most dispensable checkpoint per the Wang algorithm.
  std::set<Checkpoint, CheckpointCompare>::const_iterator most_dispensable() const;

  size_t maxNumStates_;
  std::set<Checkpoint, CheckpointCompare> cps_;
  CheckpointMetrics metrics_;
};

}  // namespace gretl
