// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "online_r2_checkpoint_strategy.hpp"
#include <cassert>
#include <iostream>
#include <limits>

namespace gretl {

OnlineR2CheckpointStrategy::OnlineR2CheckpointStrategy(size_t maxStates) : maxNumSlots_(maxStates) {}

size_t OnlineR2CheckpointStrategy::find_eviction_candidate(size_t newStep) const
{
  // Minimum-gap eviction: find the non-persistent checkpoint whose removal
  // creates the smallest merged gap between its neighbors. This maintains
  // approximately uniform spacing by preferentially removing checkpoints
  // in dense clusters.
  //
  // The newStep parameter serves as the virtual right boundary for the
  // rightmost checkpoint, preventing the most recent checkpoint from
  // being trivially evicted.

  size_t bestIdx = slots_.size();
  size_t bestMergedGap = std::numeric_limits<size_t>::max();

  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].persistent) continue;

    size_t leftStep = (i > 0) ? slots_[i - 1].step : 0;
    size_t rightStep = (i + 1 < slots_.size()) ? slots_[i + 1].step : newStep;

    size_t mergedGap = rightStep - leftStep;

    if (mergedGap < bestMergedGap) {
      bestMergedGap = mergedGap;
      bestIdx = i;
    }
  }

  return bestIdx;
}

size_t OnlineR2CheckpointStrategy::add_checkpoint_and_get_index_to_remove(size_t step, bool persistent)
{
  size_t nextEraseStep = invalidCheckpointIndex;

  Slot newSlot{step, persistent};

  if (persistent) {
    maxNumSlots_++;
    assert(slots_.size() < maxNumSlots_);
  }

  if (slots_.size() < maxNumSlots_) {
    // Space available, insert in sorted order
    auto it = std::lower_bound(slots_.begin(), slots_.end(), step,
                               [](const Slot& s, size_t st) { return s.step < st; });
    slots_.insert(it, newSlot);
  } else {
    // At capacity, must evict
    size_t evictIdx = find_eviction_candidate(step);
    if (evictIdx < slots_.size()) {
      nextEraseStep = slots_[evictIdx].step;
      slots_.erase(slots_.begin() + static_cast<ptrdiff_t>(evictIdx));
      // Insert new slot in sorted order
      auto it = std::lower_bound(slots_.begin(), slots_.end(), step,
                                 [](const Slot& s, size_t st) { return s.step < st; });
      slots_.insert(it, newSlot);
    }
  }

  metrics_.stores++;
  if (valid_checkpoint_index(nextEraseStep)) {
    metrics_.evictions++;
  }

  return nextEraseStep;
}

size_t OnlineR2CheckpointStrategy::last_checkpoint_step() const
{
  assert(!slots_.empty());
  return slots_.back().step;
}

bool OnlineR2CheckpointStrategy::erase_step(size_t stepIndex)
{
  for (auto it = slots_.begin(); it != slots_.end(); ++it) {
    if (it->step == stepIndex) {
      if (!it->persistent) {
        slots_.erase(it);
        return true;
      }
    }
  }
  return false;
}

bool OnlineR2CheckpointStrategy::contains_step(size_t stepIndex) const
{
  for (const auto& s : slots_) {
    if (s.step == stepIndex) {
      return true;
    }
  }
  return false;
}

void OnlineR2CheckpointStrategy::reset()
{
  slots_.erase(std::remove_if(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.persistent; }),
               slots_.end());
}

size_t OnlineR2CheckpointStrategy::capacity() const { return maxNumSlots_; }

size_t OnlineR2CheckpointStrategy::size() const { return slots_.size(); }

void OnlineR2CheckpointStrategy::print(std::ostream& os) const
{
  os << "CHECKPOINTS (OnlineR2): capacity = " << maxNumSlots_ << std::endl;
  for (const auto& s : slots_) {
    os << "   step=" << s.step << (s.persistent ? " (persistent)" : "") << "\n";
  }
}

CheckpointMetrics OnlineR2CheckpointStrategy::metrics() const { return metrics_; }

void OnlineR2CheckpointStrategy::reset_metrics() { metrics_ = {}; }

void OnlineR2CheckpointStrategy::record_recomputation() { metrics_.recomputations++; }

}  // namespace gretl
