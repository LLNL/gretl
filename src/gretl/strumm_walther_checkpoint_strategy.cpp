// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "strumm_walther_checkpoint_strategy.hpp"
#include <cassert>
#include <iostream>
#include <limits>

namespace gretl {

StrummWaltherCheckpointStrategy::StrummWaltherCheckpointStrategy(size_t maxStates) : maxNumSlots_(maxStates) {}

size_t StrummWaltherCheckpointStrategy::find_dispensable() const
{
  // Weight-based dispensability (analogous to Wang's most_dispensable):
  // Iterate from highest step to lowest. Track the maximum weight seen.
  // A slot is "dispensable" if its weight is LESS than the running maximum —
  // it sits behind a more important (higher-weight) checkpoint.
  //
  // Enhancement over Wang: when multiple slots share the same dispensable
  // weight, choose the one whose removal minimizes the increase in total
  // recomputation cost (gap_left * gap_right). This spacing-aware tiebreaker
  // can outperform Wang's arbitrary "first found" selection.

  // First pass: find the dispensable weight threshold
  size_t maxWeight = 0;
  size_t dispensableWeight = std::numeric_limits<size_t>::max();
  for (size_t i = slots_.size(); i > 0; --i) {
    size_t idx = i - 1;
    if (slots_[idx].persistent) continue;

    if (slots_[idx].weight < maxWeight) {
      dispensableWeight = slots_[idx].weight;
      break;
    }
    maxWeight = std::max(maxWeight, slots_[idx].weight);
  }

  if (dispensableWeight == std::numeric_limits<size_t>::max()) {
    return slots_.size();  // none found
  }

  // Second pass: among all slots at dispensableWeight, pick the one with
  // minimum gap_left * gap_right (minimum delta recomputation cost).
  size_t bestIdx = slots_.size();
  size_t bestProduct = std::numeric_limits<size_t>::max();

  for (size_t i = 0; i < slots_.size(); ++i) {
    if (slots_[i].persistent) continue;
    if (slots_[i].weight != dispensableWeight) continue;

    // Check that this slot is actually in a dispensable position
    // (there must be a higher-weight slot at a higher step)
    bool hasHigherWeightAfter = false;
    for (size_t j = i + 1; j < slots_.size(); ++j) {
      if (!slots_[j].persistent && slots_[j].weight > dispensableWeight) {
        hasHigherWeightAfter = true;
        break;
      }
    }
    if (!hasHigherWeightAfter) continue;

    size_t leftStep = (i > 0) ? slots_[i - 1].step : 0;
    size_t rightStep = (i + 1 < slots_.size()) ? slots_[i + 1].step : slots_.back().step + 1;

    size_t gapLeft = slots_[i].step - leftStep;
    size_t gapRight = rightStep - slots_[i].step;
    size_t product = gapLeft * gapRight;

    if (product < bestProduct) {
      bestProduct = product;
      bestIdx = i;
    }
  }

  return bestIdx;
}

size_t StrummWaltherCheckpointStrategy::find_rightmost_nonpersistent() const
{
  for (size_t i = slots_.size(); i > 0; --i) {
    if (!slots_[i - 1].persistent) {
      return i - 1;
    }
  }
  return slots_.size();
}

size_t StrummWaltherCheckpointStrategy::add_checkpoint_and_get_index_to_remove(size_t step, bool persistent)
{
  size_t nextEraseStep = invalidCheckpointIndex;

  size_t newWeight = 0;

  if (persistent) {
    maxNumSlots_++;
    assert(slots_.size() < maxNumSlots_);
  }

  if (slots_.size() < maxNumSlots_) {
    // Space available — insert directly
  } else {
    // At capacity — must evict
    size_t dispensableIdx = find_dispensable();

    if (dispensableIdx < slots_.size()) {
      // Found a dispensable slot: evict it, new checkpoint gets weight 0
      nextEraseStep = slots_[dispensableIdx].step;
      slots_.erase(slots_.begin() + static_cast<ptrdiff_t>(dispensableIdx));
    } else {
      // No dispensable slot (all weights equal): evict the rightmost
      // non-persistent and PROMOTE the replacement to a higher weight.
      // This is the key self-organizing mechanism: it creates a weight
      // hierarchy that forces future evictions to target older, lower-weight
      // checkpoints, producing near-logarithmic checkpoint distributions.
      size_t rightmostIdx = find_rightmost_nonpersistent();
      assert(rightmostIdx < slots_.size());
      newWeight = slots_[rightmostIdx].weight + 1;
      nextEraseStep = slots_[rightmostIdx].step;
      slots_.erase(slots_.begin() + static_cast<ptrdiff_t>(rightmostIdx));
    }
  }

  // Insert new slot in sorted order
  Slot newSlot{step, persistent, persistent ? std::numeric_limits<size_t>::max() : newWeight};
  auto it = std::lower_bound(slots_.begin(), slots_.end(), step, [](const Slot& s, size_t st) { return s.step < st; });
  slots_.insert(it, newSlot);

  metrics_.stores++;
  if (valid_checkpoint_index(nextEraseStep)) {
    metrics_.evictions++;
  }

  return nextEraseStep;
}

size_t StrummWaltherCheckpointStrategy::last_checkpoint_step() const
{
  assert(!slots_.empty());
  return slots_.back().step;
}

bool StrummWaltherCheckpointStrategy::erase_step(size_t stepIndex)
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

bool StrummWaltherCheckpointStrategy::contains_step(size_t stepIndex) const
{
  for (const auto& s : slots_) {
    if (s.step == stepIndex) {
      return true;
    }
  }
  return false;
}

void StrummWaltherCheckpointStrategy::reset()
{
  slots_.erase(std::remove_if(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.persistent; }), slots_.end());
}

size_t StrummWaltherCheckpointStrategy::capacity() const { return maxNumSlots_; }

size_t StrummWaltherCheckpointStrategy::size() const { return slots_.size(); }

void StrummWaltherCheckpointStrategy::print(std::ostream& os) const
{
  os << "CHECKPOINTS (StrummWalther): capacity = " << maxNumSlots_ << std::endl;
  for (const auto& s : slots_) {
    os << "   step=" << s.step << " weight=" << s.weight << (s.persistent ? " (persistent)" : "") << "\n";
  }
}

CheckpointMetrics StrummWaltherCheckpointStrategy::metrics() const { return metrics_; }

void StrummWaltherCheckpointStrategy::reset_metrics() { metrics_ = {}; }

void StrummWaltherCheckpointStrategy::record_recomputation() { metrics_.recomputations++; }

}  // namespace gretl
