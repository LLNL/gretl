// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "wang_checkpoint_strategy.hpp"
#include <cassert>
#include <iostream>

namespace gretl {

WangCheckpointStrategy::WangCheckpointStrategy(size_t maxStates) : maxNumStates_(maxStates) {}

std::set<WangCheckpointStrategy::Checkpoint, WangCheckpointStrategy::CheckpointCompare>::const_iterator
WangCheckpointStrategy::most_dispensable() const
{
  size_t maxHigherTimeLevel = 0;
  for (auto rIter = cps_.begin(); rIter != cps_.end(); ++rIter) {
    if (rIter->level < maxHigherTimeLevel) {
      return rIter;
    }
    maxHigherTimeLevel = std::max(rIter->level, maxHigherTimeLevel);
  }
  return cps_.end();
}

size_t WangCheckpointStrategy::add_checkpoint_and_get_index_to_remove(size_t step, bool persistent)
{
  size_t levelupAmount = 1;

  Checkpoint nextStep{.level = levelupAmount - 1, .step = step};

  size_t nextEraseStep = invalidCheckpointIndex;

  if (persistent) {
    maxNumStates_++;
    nextStep.level = Checkpoint::infinity();
    assert(cps_.size() < maxNumStates_);
  }

  if (cps_.size() < maxNumStates_) {
    cps_.insert(nextStep);
  } else {
    auto iterToMostDispensable = most_dispensable();
    if (iterToMostDispensable != cps_.end()) {
      nextEraseStep = iterToMostDispensable->step;
      cps_.erase(iterToMostDispensable);
      cps_.insert(nextStep);
    } else {
      nextEraseStep = cps_.begin()->step;
      nextStep.level = cps_.begin()->level + levelupAmount;

      cps_.erase(cps_.begin());
      cps_.insert(nextStep);
    }
  }

  metrics_.stores++;
  if (valid_checkpoint_index(nextEraseStep)) {
    metrics_.evictions++;
  }

  return nextEraseStep;
}

size_t WangCheckpointStrategy::last_checkpoint_step() const { return cps_.begin()->step; }

bool WangCheckpointStrategy::erase_step(size_t stepIndex)
{
  for (auto it = cps_.begin(); it != cps_.end(); ++it) {
    if (it->step == stepIndex) {
      if (it->level != Checkpoint::infinity()) {
        cps_.erase(it);
        return true;
      }
    }
  }
  return false;
}

bool WangCheckpointStrategy::contains_step(size_t stepIndex) const
{
  for (const auto& c : cps_) {
    if (c.step == stepIndex) {
      return true;
    }
  }
  return false;
}

void WangCheckpointStrategy::reset()
{
  for (auto cp_it = cps_.begin(); cp_it != cps_.end(); ++cp_it) {
    if (cp_it->level == Checkpoint::infinity()) {
      cps_.erase(cps_.begin(), cp_it);
      break;
    }
  }
}

size_t WangCheckpointStrategy::capacity() const { return maxNumStates_; }

size_t WangCheckpointStrategy::size() const { return cps_.size(); }

void WangCheckpointStrategy::print(std::ostream& os) const
{
  os << "CHECKPOINTS (Wang): capacity = " << maxNumStates_ << std::endl;
  for (const auto& s : cps_) {
    os << "   lvl=" << s.level << ", step=" << s.step << "\n";
  }
}

CheckpointMetrics WangCheckpointStrategy::metrics() const { return metrics_; }

void WangCheckpointStrategy::reset_metrics() { metrics_ = {}; }

void WangCheckpointStrategy::record_recomputation() { metrics_.recomputations++; }

}  // namespace gretl
