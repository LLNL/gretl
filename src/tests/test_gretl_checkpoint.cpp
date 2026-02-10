// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <stdio.h>
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "gretl/checkpoint.hpp"
#include "gretl/checkpoint_strategy.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/strumm_walther_checkpoint_strategy.hpp"
#include "gretl/state.hpp"
#include "gretl/data_store.hpp"

static size_t count = 0;

double advance_solution(double x)
{
  ++count;
  return x / 3.0 + 2.0;
}

// ---------- Original non-parameterized tests (backward compat) ----------

struct CheckpointFixture : public ::testing::Test {
  static constexpr size_t S = 6;   // max saved states
  static constexpr size_t N = 10;  // run states

  std::vector<double> get_full_state_hist(double x0)
  {
    std::vector<double> states(N + 1);
    states[0] = x0;
    for (size_t n = 0; n < N; ++n) {
      states[n + 1] = advance_solution(states[n]);
    }
    return states;
  }
};

TEST_F(CheckpointFixture, Procedural)
{
  double x0 = 0.0;

  std::vector<double> states = get_full_state_hist(x0);
  std::vector<double> reverseStates(N + 1);

  gretl::WangCheckpointStrategy checkpointStrategy(S);
  std::map<size_t, double> savedCheckpoints;

  savedCheckpoints[0] = x0;

  bool persistentCheckpoint = true;
  checkpointStrategy.add_checkpoint_and_get_index_to_remove(0, persistentCheckpoint);
  for (size_t i = 0; i < N; ++i) {
    const auto& xPrev = savedCheckpoints[i];
    auto x = advance_solution(xPrev);
    size_t stepToErase = checkpointStrategy.add_checkpoint_and_get_index_to_remove(i + 1);
    if (gretl::CheckpointStrategy::valid_checkpoint_index(stepToErase)) {
      savedCheckpoints.erase(stepToErase);
    }
    savedCheckpoints[i + 1] = x;
  }

  for (size_t i_rev = N; i_rev + 1 > 0; --i_rev) {
    for (size_t i = checkpointStrategy.last_checkpoint_step(); i < i_rev; ++i) {
      const auto& xPrev = savedCheckpoints[i];
      auto x = advance_solution(xPrev);
      size_t stepToErase = checkpointStrategy.add_checkpoint_and_get_index_to_remove(i + 1);
      if (gretl::CheckpointStrategy::valid_checkpoint_index(stepToErase)) {
        savedCheckpoints.erase(stepToErase);
      }
      savedCheckpoints[i + 1] = x;
    }

    reverseStates[i_rev] = savedCheckpoints[i_rev];

    checkpointStrategy.erase_step(i_rev);
    savedCheckpoints.erase(i_rev);
  }

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], reverseStates[n]) << n << "\n";
  }
  std::cout << "total eval count = " << count << std::endl;
  count = 0;
}

TEST_F(CheckpointFixture, Functional)
{
  double x0 = 0.0;

  std::vector<double> states = get_full_state_hist(x0);
  std::vector<double> advanceStates(N + 1);
  std::vector<double> reverseStates(N + 1);

  double xf = gretl::advance_and_reverse_steps<double>(
      N, x0,
      [&](size_t n, const double& x) {
        // update function
        advanceStates[n] = x;
        return advance_solution(x);
      },
      [&](size_t n, const double& x) {
        // callback on reverse pass for computing reverse sensitivities
        reverseStates[n] = x;
      },
      std::make_unique<gretl::WangCheckpointStrategy>(S));

  advanceStates[N] = xf;

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], advanceStates[n]) << n << "\n";
    ASSERT_EQ(states[n], reverseStates[n]) << n << "\n";
  }

  std::cout << "total eval count = " << count << std::endl;
  count = 0;
}

gretl::State<double> advance_solution(const gretl::State<double>& a)
{
  auto b = a.clone({a});

  b.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    const auto& a_ = upstreams[0];
    double B = advance_solution(a_.get<double>());
    downstream.set(B);
  });

  b.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    auto a_ = upstreams[0];
    const auto b_ = downstream;
    auto bBar = b_.get_dual<double>();
    a_.get_dual<double, double>() += bBar / 3.0;
  });

  return b.finalize();
}

TEST_F(CheckpointFixture, Automated)
{
  double x = 0.0;

  std::vector<double> states = get_full_state_hist(x);
  std::vector<double> reverseStates(N + 1);
  std::vector<double> advanceStates(N + 1);

  gretl::DataStore dataStore(std::make_unique<gretl::WangCheckpointStrategy>(S));
  gretl::State<double> X = dataStore.create_state<double, double>(x);

  advanceStates[0] = X.get();
  for (size_t n = 0; n < N; ++n) {
    X = advance_solution(X);
    advanceStates[n + 1] = X.get();
  }

  X = set_as_objective(X);
  dataStore.stillConstructingGraph_ = false;

  reverseStates[N] = X.get();
  EXPECT_EQ(X.get_dual(), 1.0);
  for (size_t n = N; n > 0; --n) {
    dataStore.reverse_state();
    auto restoredState = static_cast<gretl::Int>(n - 1);
    reverseStates[n - 1] = dataStore.get_primal<double>(restoredState);
    double dual_val = dataStore.get_dual<double, double>(restoredState);
    ASSERT_NEAR(dual_val, std::pow(1. / 3., (N - n + 1)), 1e-14);
  }

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], advanceStates[n]) << n << "\n";
    ASSERT_EQ(states[n], reverseStates[n]) << n << "\n";
  }

  std::cout << "total eval count = " << count << std::endl;
  count = 0;
}

// ---------- Parameterized tests across checkpoint strategies ----------

enum class StrategyType
{
  Wang,
  StrummWalther
};

std::string strategy_name(StrategyType t)
{
  switch (t) {
    case StrategyType::Wang:
      return "Wang";
    case StrategyType::StrummWalther:
      return "StrummWalther";
  }
  return "Unknown";
}

std::unique_ptr<gretl::CheckpointStrategy> make_strategy(StrategyType t, size_t slots)
{
  switch (t) {
    case StrategyType::Wang:
      return std::make_unique<gretl::WangCheckpointStrategy>(slots);
    case StrategyType::StrummWalther:
      return std::make_unique<gretl::StrummWaltherCheckpointStrategy>(slots);
  }
  return nullptr;
}

struct CheckpointStrategyTest : public ::testing::TestWithParam<StrategyType> {
  static constexpr size_t S = 6;
  static constexpr size_t N = 10;

  std::vector<double> get_full_state_hist(double x0)
  {
    std::vector<double> states(N + 1);
    states[0] = x0;
    for (size_t n = 0; n < N; ++n) {
      states[n + 1] = advance_solution(states[n]);
    }
    return states;
  }
};

TEST_P(CheckpointStrategyTest, Procedural)
{
  double x0 = 0.0;

  std::vector<double> states = get_full_state_hist(x0);
  std::vector<double> reverseStates(N + 1);

  auto strategy = make_strategy(GetParam(), S);
  std::map<size_t, double> savedCheckpoints;

  savedCheckpoints[0] = x0;

  strategy->add_checkpoint_and_get_index_to_remove(0, true);
  for (size_t i = 0; i < N; ++i) {
    const auto& xPrev = savedCheckpoints[i];
    auto x = advance_solution(xPrev);
    size_t stepToErase = strategy->add_checkpoint_and_get_index_to_remove(i + 1);
    if (gretl::CheckpointStrategy::valid_checkpoint_index(stepToErase)) {
      savedCheckpoints.erase(stepToErase);
    }
    savedCheckpoints[i + 1] = x;
  }

  for (size_t i_rev = N; i_rev + 1 > 0; --i_rev) {
    while (strategy->last_checkpoint_step() < i_rev) {
      size_t lastCp = strategy->last_checkpoint_step();
      const auto& xPrev = savedCheckpoints[lastCp];
      auto x = advance_solution(xPrev);
      size_t stepToErase = strategy->add_checkpoint_and_get_index_to_remove(lastCp + 1);
      if (gretl::CheckpointStrategy::valid_checkpoint_index(stepToErase)) {
        savedCheckpoints.erase(stepToErase);
      }
      savedCheckpoints[lastCp + 1] = x;
    }

    reverseStates[i_rev] = savedCheckpoints[i_rev];

    strategy->erase_step(i_rev);
    savedCheckpoints.erase(i_rev);
  }

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], reverseStates[n]) << strategy_name(GetParam()) << " step " << n << "\n";
  }

  auto m = strategy->metrics();
  std::cout << strategy_name(GetParam()) << " procedural: stores=" << m.stores << " evictions=" << m.evictions
            << " eval_count=" << count << std::endl;
  count = 0;
}

TEST_P(CheckpointStrategyTest, Functional)
{
  double x0 = 0.0;

  std::vector<double> states = get_full_state_hist(x0);
  std::vector<double> advanceStates(N + 1);
  std::vector<double> reverseStates(N + 1);

  auto strategy = make_strategy(GetParam(), S);

  double xf = gretl::advance_and_reverse_steps<double>(
      N, x0,
      [&](size_t n, const double& x) {
        advanceStates[n] = x;
        return advance_solution(x);
      },
      [&](size_t n, const double& x) { reverseStates[n] = x; }, std::move(strategy));

  advanceStates[N] = xf;

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], advanceStates[n]) << strategy_name(GetParam()) << " step " << n << "\n";
    ASSERT_EQ(states[n], reverseStates[n]) << strategy_name(GetParam()) << " step " << n << "\n";
  }

  std::cout << strategy_name(GetParam()) << " functional: eval_count=" << count << std::endl;
  count = 0;
}

TEST_P(CheckpointStrategyTest, Automated)
{
  double x = 0.0;

  std::vector<double> states = get_full_state_hist(x);
  std::vector<double> reverseStates(N + 1);
  std::vector<double> advanceStates(N + 1);

  auto strategy = make_strategy(GetParam(), S);
  gretl::DataStore dataStore(std::move(strategy));
  gretl::State<double> X = dataStore.create_state<double, double>(x);

  advanceStates[0] = X.get();
  for (size_t n = 0; n < N; ++n) {
    X = advance_solution(X);
    advanceStates[n + 1] = X.get();
  }

  X = set_as_objective(X);
  dataStore.stillConstructingGraph_ = false;

  reverseStates[N] = X.get();
  EXPECT_EQ(X.get_dual(), 1.0);
  for (size_t n = N; n > 0; --n) {
    dataStore.reverse_state();
    auto restoredState = static_cast<gretl::Int>(n - 1);
    reverseStates[n - 1] = dataStore.get_primal<double>(restoredState);
    double dual_val = dataStore.get_dual<double, double>(restoredState);
    ASSERT_NEAR(dual_val, std::pow(1. / 3., (N - n + 1)), 1e-14);
  }

  for (size_t n = 0; n < N + 1; ++n) {
    ASSERT_EQ(states[n], advanceStates[n]) << strategy_name(GetParam()) << " step " << n << "\n";
    ASSERT_EQ(states[n], reverseStates[n]) << strategy_name(GetParam()) << " step " << n << "\n";
  }

  auto m = dataStore.checkpointStrategy_->metrics();
  std::cout << strategy_name(GetParam()) << " automated: stores=" << m.stores << " evictions=" << m.evictions
            << " recomps=" << m.recomputations << " eval_count=" << count << std::endl;
  count = 0;
}

INSTANTIATE_TEST_SUITE_P(AllStrategies, CheckpointStrategyTest,
                         ::testing::Values(StrategyType::Wang, StrategyType::StrummWalther),
                         [](const ::testing::TestParamInfo<StrategyType>& param_info) {
                           return strategy_name(param_info.param);
                         });
