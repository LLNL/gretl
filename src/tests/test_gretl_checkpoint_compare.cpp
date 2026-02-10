// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/// @file test_gretl_checkpoint_compare.cpp
/// @brief Side-by-side comparison of Wang and StrummWalther checkpointing strategies.

#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include "gtest/gtest.h"
#include "gretl/checkpoint.hpp"
#include "gretl/checkpoint_strategy.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/strumm_walther_checkpoint_strategy.hpp"
#include "gretl/state.hpp"
#include "gretl/data_store.hpp"

namespace {

double forward_step(double x) { return x / 3.0 + 2.0; }

gretl::State<double> forward_step_state(const gretl::State<double>& a)
{
  auto b = a.clone({a});

  b.set_eval([](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    downstream.set(forward_step(upstreams[0].get<double>()));
  });

  b.set_vjp([](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    upstreams[0].get_dual<double, double>() += downstream.get_dual<double>() / 3.0;
  });

  return b.finalize();
}

struct AlgorithmResult {
  std::string name;
  gretl::CheckpointMetrics metrics;
  double gradient;
};

AlgorithmResult run_procedural_test(std::unique_ptr<gretl::CheckpointStrategy> strategy, const std::string& name,
                                    size_t N)
{
  double x0 = 0.0;
  std::map<size_t, double> savedCps;
  savedCps[0] = x0;
  double x = x0;

  strategy->add_checkpoint_and_get_index_to_remove(0, true);
  for (size_t i = 0; i < N; ++i) {
    x = forward_step(savedCps[i]);
    size_t eraseStep = strategy->add_checkpoint_and_get_index_to_remove(i + 1);
    if (gretl::CheckpointStrategy::valid_checkpoint_index(eraseStep)) {
      savedCps.erase(eraseStep);
    }
    savedCps[i + 1] = x;
  }

  double grad = 1.0;
  for (size_t i_rev = N; i_rev + 1 > 0; --i_rev) {
    while (strategy->last_checkpoint_step() < i_rev) {
      size_t lastCp = strategy->last_checkpoint_step();
      x = forward_step(savedCps[lastCp]);
      size_t eraseStep = strategy->add_checkpoint_and_get_index_to_remove(lastCp + 1);
      if (gretl::CheckpointStrategy::valid_checkpoint_index(eraseStep)) {
        savedCps.erase(eraseStep);
      }
      savedCps[lastCp + 1] = x;
      strategy->record_recomputation();
    }
    grad *= 1.0 / 3.0;  // derivative of forward_step

    strategy->erase_step(i_rev);
    savedCps.erase(i_rev);
  }

  return {name, strategy->metrics(), grad};
}

AlgorithmResult run_datastore_test(std::unique_ptr<gretl::CheckpointStrategy> strategy, const std::string& name,
                                   size_t N)
{
  gretl::DataStore dataStore(std::move(strategy));
  gretl::State<double> X = dataStore.create_state<double, double>(0.0);

  for (size_t n = 0; n < N; ++n) {
    X = forward_step_state(X);
  }

  X = set_as_objective(X);
  dataStore.stillConstructingGraph_ = false;
  dataStore.back_prop();

  double grad = dataStore.get_dual<double, double>(0);
  return {name, dataStore.checkpointStrategy_->metrics(), grad};
}

}  // namespace

TEST(CheckpointCompare, ProceduralComparison)
{
  struct Config {
    size_t N;
    size_t budget;
  };

  std::vector<Config> configs = {{10, 3},    {10, 5},    {10, 6},    {10, 8},     {20, 3},     {20, 5},
                                 {20, 8},    {20, 10},   {50, 3},    {50, 5},     {50, 10},    {50, 20},
                                 {100, 3},   {100, 5},   {100, 10},  {100, 20},   {200, 5},    {200, 10},
                                 {200, 20},  {500, 5},   {500, 10},  {500, 20},   {1000, 10},  {1000, 20},
                                 {1000, 50}, {5000, 10}, {5000, 50}, {5000, 100}, {5000, 200}, {5000, 500}};

  std::cout << "\n--- Procedural Checkpoint Algorithm Comparison ---\n";
  std::cout << std::setw(6) << "N" << std::setw(8) << "Budget" << " | " << std::setw(10) << "Algorithm" << std::setw(10)
            << "stores" << std::setw(10) << "evictions" << std::setw(12) << "recomps" << std::setw(14) << "ratio(r/N)"
            << "\n";
  std::cout << std::string(72, '-') << "\n";

  for (const auto& cfg : configs) {
    auto wang_result = run_procedural_test(std::make_unique<gretl::WangCheckpointStrategy>(cfg.budget), "Wang", cfg.N);
    auto r2_result = run_procedural_test(std::make_unique<gretl::StrummWaltherCheckpointStrategy>(cfg.budget),
                                         "StrummWalther", cfg.N);

    ASSERT_NEAR(wang_result.gradient, r2_result.gradient, 1e-14)
        << "Gradient mismatch at N=" << cfg.N << " budget=" << cfg.budget;

    for (const auto& r : {wang_result, r2_result}) {
      std::cout << std::setw(6) << cfg.N << std::setw(8) << cfg.budget << " | " << std::setw(10) << r.name
                << std::setw(10) << r.metrics.stores << std::setw(10) << r.metrics.evictions << std::setw(12)
                << r.metrics.recomputations << std::setw(14) << std::fixed << std::setprecision(3)
                << static_cast<double>(r.metrics.recomputations) / static_cast<double>(cfg.N) << "\n";
    }
  }
  std::cout << std::endl;
}

TEST(CheckpointCompare, DataStoreComparison)
{
  struct Config {
    size_t N;
    size_t budget;
  };

  std::vector<Config> configs = {{10, 3},    {10, 5},    {10, 6},    {10, 8},     {20, 3},     {20, 5},    {20, 8},
                                 {20, 10},   {50, 5},    {50, 10},   {50, 20},    {100, 5},    {100, 10},  {100, 20},
                                 {200, 5},   {200, 10},  {200, 20},  {500, 10},   {500, 20},   {1000, 10}, {1000, 20},
                                 {1000, 50}, {5000, 10}, {5000, 50}, {5000, 100}, {5000, 200}, {5000, 500}};

  std::cout << "\n--- DataStore Checkpoint Algorithm Comparison ---\n";
  std::cout << std::setw(6) << "N" << std::setw(8) << "Budget" << " | " << std::setw(10) << "Algorithm" << std::setw(10)
            << "stores" << std::setw(10) << "evictions" << std::setw(12) << "recomps" << std::setw(14) << "ratio(r/N)"
            << "\n";
  std::cout << std::string(72, '-') << "\n";

  for (const auto& cfg : configs) {
    auto wang_result = run_datastore_test(std::make_unique<gretl::WangCheckpointStrategy>(cfg.budget), "Wang", cfg.N);
    auto r2_result = run_datastore_test(std::make_unique<gretl::StrummWaltherCheckpointStrategy>(cfg.budget),
                                        "StrummWalther", cfg.N);

    double expected_grad = std::pow(1.0 / 3.0, cfg.N);
    ASSERT_NEAR(wang_result.gradient, expected_grad, 1e-14) << "Wang gradient wrong at N=" << cfg.N;
    ASSERT_NEAR(r2_result.gradient, expected_grad, 1e-14) << "StrummWalther gradient wrong at N=" << cfg.N;

    for (const auto& r : {wang_result, r2_result}) {
      std::cout << std::setw(6) << cfg.N << std::setw(8) << cfg.budget << " | " << std::setw(10) << r.name
                << std::setw(10) << r.metrics.stores << std::setw(10) << r.metrics.evictions << std::setw(12)
                << r.metrics.recomputations << std::setw(14) << std::fixed << std::setprecision(3)
                << static_cast<double>(r.metrics.recomputations) / static_cast<double>(cfg.N) << "\n";
    }
  }
  std::cout << std::endl;
}
