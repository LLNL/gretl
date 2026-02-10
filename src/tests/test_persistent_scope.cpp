// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <vector>
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "gretl/vector_state.hpp"
#include "gretl/data_store.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"
#include "gretl/test_utils.hpp"

using gretl::print;

// Helper function that creates the initial persistent state but doesn't return it
// This tests that the DataStore properly tracks persistent states even when
// the local State<> object goes out of scope
void create_initial_state(gretl::DataStore& dataStore, const std::vector<double>& data)
{
  // Create persistent state - it goes into dataStore's persistent list
  auto initial = dataStore.create_state(data, gretl::vec::initialize_zero_dual);

  // State object goes out of scope here, but the underlying StateData
  // should still be tracked by DataStore since it's persistent
}

TEST(PersistentScope, InitialStateGoesOutOfScope)
{
  std::vector<double> dataA = {1.5, 2.5, 3.5};

  gretl::DataStore dataStore(std::make_unique<gretl::WangCheckpointStrategy>(10));

  // Create the initial persistent state in a function - it goes out of scope
  create_initial_state(dataStore, dataA);

  // Now create some computation steps
  // The first state created after this should be at step 1
  std::vector<double> dataB = {0.5, 0.5, 0.5};
  auto b = dataStore.create_state(dataB, gretl::vec::initialize_zero_dual);

  // Do some operations
  auto c = b + b;  // 2*b
  auto d = c + b;  // 3*b
  auto e = d + c;  // 5*b

  // Verify the computation
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(e.get()[i], 5.0 * dataB[i], 1e-14);
  }

  // Set up for backpropagation
  auto qoi = gretl::inner_product(e, e);
  gretl::set_as_objective(qoi);

  dataStore.back_prop();

  // Verify gradients
  // qoi = e·e = (5b)·(5b) = 25(b·b)
  // dqoi/db = 50b
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(b.get_dual()[i], 50.0 * dataB[i], 1e-13);
  }

  // The test passes if we get here without ASAN errors
  std::cout << "Test passed - initial persistent state properly managed" << std::endl;
}

TEST(PersistentScope, MultipleStatesGoOutOfScope)
{
  std::vector<double> data1 = {1.0, 2.0};
  std::vector<double> data2 = {3.0, 4.0};
  std::vector<double> data3 = {5.0, 6.0};

  gretl::DataStore dataStore(std::make_unique<gretl::WangCheckpointStrategy>(10));

  // Create multiple persistent states that go out of scope
  {
    auto s1 = dataStore.create_state(data1, gretl::vec::initialize_zero_dual);
    auto s2 = dataStore.create_state(data2, gretl::vec::initialize_zero_dual);
    auto s3 = dataStore.create_state(data3, gretl::vec::initialize_zero_dual);
    // All three go out of scope here
  }

  // Now do some computation with new states
  std::vector<double> dataX = {0.1, 0.2};
  auto x = dataStore.create_state(dataX, gretl::vec::initialize_zero_dual);
  auto y = x + x;
  auto z = y * x;

  // Verify computation
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(y.get()[i], 2.0 * dataX[i], 1e-14);
    EXPECT_NEAR(z.get()[i], 2.0 * dataX[i] * dataX[i], 1e-14);
  }

  auto qoi = gretl::inner_product(z, z);
  gretl::set_as_objective(qoi);

  dataStore.back_prop();

  // qoi = z·z = (2x²)·(2x²) = 4(x²)·(x²) = 4x⁴
  // dqoi/dx = 16x³
  for (size_t i = 0; i < 2; ++i) {
    double xi = dataX[i];
    EXPECT_NEAR(x.get_dual()[i], 16.0 * xi * xi * xi, 1e-13);
  }

  std::cout << "Test passed - multiple persistent states properly managed" << std::endl;
}
