#include <gtest/gtest.h>
#include "gretl/data_store.hpp"
#include "gretl/state.hpp"
#include "gretl/create_state.hpp"
#include "gretl/wang_checkpoint_strategy.hpp"

using namespace gretl;

TEST(GraphTracking, StopGradient) {
    DataStore ds(10);
    auto s1 = ds.create_state<double, double>(2.0); // tracked, val=2
    auto s2 = ds.create_state<double, double>(3.0); // tracked, val=3
    
    // Create an untracked (stop-gradient) state manually using the ds toggle
    ds.set_gradients_enabled(false);
    
    // This state is added to the graph but its VJP is replaced with a no-op
    auto s3 = create_state<double, double>(
        [](const double&) { return 0.0; },
        [](const double& a, const double& b) { return a * b; },
        [](const double&, const double&, const double&, double&, double&, const double&) {
            // This original VJP code would normally fail if we somehow reached it
            // but we shouldn't reach it because it gets replaced with a no-op.
            gretl_assert_msg(false, "VJP for stop-gradient node should never be called");
        },
        s1, s2);
    
    EXPECT_EQ(s3.get(), 6.0);
    
    // Re-enable gradients
    ds.set_gradients_enabled(true);
    
    // Create a tracked state using s3 as an upstream
    // s4 = s1 + s3 = 2.0 + 6.0 = 8.0
    auto s4 = create_state<double, double>(
        [](const double&) { return 0.0; },
        [](const double& a, const double& b) { return a + b; },
        [](const double&, const double&, const double&, double& a_bar, double& b_bar, const double& c_bar) {
            a_bar += c_bar;
            b_bar += c_bar;
        },
        s1, s3);
        
    EXPECT_EQ(s4.get(), 8.0);
    
    auto obj = set_as_objective(s4); // derivative wrt s4 is 1.0
    
    ds.finalize_graph();
    ds.back_prop();
    
    // derivative of s4 wrt s1 directly is 1.0.
    // derivative of s4 wrt s3 is 1.0. 
    // BUT since s3 is stop-gradient, its derivative is NOT passed back to s1 or s2.
    // Normally, ds4/ds1 = 1 + ds3/ds1 = 1 + s2 = 1 + 3 = 4.0
    // With stop-gradient on s3, ds4/ds1 = 1.0.
    
    EXPECT_EQ(s1.get_dual(), 1.0);
    
    // s2 only affects s4 via s3. So its dual should be zero.
    EXPECT_EQ(s2.get_dual(), 0.0);
}
    
    EXPECT_TRUE(ds.is_tracking());
    
    // Create a tracked state using s3 as an upstream
    // s4 = s1 + s3 = 2.0 + 6.0 = 8.0
    auto s4 = create_state<double, double>(
        [](const double&) { return 0.0; },
        [](const double& a, const double& b) { return a + b; },
        [](const double&, const double&, const double&, double& a_bar, double& b_bar, const double& c_bar) {
            a_bar += c_bar;
            b_bar += c_bar;
        },
        s1, s3);
        
    EXPECT_EQ(s4.get(), 8.0);
    
    auto obj = set_as_objective(s4); // derivative wrt s4 is 1.0
    
    ds.finalize_graph();
    ds.back_prop();
    
    EXPECT_EQ(s1.get_dual(), 1.0);
    EXPECT_EQ(s2.get_dual(), 0.0);
}
