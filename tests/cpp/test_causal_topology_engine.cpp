#include "causal_trace_pool.h"
#include "dynamic_coordinate_engine.h"
#include <iostream>
#include <cassert>

using namespace causal_topology;

void test_causal_trace_pool() {
    CausalStatePool pool(5);
    assert(pool.count == 5);

    // Set initial agent values
    pool.values[0] = 1.0f;
    pool.resistor_x[0] = 2.5f;
    pool.is_axis[0] = 0;

    size_t idx0 = pool.push_back(2.0f, 0.5f, 101, 0x1, 0.9f, -1, 0.001f, 1);
    assert(idx0 == 5);
    assert(pool.is_axis[5] == 1);

    size_t idx1 = pool.push_back(2.5f, 0.5f, 102, 0x1, 0.95f, static_cast<int32_t>(idx0), 0.001f, 1);

    auto chain = pool.trace_back(static_cast<int32_t>(idx1));
    assert(chain.size() == 2);
    assert(chain[0] == static_cast<int32_t>(idx1));
    assert(chain[1] == static_cast<int32_t>(idx0));

    std::cout << "[PASS] CausalStatePool tests successful!" << std::endl;
}

void test_dynamic_coordinate_engine() {
    EngineConfig config;
    config.condensation_threshold = 0.8f;
    config.relativization_friction = 0.5f;

    DynamicCoordinateEngine engine(config);
    CausalStatePool pool(2);

    // Node 0: Dynamic variable x with initial invariance 0.75
    pool.values[0] = 0.0f;
    pool.resistor_x[0] = 5.0f;
    pool.is_axis[0] = 0;
    pool.invariance_scores[0] = 0.75f;

    // Node 1: Fixed Coordinate Axis (1, 2)
    pool.values[1] = 10.0f;
    pool.resistor_x[1] = 0.001f;
    pool.is_axis[1] = 1;
    pool.invariance_scores[1] = 0.95f;

    // Step 1: Forward
    std::vector<float> inputs = {10.0f, 10.0f};
    engine.forward_step(pool, inputs);
    assert(pool.values[0] > 0.0f);
    assert(pool.values[1] > 10.0f);

    // Step 2: Reflect with low error for Node 0 -> Should condense to Axis!
    std::vector<float> low_errors = {0.01f, 0.01f};
    engine.reflect_and_mutate(pool, low_errors);
    assert(pool.invariance_scores[0] >= 0.8f);
    assert(pool.is_axis[0] == 1); // Condensed to Axis!

    // Step 3: Reflect with high friction for Node 1 -> Should relativize into Variable x!
    std::vector<float> high_friction = {0.01f, 0.8f};
    engine.reflect_and_mutate(pool, high_friction);
    assert(pool.is_axis[1] == 0); // Relativized to Variable x!
    assert(pool.resistor_x[1] > config.min_resistor_x); // Impedance released!

    std::cout << "[PASS] DynamicCoordinateEngine tests successful!" << std::endl;
}

int main() {
    test_causal_trace_pool();
    test_dynamic_coordinate_engine();
    std::cout << "ALL C++ CAUSAL TOPOLOGY TESTS PASSED!" << std::endl;
    return 0;
}
