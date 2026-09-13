#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "causal_engine/core/causal_field_accelerator.hpp"

int main() {
    std::cout << "=== Running C++ CausalFieldAccelerator Test ===" << std::endl;

    causal_engine::CausalFieldAccelerator accelerator;

    // 1. Initialize control points
    std::vector<causal_engine::ControlPoint> points(2);
    points[0].pos[0] = 1.0; points[0].pos[1] = 0.0; points[0].pos[2] = 0.0; points[0].pos[3] = 0.5;
    points[0].weight = 1.0;

    points[1].pos[0] = 0.0; points[1].pos[1] = 1.0; points[1].pos[2] = 0.0; points[1].pos[3] = 0.2;
    points[1].weight = 1.0;

    // 2. Define topological edge tension (0 -> 1 with tension 2.5)
    std::vector<std::pair<int, int>> edges = {{0, 1}};
    std::vector<double> tensions = {2.5};

    double telos[4] = {2.0, 2.0, 1.0, 1.0};

    // 3. Compute initial system energy
    double initial_energy = accelerator.compute_system_energy(points, edges, tensions, telos);
    std::cout << "Initial System Energy: " << initial_energy << std::endl;
    assert(initial_energy > 0.0);

    // 4. Perform 10 parallel step transitions
    for (int step = 1; step <= 10; ++step) {
        accelerator.step_parallel(points, edges, tensions, telos, 0.05, 0.85);
        double current_energy = accelerator.compute_system_energy(points, edges, tensions, telos);
        std::cout << "Step " << step << " | Point 0 Pos: ["
                  << points[0].pos[0] << ", " << points[0].pos[1] << ", "
                  << points[0].pos[2] << ", " << points[0].pos[3] << "] | Energy: "
                  << current_energy << std::endl;
    }

    double final_energy = accelerator.compute_system_energy(points, edges, tensions, telos);
    std::cout << "Final System Energy: " << final_energy << std::endl;

    // Energy should decrease as system converges towards Telos attractor
    assert(final_energy < initial_energy);

    std::cout << "=== C++ CausalFieldAccelerator Test PASSED ===" << std::endl;
    return 0;
}
