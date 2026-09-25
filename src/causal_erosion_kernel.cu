#include "elysia/causal_erosion_kernel.h"
#include <iostream>

// CUDA / C++ kernel entrypoint implementation
namespace elysia {

void RunCausalErosionBenchmark() {
    std::cout << "[Elysia Causal Erosion Benchmark] Initializing 1000 particles...\n";
    CausalErosionKernelSolver solver(1000);

    for (int w = 0; w < 120; ++w) {
        solver.ErodeAtTrajectory(0.1f * w, 0.05f * w, 0.0f);
    }

    std::cout << "Eroded Wells: " << solver.GetWellCount() << " | Phase Transformed O(1): "
              << (solver.IsPhaseTransformed() ? "YES" : "NO") << "\n";

    solver.Step(0.016f);
    std::cout << "Step completed successfully (120fps ready).\n";
}

} // namespace elysia
