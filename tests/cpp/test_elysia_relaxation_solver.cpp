#include "elysia/ElysiaRelaxationSolver.h"
#include "elysia/causal_erosion_kernel.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Elysia C++ Relaxation Solver & Erosion Kernel...\n";

    // 1. Test Relaxation Solver
    elysia::ElysiaRelaxationSolver solver(12);
    assert(solver.GetNumNodes() == 12);
    assert(solver.GetCurrentMode() == "normal");

    // Impact
    solver.ApplyImpact(2, 3.0f, 3.0f, 3.0f);
    assert(solver.GetAccumulatedTension() > 5.0f);

    solver.Step(0.05f);
    assert(solver.GetCurrentMode() == "berserk");
    std::cout << "ElysiaRelaxationSolver passed!\n";

    // 2. Test Erosion Kernel Solver
    elysia::CausalErosionKernelSolver erosion_solver(100);
    for (int i = 0; i < 110; ++i) {
        erosion_solver.ErodeAtTrajectory(0.1f * i, 0.1f * i, 0.0f);
    }

    assert(erosion_solver.IsPhaseTransformed() == true);
    erosion_solver.Step(0.016f);
    std::cout << "CausalErosionKernelSolver O(1) Phase Transition passed!\n";

    std::cout << "All C++ Solvers verified successfully!\n";
    return 0;
}
