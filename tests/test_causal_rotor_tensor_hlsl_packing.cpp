#include "causal_rotor_tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace elysia;

int main() {
    std::cout << "[TEST] Starting Causal Rotor Tensor & HLSL Packing Verification...\n";

    // 1. Structure Alignment & Size Check
    std::cout << "  - Checking RotorNodeData struct size and alignment...\n";
    std::size_t structSize = sizeof(RotorNodeData);
    std::cout << "    RotorNodeData Size: " << structSize << " bytes\n";
    assert(structSize == 128 && "RotorNodeData must be exactly 128 bytes!");

    // Check offset alignment
    RotorNodeData dummy;
    size_t offsetQuat = offsetof(RotorNodeData, quaternion);
    size_t offsetOmega = offsetof(RotorNodeData, angularVelocity);
    size_t offsetDamping = offsetof(RotorNodeData, dampingBeta);
    size_t offsetHash = offsetof(RotorNodeData, spatialHashKey);
    size_t offsetEdges = offsetof(RotorNodeData, connectedNodeIndices);
    size_t offsetRatios = offsetof(RotorNodeData, gearRatios);

    std::cout << "    Offsets: quat=" << offsetQuat
              << ", omega=" << offsetOmega
              << ", damping=" << offsetDamping
              << ", hash=" << offsetHash
              << ", edges=" << offsetEdges
              << ", ratios=" << offsetRatios << "\n";

    assert(offsetQuat == 0);
    assert(offsetOmega == 16);
    assert(offsetDamping == 28);
    assert(offsetHash == 32);
    assert(offsetEdges == 64);
    assert(offsetRatios == 96);

    // 2. Morton 3D Code Encode / Decode Check
    std::cout << "  - Checking Morton 3D Code Interleaving...\n";
    uint32_t origX = 10, origY = 25, origZ = 42;
    uint64_t mortonCode = Morton3D::encode(origX, origY, origZ);
    uint32_t decX = 0, decY = 0, decZ = 0;
    Morton3D::decode(mortonCode, decX, decY, decZ);

    std::cout << "    Morton Encode (" << origX << ", " << origY << ", " << origZ << ") -> "
              << "0x" << std::hex << mortonCode << std::dec << "\n";
    std::cout << "    Morton Decode -> (" << decX << ", " << decY << ", " << decZ << ")\n";

    assert(decX == origX && decY == origY && decZ == origZ && "Morton 3D decode failed!");

    // 3. System Kinematics & Phase-Lock Convergence Test
    std::cout << "  - Testing CausalRotorTensorSystem Kinematics & Phase-Lock Convergence...\n";
    CausalRotorTensorSystem system(100);

    uint32_t n0 = system.addRotorNode(1, 1, 1, 0.05f);
    uint32_t n1 = system.addRotorNode(1, 1, 2, 0.05f);

    system.connectNodes(n0, n1, 1.0f);
    system.connectNodes(n1, n0, 1.0f);

    // Inject initial impulse to node 0
    system.injectImpulse(n0, 0.0f, 0.5f, 0.0f);

    // Step simulation for 50 steps
    for (int step = 0; step < 50; ++step) {
        system.stepSimulation(0.01f, 0.05f);
    }

    uint32_t attractors = system.getAttractorCount();
    std::cout << "    Attractor Basin Nodes Count after 50 steps: " << attractors << "\n";

    std::cout << "[SUCCESS] Causal Rotor Tensor HLSL Packing & Kinematics Test Passed Successfully!\n";
    return 0;
}
