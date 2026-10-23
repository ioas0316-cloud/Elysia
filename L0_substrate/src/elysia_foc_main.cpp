#include <iostream>
#include <vector>
#include <cmath>
#include "elysia_scheduler.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    constexpr int BATCH_SIZE = 10000; // 10k vector simulation
    size_t vec_size_bytes = BATCH_SIZE * 3 * sizeof(float);
    size_t angle_size_bytes = BATCH_SIZE * sizeof(float);

    std::cout << "========================================================\n";
    std::cout << "  Elysia Engine: FOC & Clifford Wave Computing Pipeline  \n";
    std::cout << "========================================================\n";

    // 1. Initialize FOC Scheduler (3072MB limit, 15% safety margin)
    ElysiaFOCScheduler scheduler(3072.0f, 0.15f, 0.008f);

    // 2. Host data preparation
    std::vector<float> h_abc(BATCH_SIZE * 3);
    std::vector<float> h_angles(BATCH_SIZE);
    std::vector<float> h_dq0(BATCH_SIZE * 3);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        float t = i * 0.01f;
        h_abc[i * 3 + 0] = std::sin(t);
        h_abc[i * 3 + 1] = std::sin(t - 2.0f * M_PI / 3.0f);
        h_abc[i * 3 + 2] = std::sin(t + 2.0f * M_PI / 3.0f);
        h_angles[i] = t;
    }

    // 3. GPU Memory allocation
    float *d_abc = nullptr, *d_angles = nullptr, *d_dq0 = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_abc, vec_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_angles, angle_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_dq0, vec_size_bytes));

    CUDA_CHECK(cudaMemcpy(d_abc, h_abc.data(), vec_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_angles, h_angles.data(), angle_size_bytes, cudaMemcpyHostToDevice));

    std::cout << ">> Elysia FOC Scheduler initialized successfully." << std::endl;

    // 4. Step 1: Normal VRAM Operation Loop
    for (int step = 0; step < 5; ++step) {
        scheduler.step(d_abc, d_angles, d_dq0, BATCH_SIZE);
    }
    scheduler.synchronize();

    CUDA_CHECK(cudaMemcpy(h_dq0.data(), d_dq0, vec_size_bytes, cudaMemcpyDeviceToHost));
    std::cout << ">> Normal Step Sample d_dq0[0]: D=" << h_dq0[0]
              << ", Q=" << h_dq0[1] << ", Zero=" << h_dq0[2] << std::endl;

    // 5. Step 2: High VRAM Pressure Simulation (Flux Weakening active)
    std::cout << ">> Simulating VRAM spike (2800MB used)..." << std::endl;
    for (int step = 0; step < 5; ++step) {
        scheduler.step(d_abc, d_angles, d_dq0, BATCH_SIZE, 2800.0f);
    }
    scheduler.synchronize();

    CUDA_CHECK(cudaMemcpy(h_dq0.data(), d_dq0, vec_size_bytes, cudaMemcpyDeviceToHost));
    std::cout << ">> Flux Weakened Sample d_dq0[0]: D=" << h_dq0[0]
              << ", Q=" << h_dq0[1] << ", Zero=" << h_dq0[2] << std::endl;

    // 6. Step 3: Clifford 3-Phase Cognitive Pipeline (Gas -> Liquid -> Solid)
    std::cout << ">> Running Clifford 3-Phase Cognitive Pipeline..." << std::endl;
    int num_morphisms = 128;
    std::vector<Rotor3D> h_rotors(num_morphisms, {1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<Multivector3D> h_gas(num_morphisms, {0.0f, 1.0f, 0.5f, 0.0f, 0.1f, 0.2f, 0.0f, 0.0f});
    std::vector<Multivector3D> h_fluid(num_morphisms, {0.0f, 0.8f, 0.4f, 0.0f, 0.05f, 0.1f, 0.0f, 0.0f});

    Rotor3D *d_rotors = nullptr;
    Multivector3D *d_gas = nullptr, *d_fluid = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rotors, num_morphisms * sizeof(Rotor3D)));
    CUDA_CHECK(cudaMalloc((void**)&d_gas, num_morphisms * sizeof(Multivector3D)));
    CUDA_CHECK(cudaMalloc((void**)&d_fluid, num_morphisms * sizeof(Multivector3D)));

    CUDA_CHECK(cudaMemcpy(d_rotors, h_rotors.data(), num_morphisms * sizeof(Rotor3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gas, h_gas.data(), num_morphisms * sizeof(Multivector3D), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fluid, h_fluid.data(), num_morphisms * sizeof(Multivector3D), cudaMemcpyHostToDevice));

    // Gas -> Liquid Receptor Launch
    launch_elysia_open_system_receptor_kernel(d_gas, d_fluid, 0.5f, 0.01f, num_morphisms, scheduler.get_stream());
    scheduler.synchronize();

    std::cout << ">> Open-System Receptor successfully transformed Gas waves into Liquid momentum." << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_abc));
    CUDA_CHECK(cudaFree(d_angles));
    CUDA_CHECK(cudaFree(d_dq0));
    CUDA_CHECK(cudaFree(d_rotors));
    CUDA_CHECK(cudaFree(d_gas));
    CUDA_CHECK(cudaFree(d_fluid));

    std::cout << ">> Elysia FOC Engine pipeline completed cleanly without error." << std::endl;
    return 0;
}
