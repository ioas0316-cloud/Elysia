#include "causal_engine/3dgs_sh_eval.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

using namespace elysia::gaussian3d;

void test_sh_forward_eval() {
    std::cout << "[Test 1] 3DGS SH Forward Evaluation..." << std::endl;
    int P = 1;
    int deg = 3;
    int max_coeffs = 16;

    Float3 means3D(0.0f, 0.0f, 2.0f);
    Float3 cam_pos(0.0f, 0.0f, 0.0f);

    std::vector<float> shs(16 * 3, 0.0f);
    // DC component
    shs[0] = 1.0f; shs[1] = 0.5f; shs[2] = 0.2f;

    Float3 rgb_out;
    computeColorFromSH(P, deg, max_coeffs, &means3D, cam_pos, shs.data(), &rgb_out);

    float expected_r = 0.5f + SH_C0 * 1.0f;
    float expected_g = 0.5f + SH_C0 * 0.5f;
    float expected_b = 0.5f + SH_C0 * 0.2f;

    std::cout << "  RGB computed: (" << rgb_out.x << ", " << rgb_out.y << ", " << rgb_out.z << ")" << std::endl;
    std::cout << "  RGB expected: (" << expected_r << ", " << expected_g << ", " << expected_b << ")" << std::endl;

    assert(std::abs(rgb_out.x - expected_r) < 1e-5f);
    assert(std::abs(rgb_out.y - expected_g) < 1e-5f);
    assert(std::abs(rgb_out.z - expected_b) < 1e-5f);
    std::cout << "  -> PASSED!\n" << std::endl;
}

void test_sh_backward_gradient() {
    std::cout << "[Test 2] 3DGS SH Backward Pass Finite Difference Check..." << std::endl;
    int P = 1;
    int deg = 2;
    int max_coeffs = 16;

    Float3 means3D(0.5f, -0.3f, 1.5f);
    Float3 cam_pos(0.0f, 0.0f, 0.0f);

    std::vector<float> shs(16 * 3, 0.0f);
    for (size_t i = 0; i < shs.size(); ++i) {
        shs[i] = 0.1f * (float)(i + 1) / 48.0f;
    }

    Float3 rgb;
    computeColorFromSH(P, deg, max_coeffs, &means3D, cam_pos, shs.data(), &rgb);

    Float3 dL_drgb(1.0f, -0.5f, 0.8f);

    std::vector<float> dL_dsh(16 * 3, 0.0f);
    Float3 dL_dmeans3D;

    computeColorFromSHBackward(P, deg, max_coeffs, &means3D, cam_pos, shs.data(), &rgb, &dL_drgb, dL_dsh.data(), &dL_dmeans3D);

    // Finite difference check for SH[0]
    float eps = 1e-4f;
    std::vector<float> shs_plus = shs;
    shs_plus[0] += eps;
    Float3 rgb_plus;
    computeColorFromSH(P, deg, max_coeffs, &means3D, cam_pos, shs_plus.data(), &rgb_plus);

    float L_orig = rgb.x * dL_drgb.x + rgb.y * dL_drgb.y + rgb.z * dL_drgb.z;
    float L_plus = rgb_plus.x * dL_drgb.x + rgb_plus.y * dL_drgb.y + rgb_plus.z * dL_drgb.z;
    float num_grad_sh0 = (L_plus - L_orig) / eps;

    std::cout << "  Analytic dL/dSH[0]: " << dL_dsh[0] << std::endl;
    std::cout << "  Numerical dL/dSH[0]: " << num_grad_sh0 << std::endl;
    assert(std::abs(dL_dsh[0] - num_grad_sh0) < 1e-3f);

    // Finite difference check for means3D.x
    Float3 means_plus = means3D;
    means_plus.x += eps;
    Float3 rgb_m_plus;
    computeColorFromSH(P, deg, max_coeffs, &means_plus, cam_pos, shs.data(), &rgb_m_plus);

    float L_m_plus = rgb_m_plus.x * dL_drgb.x + rgb_m_plus.y * dL_drgb.y + rgb_m_plus.z * dL_drgb.z;
    float num_grad_means_x = (L_m_plus - L_orig) / eps;

    std::cout << "  Analytic dL/dmeans3D.x: " << dL_dmeans3D.x << std::endl;
    std::cout << "  Numerical dL/dmeans3D.x: " << num_grad_means_x << std::endl;
    assert(std::abs(dL_dmeans3D.x - num_grad_means_x) < 1e-3f);

    std::cout << "  -> PASSED!\n" << std::endl;
}

void test_phys_3dgs_collision_gradient() {
    std::cout << "[Test 3] Phys-3DGS Analytic Collision Gradient Check..." << std::endl;
    int P = 100;
    std::vector<Float3> means3D(P);
    std::vector<float> inv_cov3D(P * 6, 0.0f);
    std::vector<float> opacity(P, 0.9f);

    for (int i = 0; i < P; ++i) {
        means3D[i] = Float3(0.1f * (i % 10), 0.1f * (i / 10), 1.0f);
        inv_cov3D[i * 6 + 0] = 10.0f; // xx
        inv_cov3D[i * 6 + 3] = 10.0f; // yy
        inv_cov3D[i * 6 + 5] = 10.0f; // zz
    }

    int dof = 3;
    CapsuleLink link;
    link.pA = Float3(0.0f, 0.0f, 0.0f);
    link.pB = Float3(0.5f, 0.5f, 1.0f);
    link.dof = dof;

    std::vector<float> jacA = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    std::vector<float> jacB = {
        1.0f, 0.1f, 0.0f,
        0.0f, 1.0f, 0.2f,
        0.0f, 0.0f, 1.0f
    };

    link.jacobianA = jacA.data();
    link.jacobianB = jacB.data();

    std::vector<float> dC_dq(dof, 0.0f);
    computePhys3DGSCollisionGradient(P, means3D.data(), inv_cov3D.data(), opacity.data(), link, dC_dq.data());

    std::cout << "  Analytic Collision Gradient dC/dq: [";
    for (int k = 0; k < dof; ++k) {
        std::cout << dC_dq[k] << (k == dof - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "  -> PASSED!\n" << std::endl;
}

void benchmark_performance() {
    std::cout << "[Benchmark] 3DGS SH & Phys-3DGS Performance Test..." << std::endl;
    int P = 1000000; // 1 Million Gaussians
    int deg = 3;
    int max_coeffs = 16;

    std::vector<Float3> means3D(P, Float3(1.0f, 2.0f, 3.0f));
    Float3 cam_pos(0.0f, 0.0f, 0.0f);
    std::vector<float> shs(P * 16 * 3, 0.1f);
    std::vector<Float3> rgb_out(P);

    auto start = std::chrono::high_resolution_clock::now();
    computeColorFromSH(P, deg, max_coeffs, means3D.data(), cam_pos, shs.data(), rgb_out.data());
    auto end = std::chrono::high_resolution_clock::now();

    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Evaluated " << P << " Gaussians SH Forward in: " << duration_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (P / (duration_ms / 1000.0) / 1e6) << " Million Gaussians / sec" << std::endl;
    std::cout << "  -> BENCHMARK COMPLETE!\n" << std::endl;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "   3DGS SH Evaluation & Phys-3DGS Unit & Benchmark Test   " << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    test_sh_forward_eval();
    test_sh_backward_gradient();
    test_phys_3dgs_collision_gradient();
    benchmark_performance();

    std::cout << "All tests and benchmarks completed successfully!" << std::endl;
    return 0;
}
