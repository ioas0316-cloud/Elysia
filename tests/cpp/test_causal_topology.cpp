#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cassert>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

struct alignas(64) SIMDTransform3D {
    float basis[9];
    float pad0;
    float origin[3];
    float pad1;
};

void simd_transform_multiply(const SIMDTransform3D &p_parent, const SIMDTransform3D &p_local, SIMDTransform3D &r_out) {
#if defined(__AVX2__)
    // Basis multiplication
    // p_parent.basis: 3x3
    // p_local.basis: 3x3
    const float *p_b = p_parent.basis;
    const float *l_b = p_local.basis;

    r_out.basis[0] = p_b[0] * l_b[0] + p_b[1] * l_b[3] + p_b[2] * l_b[6];
    r_out.basis[1] = p_b[0] * l_b[1] + p_b[1] * l_b[4] + p_b[2] * l_b[7];
    r_out.basis[2] = p_b[0] * l_b[2] + p_b[1] * l_b[5] + p_b[2] * l_b[8];

    r_out.basis[3] = p_b[3] * l_b[0] + p_b[4] * l_b[3] + p_b[5] * l_b[6];
    r_out.basis[4] = p_b[3] * l_b[1] + p_b[4] * l_b[4] + p_b[5] * l_b[7];
    r_out.basis[5] = p_b[3] * l_b[2] + p_b[4] * l_b[5] + p_b[5] * l_b[8];

    r_out.basis[6] = p_b[6] * l_b[0] + p_b[7] * l_b[3] + p_b[8] * l_b[6];
    r_out.basis[7] = p_b[6] * l_b[1] + p_b[7] * l_b[4] + p_b[8] * l_b[7];
    r_out.basis[8] = p_b[6] * l_b[2] + p_b[7] * l_b[5] + p_b[8] * l_b[8];

    r_out.origin[0] = p_b[0] * p_local.origin[0] + p_b[1] * p_local.origin[1] + p_b[2] * p_local.origin[2] + p_parent.origin[0];
    r_out.origin[1] = p_b[3] * p_local.origin[0] + p_b[4] * p_local.origin[1] + p_b[5] * p_local.origin[2] + p_parent.origin[1];
    r_out.origin[2] = p_b[6] * p_local.origin[0] + p_b[7] * p_local.origin[1] + p_b[8] * p_local.origin[2] + p_parent.origin[2];
#else
    const float *p_b = p_parent.basis;
    const float *l_b = p_local.basis;

    r_out.basis[0] = p_b[0] * l_b[0] + p_b[1] * l_b[3] + p_b[2] * l_b[6];
    r_out.basis[1] = p_b[0] * l_b[1] + p_b[1] * l_b[4] + p_b[2] * l_b[7];
    r_out.basis[2] = p_b[0] * l_b[2] + p_b[1] * l_b[5] + p_b[2] * l_b[8];

    r_out.basis[3] = p_b[3] * l_b[0] + p_b[4] * l_b[3] + p_b[5] * l_b[6];
    r_out.basis[4] = p_b[3] * l_b[1] + p_b[4] * l_b[4] + p_b[5] * l_b[7];
    r_out.basis[5] = p_b[3] * l_b[2] + p_b[4] * l_b[5] + p_b[5] * l_b[8];

    r_out.basis[6] = p_b[6] * l_b[0] + p_b[7] * l_b[3] + p_b[8] * l_b[6];
    r_out.basis[7] = p_b[6] * l_b[1] + p_b[7] * l_b[4] + p_b[8] * l_b[7];
    r_out.basis[8] = p_b[6] * l_b[2] + p_b[7] * l_b[5] + p_b[8] * l_b[8];

    r_out.origin[0] = p_b[0] * p_local.origin[0] + p_b[1] * p_local.origin[1] + p_b[2] * p_local.origin[2] + p_parent.origin[0];
    r_out.origin[1] = p_b[3] * p_local.origin[0] + p_b[4] * p_local.origin[1] + p_b[5] * p_local.origin[2] + p_parent.origin[1];
    r_out.origin[2] = p_b[6] * p_local.origin[0] + p_b[7] * p_local.origin[1] + p_b[8] * p_local.origin[2] + p_parent.origin[2];
#endif
}

class TestCausalServerSIMD {
public:
    std::vector<SIMDTransform3D> local_transforms;
    std::vector<SIMDTransform3D> global_transforms;
    std::vector<int32_t> parent_indices;
    std::vector<uint8_t> dirty_flags;
    std::vector<int32_t> cell_depths;
    std::vector<std::vector<int32_t>> depth_layers;

    SIMDTransform3D create_identity() {
        SIMDTransform3D t{};
        t.basis[0] = 1.0f; t.basis[4] = 1.0f; t.basis[8] = 1.0f;
        return t;
    }

    int32_t register_cell(int32_t parent_idx = -1, float origin_x = 0.0f, float origin_y = 0.0f, float origin_z = 0.0f) {
        int32_t new_idx = (int32_t)local_transforms.size();
        int32_t depth = 0;
        if (parent_idx >= 0 && parent_idx < new_idx) {
            depth = cell_depths[parent_idx] + 1;
        }

        SIMDTransform3D t = create_identity();
        t.origin[0] = origin_x;
        t.origin[1] = origin_y;
        t.origin[2] = origin_z;

        local_transforms.push_back(t);
        global_transforms.push_back(t);
        parent_indices.push_back(parent_idx);
        dirty_flags.push_back(1);
        cell_depths.push_back(depth);

        if ((int32_t)depth_layers.size() <= depth) {
            depth_layers.resize(depth + 1);
        }
        depth_layers[depth].push_back(new_idx);

        return new_idx;
    }

    void shift_variable(int32_t cell_idx, float dx, float dy, float dz) {
        if (cell_idx < 0 || cell_idx >= (int32_t)local_transforms.size()) return;
        local_transforms[cell_idx].origin[0] += dx;
        local_transforms[cell_idx].origin[1] += dy;
        local_transforms[cell_idx].origin[2] += dz;
        dirty_flags[cell_idx] = 1;
    }

    void propagate() {
        for (size_t layer = 0; layer < depth_layers.size(); layer++) {
            for (int32_t idx : depth_layers[layer]) {
                int32_t parent_idx = parent_indices[idx];
                if (dirty_flags[idx] || (parent_idx >= 0 && dirty_flags[parent_idx])) {
                    if (parent_idx >= 0) {
                        simd_transform_multiply(global_transforms[parent_idx], local_transforms[idx], global_transforms[idx]);
                    } else {
                        global_transforms[idx] = local_transforms[idx];
                    }
                    dirty_flags[idx] = 1;
                }
            }
        }
        std::fill(dirty_flags.begin(), dirty_flags.end(), 0);
    }
};

int main() {
    std::cout << "[TEST] Running Causal Topology Engine Mechanics Standalone Verification..." << std::endl;

    TestCausalServerSIMD server;
    int32_t root = server.register_cell(-1, 0.0f, 0.0f, 0.0f);
    int32_t child = server.register_cell(root, 2.0f, 0.0f, 0.0f);
    int32_t grandchild = server.register_cell(child, 0.0f, 3.0f, 0.0f);

    std::cout << "Root Depth: " << server.cell_depths[root] << std::endl;
    std::cout << "Child Depth: " << server.cell_depths[child] << std::endl;
    std::cout << "Grandchild Depth: " << server.cell_depths[grandchild] << std::endl;

    server.propagate();

    std::cout << "Grandchild Global Origin: ("
              << server.global_transforms[grandchild].origin[0] << ", "
              << server.global_transforms[grandchild].origin[1] << ", "
              << server.global_transforms[grandchild].origin[2] << ")" << std::endl;

    assert(std::abs(server.global_transforms[grandchild].origin[0] - 2.0f) < 1e-4);
    assert(std::abs(server.global_transforms[grandchild].origin[1] - 3.0f) < 1e-4);

    // Shift root
    server.shift_variable(root, 5.0f, 1.0f, -1.0f);
    server.propagate();

    assert(std::abs(server.global_transforms[root].origin[0] - 5.0f) < 1e-4);
    assert(std::abs(server.global_transforms[child].origin[0] - 7.0f) < 1e-4);
    assert(std::abs(server.global_transforms[grandchild].origin[0] - 7.0f) < 1e-4);
    assert(std::abs(server.global_transforms[grandchild].origin[1] - 4.0f) < 1e-4);

    std::cout << "  ✓ Direct Parent-Child-Grandchild Variable Propagation Verified." << std::endl;

    // 100,000 Cells Benchmark
    std::cout << "[TEST] Running 100,000 Cells Propagation Benchmark..." << std::endl;
    TestCausalServerSIMD bench_server;
    const int TOTAL_CELLS = 100000;

    auto start_setup = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TOTAL_CELLS; i++) {
        int parent = (i >= 10000) ? (i % 10000) : -1;
        bench_server.register_cell(parent, 1.0f, 0.0f, 0.0f);
    }
    auto end_setup = std::chrono::high_resolution_clock::now();
    double setup_ms = std::chrono::duration<double, std::milli>(end_setup - start_setup).count();

    std::cout << "  ✓ 100,000 Cells Registered & Depth-Sorted in " << setup_ms << " ms." << std::endl;

    double total_us = 0.0;
    for (int iter = 0; iter < 10; iter++) {
        for (int i = 0; i < 10000; i++) {
            bench_server.shift_variable(i, 0.1f, 0.0f, 0.0f);
        }
        auto start_prop = std::chrono::high_resolution_clock::now();
        bench_server.propagate();
        auto end_prop = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end_prop - start_prop).count();
        total_us += us;
    }

    double avg_us = total_us / 10.0;
    std::cout << "  ✓ 100,000 Cells Avg Propagation Time: " << avg_us << " us (" << (avg_us / 1000.0) << " ms)." << std::endl;
    std::cout << "[SUCCESS] Causal Topology Engine Verification Complete!" << std::endl;

    return 0;
}
