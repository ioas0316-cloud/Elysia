#include "causal_server_simd.h"
#include "core/os/os.h"
#include "core/string/godot_string.h"
#include "core/string/print_string.h"
#include "scene/resources/3d/box_mesh.h"
#include <chrono>

struct BenchmarkResult {
    String summary_text;
    Vector<double> iteration_times_us;
};

BenchmarkResult run_causal_simd_100k_benchmark_with_data() {
    BenchmarkResult result;

    CausalServerSIMD *causal = CausalServerSIMD::get_singleton();
    if (!causal) {
        result.summary_text = "CausalServerSIMD singleton not initialized!";
        return result;
    }

    print_line("==================================================");
    print_line("[CausalServerSIMD] 100,000 Cells Benchmark Start");
    print_line("==================================================");

    Ref<BoxMesh> dummy_mesh;
    dummy_mesh.instantiate();
    RID mesh_rid = dummy_mesh.is_valid() ? dummy_mesh->get_rid() : RID();

    const int32_t TOTAL_CELLS = 100000;

    uint64_t setup_start = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;

    LocalVector<int32_t> cell_indices;
    cell_indices.resize(TOTAL_CELLS);

    Transform3D init_transform;
    init_transform.origin = Vector3(1.0f, 0.0f, 0.0f);

    for (int32_t i = 0; i < TOTAL_CELLS; i++) {
        int32_t parent_idx = -1;
        if (i >= 10000) {
            parent_idx = i % 10000;
        }
        cell_indices[i] = causal->register_causal_cell_simd(mesh_rid, parent_idx, init_transform);
    }

    uint64_t setup_end = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
    print_line(vformat("1. Topology Setup Time: %d us (%.2f ms)",
                       (setup_end - setup_start), (setup_end - setup_start) / 1000.0f));

    for (int32_t i = 0; i < 10000; i++) {
        causal->shift_variable(cell_indices[i], Vector3(0.1f, 0.5f, -0.2f));
    }

    causal->propagate_and_project_parallel();

    const int TEST_ITERATIONS = 10;
    double total_elapsed_us = 0.0;

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        for (int32_t i = 0; i < 10000; i++) {
            causal->shift_variable(cell_indices[i], Vector3(0.01f, 0.0f, 0.01f));
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        causal->propagate_and_project_parallel();

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_us = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_elapsed_us += elapsed_us;
        result.iteration_times_us.push_back(elapsed_us);

        print_line(vformat("   - Run #%d Execution Time: %.2f us (%.3f ms)", iter + 1, elapsed_us, elapsed_us / 1000.0f));
    }

    double avg_us = total_elapsed_us / TEST_ITERATIONS;
    double avg_ms = avg_us / 1000.0f;

    result.summary_text = vformat(
        "100,000 Causal Cells Benchmark Completed!\n\n"
        "- Total Iterations: %d\n"
        "- Average Execution Time: %.2f us (%.3f ms)\n"
        "- Target Engine Pipeline: Zero-Node Direct RID\n"
        "- SIMD Architecture: SoA / AVX2 / NEON FMA\n"
        "- Threading: WorkerThreadPool Lock-Free",
        TEST_ITERATIONS, avg_us, avg_ms
    );

    print_line("--------------------------------------------------");
    print_line(result.summary_text);
    print_line("==================================================");

    return result;
}

void run_causal_simd_100k_benchmark() {
    run_causal_simd_100k_benchmark_with_data();
}
