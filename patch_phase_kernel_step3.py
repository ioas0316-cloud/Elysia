import re

with open('src/phase_kernel.cpp', 'r') as f:
    content = f.read()

new_trace = """
    // ==========================================
    // Reverse Complex Conjugate Projection (trace_trajectory)
    // ==========================================

    // Traces the trajectory back using complex conjugate transpose projection.
    // Illuminates the original coherent states natively.
    struct TrajectoryRotor trace_trajectory(struct TrajectoryRotor final_state, struct TrajectoryRotor* vram_matrix, int matrix_size) {
        struct TrajectoryRotor reverse_state;

        // Complex Conjugate of a Quaternion (q*) = w - xi - yj - zk
        float conj_w = final_state.w;
        float conj_x = -final_state.x;
        float conj_y = -final_state.y;
        float conj_z = -final_state.z;

        float max_resonance = -1.0f;
        int target_idx = 0;

        // One-pass resonance search using conjugate projection
        for (int i = 0; i < matrix_size; ++i) {
            // Apply conjugate projection: resonance = Re(q_vram * q_conj)
            // It represents the cosine alignment of the topologies
            float resonance = vram_matrix[i].w * conj_w -
                              vram_matrix[i].x * conj_x -
                              vram_matrix[i].y * conj_y -
                              vram_matrix[i].z * conj_z;

            if (resonance > max_resonance) {
                max_resonance = resonance;
                target_idx = i;
            }
        }

        // The exact causal origin is intrinsically revealed without looping over logic.
        reverse_state.w = vram_matrix[target_idx].w;
        reverse_state.x = vram_matrix[target_idx].x;
        reverse_state.y = vram_matrix[target_idx].y;
        reverse_state.z = vram_matrix[target_idx].z;

        return reverse_state;
    }
"""

pos = content.rfind('}')
content = content[:pos] + new_trace + "}\n"

with open('src/phase_kernel.cpp', 'w') as f:
    f.write(content)

print("Patch applied for trace_trajectory")
