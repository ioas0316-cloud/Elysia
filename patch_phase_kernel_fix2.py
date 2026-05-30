import re

with open('src/phase_kernel.cpp', 'r') as f:
    content = f.read()

# I accidentally placed the new function INSIDE ascii_to_phase_wave. Let's fix that.
# Find the end of ascii_to_phase_wave correctly.
wrong_code = """            out_phase_tensors[i * 2] = cosf(angular_momentum);
            out_phase_tensors[i * 2 + 1] = sinf(angular_momentum);
        }
        // ==========================================
    // Isomorphic Phase Projection (Single-pass Gemm Equivalent)
    // ==========================================

    // Instead of 27-base pointer crawling, project the incoming state globally across the VRAM array
    void project_phase_tensor(struct TrajectoryRotor incoming_state, struct TrajectoryRotor* vram_matrix, int matrix_size) {
        // Purely topological projection over physical array
        // In full GPU implementation, this would be a single kernel launch block

        for (int i = 0; i < matrix_size; ++i) {
            // Natural coherence matching (Holographic Overlay)
            // Tension pulls local state towards the incoming state
            float dw = (incoming_state.w - vram_matrix[i].w) * 0.05f;
            float dx = (incoming_state.x - vram_matrix[i].x) * 0.05f;
            float dy = (incoming_state.y - vram_matrix[i].y) * 0.05f;
            float dz = (incoming_state.z - vram_matrix[i].z) * 0.05f;

            vram_matrix[i].w += dw;
            vram_matrix[i].x += dx;
            vram_matrix[i].y += dy;
            vram_matrix[i].z += dz;

            // Renormalize local matrix element to maintain quantum coherence bounds
            float mag = std::sqrt(vram_matrix[i].w * vram_matrix[i].w +
                                  vram_matrix[i].x * vram_matrix[i].x +
                                  vram_matrix[i].y * vram_matrix[i].y +
                                  vram_matrix[i].z * vram_matrix[i].z);
            if (mag > 0.000001f) {
                vram_matrix[i].w /= mag;
                vram_matrix[i].x /= mag;
                vram_matrix[i].y /= mag;
                vram_matrix[i].z /= mag;
            }
        }
    }
}
"""

right_code = """            out_phase_tensors[i * 2] = cosf(angular_momentum);
            out_phase_tensors[i * 2 + 1] = sinf(angular_momentum);
        }
    }
        // ==========================================
    // Isomorphic Phase Projection (Single-pass Gemm Equivalent)
    // ==========================================

    // Instead of 27-base pointer crawling, project the incoming state globally across the VRAM array
    void project_phase_tensor(struct TrajectoryRotor incoming_state, struct TrajectoryRotor* vram_matrix, int matrix_size) {
        // Purely topological projection over physical array
        // In full GPU implementation, this would be a single kernel launch block

        for (int i = 0; i < matrix_size; ++i) {
            // Natural coherence matching (Holographic Overlay)
            // Tension pulls local state towards the incoming state
            float dw = (incoming_state.w - vram_matrix[i].w) * 0.05f;
            float dx = (incoming_state.x - vram_matrix[i].x) * 0.05f;
            float dy = (incoming_state.y - vram_matrix[i].y) * 0.05f;
            float dz = (incoming_state.z - vram_matrix[i].z) * 0.05f;

            vram_matrix[i].w += dw;
            vram_matrix[i].x += dx;
            vram_matrix[i].y += dy;
            vram_matrix[i].z += dz;

            // Renormalize local matrix element to maintain quantum coherence bounds
            float mag = std::sqrt(vram_matrix[i].w * vram_matrix[i].w +
                                  vram_matrix[i].x * vram_matrix[i].x +
                                  vram_matrix[i].y * vram_matrix[i].y +
                                  vram_matrix[i].z * vram_matrix[i].z);
            if (mag > 0.000001f) {
                vram_matrix[i].w /= mag;
                vram_matrix[i].x /= mag;
                vram_matrix[i].y /= mag;
                vram_matrix[i].z /= mag;
            }
        }
    }
}
"""

content = content.replace(wrong_code, right_code)

with open('src/phase_kernel.cpp', 'w') as f:
    f.write(content)

print("Fix 2 applied")
