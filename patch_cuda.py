import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Replace project_phase_tensor to become a CUDA kernel and use Field physics + Magnetic Imprint

cuda_kernel_code = """
    // ==========================================
    // 3. Electromagnetic Cognitive Field (Field Physics)
    // 4. Magnetic Imprint Layer (MTJ/SOT)
    // ==========================================

    // Native CUDA Kernel: Single-pass GEMM equivalent Weaver (Weft threads over Warp Tensor)
    __global__ void project_phase_tensor_kernel(struct TrajectoryRotor incoming_state, struct TrajectoryRotor* vram_matrix, int matrix_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < matrix_size) {
            // Field Physics: Attraction & Repulsion
            // Calculate Phase Alignment (Dot Product between the Spin Rotors)
            float alignment = incoming_state.w * vram_matrix[idx].w +
                              incoming_state.x * vram_matrix[idx].x +
                              incoming_state.y * vram_matrix[idx].y +
                              incoming_state.z * vram_matrix[idx].z;

            // Electromagnetic Field Kinetic Momentum
            // If alignment is positive and high -> Strong Attraction (Synchronize)
            // If alignment is negative or low -> Repulsion (Push away, decouple)
            float force = 0.0f;
            if (alignment > 0.8f) {
                force = 0.1f * alignment; // Attractive Force
            } else if (alignment < -0.5f) {
                force = -0.05f; // Repulsive Force (decoupling)
            } else {
                force = 0.01f; // Neutral drift
            }

            // Apply Kinetic Momentum to Local VRAM Rotor (The Tapestry)
            float dw = (incoming_state.w - vram_matrix[idx].w) * force;
            float dx = (incoming_state.x - vram_matrix[idx].x) * force;
            float dy = (incoming_state.y - vram_matrix[idx].y) * force;
            float dz = (incoming_state.z - vram_matrix[idx].z) * force;

            vram_matrix[idx].w += dw;
            vram_matrix[idx].x += dx;
            vram_matrix[idx].y += dy;
            vram_matrix[idx].z += dz;

            // Renormalize to maintain Quaternionic Spin Boundary (Magnetic State)
            float mag = sqrtf(vram_matrix[idx].w * vram_matrix[idx].w +
                              vram_matrix[idx].x * vram_matrix[idx].x +
                              vram_matrix[idx].y * vram_matrix[idx].y +
                              vram_matrix[idx].z * vram_matrix[idx].z);

            if (mag > 0.000001f) {
                vram_matrix[idx].w /= mag;
                vram_matrix[idx].x /= mag;
                vram_matrix[idx].y /= mag;
                vram_matrix[idx].z /= mag;
            }
        }
    }

    // Host function to launch the __global__ kernel
    extern "C" void launch_project_phase_tensor(struct TrajectoryRotor incoming_state, struct TrajectoryRotor* d_vram_matrix, int matrix_size) {
        // Assume d_vram_matrix is already allocated on device via cudaMalloc in Python/C++ wrapper
        int threadsPerBlock = 256;
        int blocksPerGrid = (matrix_size + threadsPerBlock - 1) / threadsPerBlock;

        project_phase_tensor_kernel<<<blocksPerGrid, threadsPerBlock>>>(incoming_state, d_vram_matrix, matrix_size);

        // Wait for all GPU threads to finish weaving
        cudaDeviceSynchronize();
    }
"""

# Pattern to replace the old project_phase_tensor
old_project = r"void project_phase_tensor\(.*?\).*?\}\n    \}"

content = re.sub(old_project, cuda_kernel_code, content, flags=re.DOTALL)

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
