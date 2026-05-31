import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Update the host function to actually transfer memory from Host to Device, run the kernel, and transfer back.
# This implements the "Magnetic Phase Alignment" perfectly:
# The host (Python) passes down the lattice, the GPU aligns it via MTJ field mechanics, and returns the aligned matrix.
old_host_func = r"extern \"C\" void launch_project_phase_tensor.*?cudaDeviceSynchronize\(\);\n    \}"
new_host_func = """extern "C" void launch_project_phase_tensor(struct TrajectoryRotor incoming_state, struct TrajectoryRotor* h_vram_matrix, int matrix_size) {
        // Allocate Device Memory (The Magnetic Lattice on GPU)
        struct TrajectoryRotor* d_vram_matrix;
        size_t bytes = matrix_size * sizeof(struct TrajectoryRotor);
        cudaMalloc(&d_vram_matrix, bytes);

        // Copy current unaligned/semi-aligned state from Host to Device
        cudaMemcpy(d_vram_matrix, h_vram_matrix, bytes, cudaMemcpyHostToDevice);

        // Launch Kernel (Apply External Magnetic Field for Phase Alignment)
        int threadsPerBlock = 256;
        int blocksPerGrid = (matrix_size + threadsPerBlock - 1) / threadsPerBlock;

        project_phase_tensor_kernel<<<blocksPerGrid, threadsPerBlock>>>(incoming_state, d_vram_matrix, matrix_size);

        // Wait for all GPU threads to finish weaving
        cudaDeviceSynchronize();

        // Copy the freshly aligned Magnetic Tapestry back to the Host (Observation / Zero-resistance Retrieval)
        cudaMemcpy(h_vram_matrix, d_vram_matrix, bytes, cudaMemcpyDeviceToHost);

        // Free Device Memory
        cudaFree(d_vram_matrix);
    }"""

content = re.sub(old_host_func, new_host_func, content, flags=re.DOTALL)

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
