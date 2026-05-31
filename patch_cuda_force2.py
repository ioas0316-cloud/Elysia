import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Let's check why there's no modification happening in the test.
# incoming w=0, x=1, y=0, z=0
# vram w=1, x=0, y=0, z=0
# alignment = 0*1 + 1*0 = 0.
# resistance_sensor = 1 - abs(0) = 1.0f
# kinetic_twist_force = 0.5f * 1.0 = 0.5f.
# dw = (0.0 - 1.0) * 0.5 = -0.5.
# vram_matrix[idx].w += dw -> 1.0 - 0.5 = 0.5.
# dx = (1.0 - 0.0) * 0.5 = 0.5.
# vram_matrix[idx].x += dx -> 0.0 + 0.5 = 0.5.

# Ah, the kernel launch might be failing because of some setup.
# Let's add a cudaGetErrorString print inside the host function to see if kernel launch succeeds.

old_host_func = r"cudaFree\(d_vram_matrix\);\n    \}"
new_host_func = """cudaFree(d_vram_matrix);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\\n", cudaGetErrorString(err));
        }
    }"""

content = re.sub(old_host_func, new_host_func, content)

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
