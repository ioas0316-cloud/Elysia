#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// CUDA Kernel for Cl(3,1) SoA Coalesced Weaving
__global__ void cl31_weave_soa_coalesced_kernel(
    const float* __restrict__ A_s,
    const float* __restrict__ A_v0, const float* __restrict__ A_v1,
    const float* __restrict__ A_v2, const float* __restrict__ A_v3,
    const float* __restrict__ B_s,
    const float* __restrict__ B_v0, const float* __restrict__ B_v1,
    const float* __restrict__ B_v2, const float* __restrict__ B_v3,
    float* __restrict__ Out_s,
    float* __restrict__ Out_b0, float* __restrict__ Out_b1, float* __restrict__ Out_b2,
    float* __restrict__ Out_b3, float* __restrict__ Out_b4, float* __restrict__ Out_b5,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Coalesced loads
    float a_s  = A_s[idx];
    float a_v0 = A_v0[idx]; float a_v1 = A_v1[idx];
    float a_v2 = A_v2[idx]; float a_v3 = A_v3[idx];

    float b_s  = B_s[idx];
    float b_v0 = B_v0[idx]; float b_v1 = B_v1[idx];
    float b_v2 = B_v2[idx]; float b_v3 = B_v3[idx];

    // Geometric product scalar component under Minkowski metric (+,-,-,-)
    float out_s = a_s * b_s + a_v0 * b_v0 - a_v1 * b_v1 - a_v2 * b_v2 - a_v3 * b_v3;

    // Wedge product components (1D Causal lines -> 2D Bivector context sheet)
    float out_b0 = a_v0 * b_v1 - a_v1 * b_v0; // e01
    float out_b1 = a_v0 * b_v2 - a_v2 * b_v0; // e02
    float out_b2 = a_v0 * b_v3 - a_v3 * b_v0; // e03
    float out_b3 = a_v1 * b_v2 - a_v2 * b_v1; // e12
    float out_b4 = a_v2 * b_v3 - a_v3 * b_v2; // e23
    float out_b5 = a_v3 * b_v1 - a_v1 * b_v3; // e31

    // Coalesced writes
    Out_s[idx]  = out_s;
    Out_b0[idx] = out_b0; Out_b1[idx] = out_b1; Out_b2[idx] = out_b2;
    Out_b3[idx] = out_b3; Out_b4[idx] = out_b4; Out_b5[idx] = out_b5;
}

extern "C" void launch_cl31_soa_weave_cuda(
    const float* d_A_s, const float* d_A_v0, const float* d_A_v1, const float* d_A_v2, const float* d_A_v3,
    const float* d_B_s, const float* d_B_v0, const float* d_B_v1, const float* d_B_v2, const float* d_B_v3,
    float* d_Out_s, float* d_Out_b0, float* d_Out_b1, float* d_Out_b2,
    float* d_Out_b3, float* d_Out_b4, float* d_Out_b5,
    int N)
{
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cl31_weave_soa_coalesced_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_A_s, d_A_v0, d_A_v1, d_A_v2, d_A_v3,
        d_B_s, d_B_v0, d_B_v1, d_B_v2, d_B_v3,
        d_Out_s, d_Out_b0, d_Out_b1, d_Out_b2,
        d_Out_b3, d_Out_b4, d_Out_b5,
        N);
    cudaDeviceSynchronize();
}
