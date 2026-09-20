#include <torch/extension.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void phase_lock_sparse_kernel(
    const float* __restrict__ X,          // [N, 3] Node Coordinates
    const float* __restrict__ V,          // [N, 3] Node Velocities
    const int32_t* __restrict__ row_ptr,  // [N + 1] CSR Row Pointers
    const int32_t* __restrict__ col_idx,  // [E] CSR Column Indices
    float* __restrict__ C_edge,           // [E] Constraint Tension (In/Out)
    float* __restrict__ M_edge,           // [E] Mobility (In/Out)
    float* __restrict__ Phi_edge,         // [E] Phase-Lock Index (Out)
    const int num_nodes,
    const float R_cut_sq,
    const float gamma_m,
    const float alpha,
    const float beta,
    const float tau_c,
    const float c0,
    const float lambda_c,
    const float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float3 pos_i = make_float3(X[i * 3], X[i * 3 + 1], X[i * 3 + 2]);
    float3 vel_i = make_float3(V[i * 3], V[i * 3 + 1], V[i * 3 + 2]);

    int start_e = row_ptr[i];
    int end_e = row_ptr[i + 1];

    for (int e = start_e; e < end_e; ++e) {
        int j = col_idx[e];

        float3 pos_j = make_float3(X[j * 3], X[j * 3 + 1], X[j * 3 + 2]);
        float3 vel_j = make_float3(V[j * 3], V[j * 3 + 1], V[j * 3 + 2]);

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float dist_sq = dx * dx + dy * dy + dz * dz + 1e-6f;

        if (dist_sq > R_cut_sq) {
            M_edge[e] = 0.0f;
            C_edge[e] = 0.0f;
            Phi_edge[e] = 0.0f;
            continue;
        }

        float dvx = vel_i.x - vel_j.x;
        float dvy = vel_i.y - vel_j.y;
        float dvz = vel_i.z - vel_j.z;
        float v_rel = sqrtf(dvx * dvx + dvy * dvy + dvz * dvz);
        float dist = sqrtf(dist_sq);

        // Mobility & Constraint Fused State Transition
        float m_target = alpha * v_rel + (beta / dist);
        float m_val = (1.0f - gamma_m) * M_edge[e] + gamma_m * m_target;
        M_edge[e] = m_val;

        float sigmoid_decay = 1.0f / (1.0f + expf(-(tau_c - m_val)));
        float c_val = C_edge[e] * sigmoid_decay + c0 * expf(-lambda_c * dist);
        C_edge[e] = c_val;

        Phi_edge[e] = c_val / (m_val + 1e-5f);
    }
}

// Host Launcher Function called by C++ Pipeline
void launch_phase_lock_kernel(
    const at::Tensor& X,
    const at::Tensor& V,
    const at::Tensor& row_ptr,
    const at::Tensor& col_idx,
    at::Tensor& C_edge,
    at::Tensor& M_edge,
    at::Tensor& Phi_edge,
    float R_cut,
    float gamma_m,
    float alpha,
    float beta,
    float tau_c,
    float c0,
    float lambda_c,
    float dt,
    cudaStream_t stream
) {
    int num_nodes = X.size(0);
    int threads_per_block = 256;
    int blocks_per_grid = (num_nodes + threads_per_block - 1) / threads_per_block;

    phase_lock_sparse_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        X.data_ptr<float>(),
        V.data_ptr<float>(),
        row_ptr.data_ptr<int32_t>(),
        col_idx.data_ptr<int32_t>(),
        C_edge.data_ptr<float>(),
        M_edge.data_ptr<float>(),
        Phi_edge.data_ptr<float>(),
        num_nodes,
        R_cut * R_cut,
        gamma_m,
        alpha,
        beta,
        tau_c,
        c0,
        lambda_c,
        dt
    );
}
#endif
