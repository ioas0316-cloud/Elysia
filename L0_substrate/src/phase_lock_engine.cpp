#include "phase_lock_engine.hpp"
#include <cmath>

#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

void launch_phase_lock_kernel(
    const at::Tensor& X, const at::Tensor& V,
    const at::Tensor& row_ptr, const at::Tensor& col_idx,
    at::Tensor& C_edge, at::Tensor& M_edge, at::Tensor& Phi_edge,
    float R_cut, float gamma_m, float alpha, float beta,
    float tau_c, float c0, float lambda_c, float dt,
    cudaStream_t stream
);
#endif

namespace ElysiaEngine {

PhaseLockEnginePipeline::PhaseLockEnginePipeline()
    : config_(Config()) {}

PhaseLockEnginePipeline::PhaseLockEnginePipeline(const Config& config)
    : config_(config) {}

void PhaseLockEnginePipeline::validate_inputs(
    const at::Tensor& X, const at::Tensor& V,
    const at::Tensor& row_ptr, const at::Tensor& col_idx,
    const at::Tensor& C_edge, const at::Tensor& M_edge
) {
    TORCH_CHECK(X.device() == V.device() && X.device() == row_ptr.device() &&
                X.device() == col_idx.device() && X.device() == C_edge.device() &&
                X.device() == M_edge.device(),
                "All input tensors must reside on the same device.");
    TORCH_CHECK(X.is_contiguous() && V.is_contiguous() && row_ptr.is_contiguous() &&
                col_idx.is_contiguous() && C_edge.is_contiguous() && M_edge.is_contiguous(),
                "All input tensors must be contiguous in memory.");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && V.dtype() == torch::kFloat32,
                "Node coordinates X and velocities V must be Float32.");
    TORCH_CHECK(row_ptr.dtype() == torch::kInt32 && col_idx.dtype() == torch::kInt32,
                "CSR pointers row_ptr and col_idx must be Int32.");
}

std::tuple<at::Tensor, at::Tensor> PhaseLockEnginePipeline::step(
    const at::Tensor& X,
    const at::Tensor& V,
    const at::Tensor& row_ptr,
    const at::Tensor& col_idx,
    at::Tensor& C_edge,
    at::Tensor& M_edge
) {
    validate_inputs(X, V, row_ptr, col_idx, C_edge, M_edge);

    if (X.is_cuda()) {
#ifdef WITH_CUDA
        int64_t num_edges = col_idx.size(0);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(X.device());
        at::Tensor Phi_edge = torch::empty({num_edges}, options);

        c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(X.device().index());

        launch_phase_lock_kernel(
            X, V, row_ptr, col_idx,
            C_edge, M_edge, Phi_edge,
            config_.R_cut, config_.gamma_m, config_.alpha, config_.beta,
            config_.tau_c, config_.c0, config_.lambda_c, config_.dt,
            stream.stream()
        );

        at::Tensor A_solid = (Phi_edge >= config_.phi_solid).to(torch::kFloat32);
        at::Tensor A_liquid = ((Phi_edge < config_.phi_solid) & (Phi_edge >= config_.phi_gas)).to(torch::kFloat32);
        at::Tensor A_edge = A_solid + A_liquid * torch::sigmoid(Phi_edge);

        return std::make_tuple(Phi_edge, A_edge);
#else
        TORCH_CHECK(false, "Engine compiled without CUDA support, but received CUDA tensors.");
#endif
    } else {
        return step_cpu(X, V, row_ptr, col_idx, C_edge, M_edge);
    }
}

std::tuple<at::Tensor, at::Tensor> PhaseLockEnginePipeline::step_cpu(
    const at::Tensor& X,
    const at::Tensor& V,
    const at::Tensor& row_ptr,
    const at::Tensor& col_idx,
    at::Tensor& C_edge,
    at::Tensor& M_edge
) {
    int64_t num_nodes = X.size(0);
    int64_t num_edges = col_idx.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    at::Tensor Phi_edge = torch::empty({num_edges}, options);
    at::Tensor A_edge = torch::empty({num_edges}, options);

    const float* x_ptr = X.data_ptr<float>();
    const float* v_ptr = V.data_ptr<float>();
    const int32_t* r_ptr = row_ptr.data_ptr<int32_t>();
    const int32_t* c_ptr = col_idx.data_ptr<int32_t>();

    float* C_ptr = C_edge.data_ptr<float>();
    float* M_ptr = M_edge.data_ptr<float>();
    float* Phi_ptr = Phi_edge.data_ptr<float>();
    float* A_ptr = A_edge.data_ptr<float>();

    float r_cut_sq = config_.R_cut * config_.R_cut;
    int32_t feat_dim = X.size(1);

    for (int64_t i = 0; i < num_nodes; ++i) {
        int32_t start_e = r_ptr[i];
        int32_t end_e = r_ptr[i + 1];

        for (int32_t e = start_e; e < end_e; ++e) {
            int32_t j = c_ptr[e];

            float dist_sq = 0.0f;
            float vel_sq = 0.0f;

            for (int32_t d = 0; d < feat_dim; ++d) {
                float dx = x_ptr[i * feat_dim + d] - x_ptr[j * feat_dim + d];
                float dv = v_ptr[i * feat_dim + d] - v_ptr[j * feat_dim + d];
                dist_sq += dx * dx;
                vel_sq += dv * dv;
            }

            dist_sq += 1e-6f;

            if (dist_sq > r_cut_sq) {
                M_ptr[e] = 0.0f;
                C_ptr[e] = 0.0f;
                Phi_ptr[e] = 0.0f;
                A_ptr[e] = 0.0f;
                continue;
            }

            float dist = std::sqrt(dist_sq);
            float v_rel = std::sqrt(vel_sq);

            float m_target = config_.alpha * v_rel + (config_.beta / dist);
            float m_val = (1.0f - config_.gamma_m) * M_ptr[e] + config_.gamma_m * m_target;
            M_ptr[e] = m_val;

            float sigmoid_decay = 1.0f / (1.0f + std::exp(-(config_.tau_c - m_val)));
            float c_val = C_ptr[e] * sigmoid_decay + config_.c0 * std::exp(-config_.lambda_c * dist);
            C_ptr[e] = c_val;

            float phi_val = c_val / (m_val + 1e-5f);
            Phi_ptr[e] = phi_val;

            if (phi_val >= config_.phi_solid) {
                A_ptr[e] = 1.0f;
            } else if (phi_val >= config_.phi_gas) {
                A_ptr[e] = 1.0f / (1.0f + std::exp(-phi_val));
            } else {
                A_ptr[e] = 0.0f;
            }
        }
    }

    return std::make_tuple(Phi_edge, A_edge);
}

} // namespace ElysiaEngine
