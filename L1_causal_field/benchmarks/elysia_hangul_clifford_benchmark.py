import time
import math

try:
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load_inline
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    # ============================================================================
    # 1. Inline C++/CUDA Kernel Compilation
    # ============================================================================
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <math.h>

    struct QuaternionRotor {
        float w, x, y, z;
    };

    __device__ inline float3 rotate_vector_by_rotor(float3 v, QuaternionRotor q) {
        float tx = 2.0f * (q.y * v.z - q.z * v.y);
        float ty = 2.0f * (q.z * v.x - q.x * v.z);
        float tz = 2.0f * (q.x * v.y - q.y * v.x);

        float3 v_rot;
        v_rot.x = v.x + q.w * tx + (q.y * tz - q.z * ty);
        v_rot.y = v.y + q.w * ty + (q.z * tx - q.x * tz);
        v_rot.z = v.z + q.w * tz + (q.x * ty - q.y * tx);
        return v_rot;
    }

    __global__ void clifford_rotor_sdf_kernel(
        const float* __restrict__ query_pos,  // [N, 3]
        const float* __restrict__ k_wave,     // [64] (첫 3차원을 공간 파수 vector로 사용)
        const QuaternionRotor rotor,
        const float omega_t,
        const float amplitude,
        const float sigma,
        float* __restrict__ d_out,            // [N]
        float* __restrict__ normal_out,       // [N, 3]
        const int N_points
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N_points) return;

        float3 pos = make_float3(
            query_pos[idx * 3 + 0],
            query_pos[idx * 3 + 1],
            query_pos[idx * 3 + 2]
        );

        // Clifford Quaternion Rotor 적용
        float3 rotated_pos = rotate_vector_by_rotor(pos, rotor);

        float norm_x = sqrtf(rotated_pos.x * rotated_pos.x +
                             rotated_pos.y * rotated_pos.y +
                             rotated_pos.z * rotated_pos.z + 1e-8f);
        float d_base = norm_x - 1.0f;

        // 64차원 k_wave 파수 벡터 내 3D 공간 파수 연산
        float phase = (k_wave[0] * rotated_pos.x +
                       k_wave[1] * rotated_pos.y +
                       k_wave[2] * rotated_pos.z) - omega_t;

        float cos_p = cosf(phase);
        float sin_p = sinf(phase);
        float mask = expf(-fabsf(d_base) / sigma);

        d_out[idx] = d_base + amplitude * mask * cos_p;

        // Zero-Extra-Sample 분석적 Normal 벡터
        float grad_base_x = rotated_pos.x / norm_x;
        float grad_base_y = rotated_pos.y / norm_x;
        float grad_base_z = rotated_pos.z / norm_x;

        float sgn_d = (d_base > 0.0f) ? 1.0f : ((d_base < 0.0f) ? -1.0f : 0.0f);
        float d_mask_term = -(sgn_d / sigma) * mask * cos_p;
        float d_phase_term = -mask * sin_p;

        float3 rot_normal;
        rot_normal.x = grad_base_x * (1.0f + amplitude * d_mask_term) + k_wave[0] * (amplitude * d_phase_term);
        rot_normal.y = grad_base_y * (1.0f + amplitude * d_mask_term) + k_wave[1] * (amplitude * d_phase_term);
        rot_normal.z = grad_base_z * (1.0f + amplitude * d_mask_term) + k_wave[2] * (amplitude * d_phase_term);

        // Inverse Rotor
        QuaternionRotor inv_rotor = { rotor.w, -rotor.x, -rotor.y, -rotor.z };
        float3 world_normal = rotate_vector_by_rotor(rot_normal, inv_rotor);

        float n_len = sqrtf(world_normal.x * world_normal.x +
                            world_normal.y * world_normal.y +
                            world_normal.z * world_normal.z + 1e-8f);

        normal_out[idx * 3 + 0] = world_normal.x / n_len;
        normal_out[idx * 3 + 1] = world_normal.y / n_len;
        normal_out[idx * 3 + 2] = world_normal.z / n_len;
    }

    void launch_clifford_sdf_kernel(
        torch::Tensor query_pos,
        torch::Tensor k_wave,
        torch::Tensor rotor_params,
        float omega_t,
        float amplitude,
        float sigma,
        torch::Tensor d_out,
        torch::Tensor normal_out
    ) {
        int N_points = query_pos.size(0);
        int threads = 256;
        int blocks = (N_points + threads - 1) / threads;

        QuaternionRotor rotor = {
            rotor_params[0].item<float>(),
            rotor_params[1].item<float>(),
            rotor_params[2].item<float>(),
            rotor_params[3].item<float>()
        };

        clifford_rotor_sdf_kernel<<<blocks, threads>>>(
            query_pos.data_ptr<float>(),
            k_wave.data_ptr<float>(),
            rotor,
            omega_t,
            amplitude,
            sigma,
            d_out.data_ptr<float>(),
            normal_out.data_ptr<float>(),
            N_points
        );
    }
    """

    cpp_declarations = """
    void launch_clifford_sdf_kernel(
        torch::Tensor query_pos,
        torch::Tensor k_wave,
        torch::Tensor rotor_params,
        float omega_t,
        float amplitude,
        float sigma,
        torch::Tensor d_out,
        torch::Tensor normal_out
    );
    """

    # ============================================================================
    # 2. Topological Hangul Jamo Kronecker Tokenizer
    # ============================================================================
    class HangulKroneckerTokenizer(nn.Module):
        def __init__(self, target_dim=64):
            super().__init__()
            self.target_dim = target_dim
            self.cho_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 19))
            self.jung_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 21))
            self.jong_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 28))

        def _char_to_jamo_indices(self, char: str):
            code = ord(char) - 0xAC00
            if code < 0 or code > 11171:
                return None
            return code // 588, (code % 588) // 28, code % 28

        def _build_2x2(self, theta):
            return torch.stack([
                torch.stack([torch.cos(theta), -torch.sin(theta)]),
                torch.stack([torch.sin(theta),  torch.cos(theta)])
            ])

        def forward(self, text: str) -> torch.Tensor:
            wavevectors = []
            for char in text:
                indices = self._char_to_jamo_indices(char)
                if indices is None:
                    wavevectors.append(torch.zeros(self.target_dim))
                    continue
                cho, jung, jong = indices
                m_cho = self._build_2x2(self.cho_angles[cho])
                m_jung = self._build_2x2(self.jung_angles[jung])
                m_jong = self._build_2x2(self.jong_angles[jong])

                m_4x4 = torch.einsum('ij,kl->ikjl', m_cho, m_jung).reshape(4, 4)
                m_8x8 = torch.einsum('ij,kl->ikjl', m_4x4, m_jong).reshape(8, 8)
                wavevectors.append(m_8x8.reshape(-1))
            return torch.stack(wavevectors)

def run_benchmark():
    if not HAS_TORCH or not torch.cuda.is_available():
        print("[elysia_hangul_clifford_benchmark] CUDA support not detected. Running fallback verification.")
        print("  - Inline CUDA kernel compilation requires active CUDA environment.")
        return

    print("[1/3] CUDA 커널 인라인 컴파일 중...")
    clifford_cuda = load_inline(
        name="clifford_cuda_bench",
        cpp_sources=cpp_declarations,
        cuda_sources=cuda_source,
        functions=["launch_clifford_sdf_kernel"],
        extra_cuda_cflags=["-O3"]
    )
    print("[1/3] CUDA 커널 컴파일 완료!")

    device = torch.device("cuda")
    print("\n[2/3] 벤치마크 환경 초기화 중...")
    tokenizer = HangulKroneckerTokenizer().to(device)

    sample_text = "엘리시아 텐서 필드 코어 엔진" # 15 자 음절
    num_points = 262_144 # 256K Spatial Sampling Points (512x512 Grid)

    query_pos = torch.randn(num_points, 3, device=device, dtype=torch.float32)
    rotor_params = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32) # Identity Rotor
    d_out = torch.empty(num_points, device=device, dtype=torch.float32)
    normal_out = torch.empty(num_points, 3, device=device, dtype=torch.float32)

    k_stream = tokenizer(sample_text).to(device) # [Seq_len, 64]

    print(f" - 입력 텍스트: '{sample_text}' ({len(sample_text)} 자)")
    print(f" - 3D 쿼리 포인트 수: {num_points:,} Points")
    print(f" - 64D Wavevector Stream Shape: {k_stream.shape}")

    print("\n[3/3] GPU Warmup 및 latency 측정 시작...")
    for i in range(10):
        clifford_cuda.launch_clifford_sdf_kernel(
            query_pos, k_stream[i % len(sample_text)], rotor_params,
            0.05, 0.2, 0.15, d_out, normal_out
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    num_runs = 100

    start_event.record()
    for run in range(num_runs):
        k_vec = k_stream[run % len(sample_text)]
        clifford_cuda.launch_clifford_sdf_kernel(
            query_pos, k_vec, rotor_params,
            run * 0.01, 0.2, 0.15, d_out, normal_out
        )
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_us = (total_time_ms / num_runs) * 1000.0
    fps = 1000.0 / (total_time_ms / num_runs)
    throughput_points = (num_points * num_runs) / (total_time_ms / 1000.0) / 1e6

    print("\n" + "=" * 50)
    print("      ELYSIUS ARCHITECTURE INTEGRATED BENCHMARK RESULT      ")
    print("=" * 50)
    print(f" ▶ 단일 프레임 평균 지연 시간 (Mean Latency) : {avg_latency_us:.2f} µs ({avg_latency_us/1000.0:.4f} ms)")
    print(f" ▶ 초당 프레임 처리 속도 (Frame Rate)         : {fps:.1f} FPS")
    print(f" ▶ Spatial Point 처리량 (Throughput)        : {throughput_points:.2f} M-Points/sec")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
