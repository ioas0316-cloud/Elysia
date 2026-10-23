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
        const float* __restrict__ k_wave,     // [64]
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

        float3 rotated_pos = rotate_vector_by_rotor(pos, rotor);

        float norm_x = sqrtf(rotated_pos.x * rotated_pos.x +
                             rotated_pos.y * rotated_pos.y +
                             rotated_pos.z * rotated_pos.z + 1e-8f);
        float d_base = norm_x - 1.0f;

        float phase = (k_wave[0] * rotated_pos.x +
                       k_wave[1] * rotated_pos.y +
                       k_wave[2] * rotated_pos.z) - omega_t;

        float cos_p = cosf(phase);
        float sin_p = sinf(phase);
        float mask = expf(-fabsf(d_base) / sigma);

        d_out[idx] = d_base + amplitude * mask * cos_p;

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

        QuaternionRotor inv_rotor = { rotor.w, -rotor.x, -rotor.y, -rotor.z };
        float3 world_normal = rotate_vector_by_rotor(rot_normal, inv_rotor);

        float n_len = sqrtf(world_normal.x * world_normal.x +
                            world_normal.y * world_normal.y +
                            world_normal.z * world_normal.z + 1e-8f);

        normal_out[idx * 3 + 0] = world_normal.x / n_len;
        normal_out[idx * 3 + 1] = world_normal.y / n_len;
        normal_out[idx * 3 + 2] = world_normal.z / n_len;
    }

    void launch_clifford_sdf_kernel_stream(
        torch::Tensor query_pos,
        torch::Tensor k_wave,
        torch::Tensor rotor_params,
        float omega_t,
        float amplitude,
        float sigma,
        torch::Tensor d_out,
        torch::Tensor normal_out,
        uintptr_t stream_ptr
    ) {
        int N_points = query_pos.size(0);
        int threads = 256;
        int blocks = (N_points + threads - 1) / threads;

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

        QuaternionRotor rotor = {
            rotor_params[0].item<float>(),
            rotor_params[1].item<float>(),
            rotor_params[2].item<float>(),
            rotor_params[3].item<float>()
        };

        clifford_rotor_sdf_kernel<<<blocks, threads, 0, stream>>>(
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
    void launch_clifford_sdf_kernel_stream(
        torch::Tensor query_pos,
        torch::Tensor k_wave,
        torch::Tensor rotor_params,
        float omega_t,
        float amplitude,
        float sigma,
        torch::Tensor d_out,
        torch::Tensor normal_out,
        uintptr_t stream_ptr
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

    # ============================================================================
    # 3. VRAM 메모리 추적 유틸리티
    # ============================================================================
    def reset_vram_stats():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def get_vram_metrics_mb():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return allocated, max_allocated, reserved

    def execute_benchmark_mode(clifford_cuda, use_multi_stream: bool, num_streams: int = 4, num_runs: int = 100):
        device = torch.device("cuda")
        tokenizer = HangulKroneckerTokenizer().to(device)

        sample_text = "엘리시아 텐서 필드 코어 엔진" # 15 자 음절
        num_points = 262_144 # 256K Points

        query_pos = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        rotor_params = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        d_out = torch.empty(num_points, device=device, dtype=torch.float32)
        normal_out = torch.empty(num_points, 3, device=device, dtype=torch.float32)

        reset_vram_stats()

        streams = [torch.cuda.Stream() for _ in range(num_streams)] if use_multi_stream else [torch.cuda.default_stream()]

        k_stream = tokenizer(sample_text).to(device)

        # Warmup
        for i in range(10):
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream):
                clifford_cuda.launch_clifford_sdf_kernel_stream(
                    query_pos, k_stream[i % len(sample_text)], rotor_params,
                    0.05, 0.2, 0.15, d_out, normal_out, stream.cuda_stream
                )
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for run in range(num_runs):
            stream = streams[run % len(streams)]
            k_vec = k_stream[run % len(sample_text)]

            with torch.cuda.stream(stream):
                clifford_cuda.launch_clifford_sdf_kernel_stream(
                    query_pos, k_vec, rotor_params,
                    run * 0.01, 0.2, 0.15, d_out, normal_out, stream.cuda_stream
                )

        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_us = (total_time_ms / num_runs) * 1000.0
        fps = 1000.0 / (total_time_ms / num_runs)

        alloc_mb, max_alloc_mb, reserved_mb = get_vram_metrics_mb()

        return {
            "mode": "Multi-Stream Pipeline" if use_multi_stream else "Single-Stream",
            "avg_latency_us": avg_latency_us,
            "fps": fps,
            "allocated_mb": alloc_mb,
            "peak_vram_mb": max_alloc_mb,
            "reserved_mb": reserved_mb
        }

def run_integrated_benchmark():
    if not HAS_TORCH or not torch.cuda.is_available():
        print("[elysia_hangul_clifford_multi_stream_bench] CUDA support not detected. Exiting gracefully.")
        return

    print("[1/4] CUDA Multi-Stream 커널 인라인 컴파일 중...")
    clifford_cuda = load_inline(
        name="clifford_cuda_multi_stream_bench",
        cpp_sources=cpp_declarations,
        cuda_sources=cuda_source,
        functions=["launch_clifford_sdf_kernel_stream"],
        extra_cuda_cflags=["-O3"]
    )
    print("[1/4] CUDA 커널 컴파일 완료!")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print("\n[2/4] GPU 디바이스 정보 감지 완료:")
    print(f" ▶ Device Name : {gpu_name}")
    print(f" ▶ Total VRAM : {total_vram_gb:.2f} GB")

    print("\n[3/4] Single-Stream vs Multi-Stream 성능 측정 중...")

    single_res = execute_benchmark_mode(clifford_cuda, use_multi_stream=False)
    multi_res = execute_benchmark_mode(clifford_cuda, use_multi_stream=True, num_streams=4)

    speedup = ((single_res["avg_latency_us"] - multi_res["avg_latency_us"]) / single_res["avg_latency_us"]) * 100.0

    print("\n[4/4] 벤치마크 결과 비교 분석:")
    print("=" * 72)
    print(f"{'Metric':<25} | {'Single-Stream':<18} | {'Multi-Stream (4 Streams)':<20}")
    print("-" * 72)
    print(f"{'Mean Latency':<25} | {single_res['avg_latency_us']:>10.2f} µs     | {multi_res['avg_latency_us']:>12.2f} µs")
    print(f"{'Frame Rate (FPS)':<25} | {single_res['fps']:>10.1f} FPS    | {multi_res['fps']:>12.1f} FPS")
    print(f"{'Current Allocated VRAM':<25} | {single_res['allocated_mb']:>10.2f} MB     | {multi_res['allocated_mb']:>12.2f} MB")
    print(f"{'Peak VRAM Usage':<25} | {single_res['peak_vram_mb']:>10.2f} MB     | {multi_res['peak_vram_mb']:>12.2f} MB")
    print(f"{'Reserved VRAM Pool':<25} | {single_res['reserved_mb']:>10.2f} MB     | {multi_res['reserved_mb']:>12.2f} MB")
    print("=" * 72)
    print(f" ▶ 비동기 파이프라인 지연 시간 단축률 (Speedup): {speedup:+.2f} %")
    print("=" * 72)

if __name__ == "__main__":
    run_integrated_benchmark()
