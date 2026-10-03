#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// CUDA Volumetric Ray-Marching Kernel for Active Inference SDF
// ============================================================================
__device__ inline float query_sdf_device(
    float x, float y, float z,
    const float* k_audio, float omega_t, float amp, float sigma
) {
    float norm_pos = sqrtf(x * x + y * y + z * z + 1e-8f);
    float d_base = norm_pos - 1.0f; // 기본 반지름 1.0 구형 표면

    float phase = (k_audio[0] * x + k_audio[1] * y + k_audio[2] * z) - omega_t;
    float mask = expf(-fabsf(d_base) / sigma);

    return d_base + amp * mask * cosf(phase);
}

__global__ void render_sdf_raymarch_kernel(
    float3 cam_pos,
    float3 cam_dir,
    float3 cam_up,
    float3 cam_right,
    int width,
    int height,
    const float* k_audio,
    float omega_t,
    float amp,
    float sigma,
    float* __restrict__ frame_buffer_out // [width * height * 4] (RGBA)
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // 1. Normalized Device Coordinates (NDC) & Ray Direction
    float u = (2.0f * (px + 0.5f) / (float)width - 1.0f) * ((float)width / (float)height);
    float v = (1.0f - 2.0f * (py + 0.5f) / (float)height);

    float3 ray_dir = make_float3(
        cam_dir.x + u * cam_right.x + v * cam_up.x,
        cam_dir.y + u * cam_right.y + v * cam_up.y,
        cam_dir.z + u * cam_right.z + v * cam_up.z
    );
    float ray_len = sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
    ray_dir.x /= ray_len; ray_dir.y /= ray_len; ray_dir.z /= ray_len;

    // 2. Sphere Tracing Loop (Sphere Marching)
    float t = 0.1f; // Initial Near Plane
    float t_max = 10.0f;
    float hit_d = 1e5f;
    bool hit = false;

    #pragma unroll 32
    for (int step = 0; step < 64; ++step) {
        float3 pos = make_float3(
            cam_pos.x + t * ray_dir.x,
            cam_pos.y + t * ray_dir.y,
            cam_pos.z + t * ray_dir.z
        );

        hit_d = query_sdf_device(pos.x, pos.y, pos.z, k_audio, omega_t, amp, sigma);

        if (hit_d < 0.001f) { // Surface Hit Threshold
            hit = true;
            break;
        }
        t += hit_d * 0.8f; // Overshooting 방지 가율 스텝
        if (t > t_max) break;
    }

    int pixel_idx = (py * width + px) * 4;

    // 3. Shading & Output Color Assignment
    if (hit) {
        float3 hit_pos = make_float3(
            cam_pos.x + t * ray_dir.x,
            cam_pos.y + t * ray_dir.y,
            cam_pos.z + t * ray_dir.z
        );

        // Analytic Normal Derivative (3D Exact)
        float eps = 0.001f;
        float nx = query_sdf_device(hit_pos.x + eps, hit_pos.y, hit_pos.z, k_audio, omega_t, amp, sigma) -
                   query_sdf_device(hit_pos.x - eps, hit_pos.y, hit_pos.z, k_audio, omega_t, amp, sigma);
        float ny = query_sdf_device(hit_pos.x, hit_pos.y + eps, hit_pos.z, k_audio, omega_t, amp, sigma) -
                   query_sdf_device(hit_pos.x, hit_pos.y - eps, hit_pos.z, k_audio, omega_t, amp, sigma);
        float nz = query_sdf_device(hit_pos.x, hit_pos.y, hit_pos.z + eps, k_audio, omega_t, amp, sigma) -
                   query_sdf_device(hit_pos.x, hit_pos.y, hit_pos.z - eps, k_audio, omega_t, amp, sigma);

        float n_len = sqrtf(nx * nx + ny * ny + nz * nz + 1e-8f);
        float3 N = make_float3(nx / n_len, ny / n_len, nz / n_len);

        // Lambertian Diffuse Lighting
        float3 light_dir = make_float3(0.577f, 0.577f, 0.577f);
        float diffuse = fmaxf(0.1f, N.x * light_dir.x + N.y * light_dir.y + N.z * light_dir.z);

        frame_buffer_out[pixel_idx + 0] = diffuse * 0.2f + 0.1f; // R
        frame_buffer_out[pixel_idx + 1] = diffuse * 0.7f + 0.2f; // G (Resonance Emerald)
        frame_buffer_out[pixel_idx + 2] = diffuse * 0.9f + 0.3f; // B
        frame_buffer_out[pixel_idx + 3] = 1.0f;                  // Alpha
    } else {
        // Background Color (Deep Space Sky)
        frame_buffer_out[pixel_idx + 0] = 0.02f;
        frame_buffer_out[pixel_idx + 1] = 0.02f;
        frame_buffer_out[pixel_idx + 2] = 0.05f;
        frame_buffer_out[pixel_idx + 3] = 1.0f;
    }
}
