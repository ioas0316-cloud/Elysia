#include <elysia/elysia_holographic_engine.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>

#if defined(HAS_OPENGL)
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_gl_interop.h>
#endif
#endif

#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 1024
#define GRID_DIM      128
#define TILE_DIM      8

#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
#define LAUNCH_KERNEL(kernel, grid, block, ...) kernel(__VA_ARGS__)
#else
#define LAUNCH_KERNEL(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#endif

// ----------------------------------------------------------------------------
// 1. Arcball Camera State
// ----------------------------------------------------------------------------
struct ArcballCamera {
    float yaw   = 0.4f;
    float pitch = 0.3f;
    float dist  = 170.0f;

    double last_x = 0.0;
    double last_y = 0.0;
    bool is_dragging = false;

    const float min_dist = 20.0f;
    const float max_dist = 500.0f;
    const float pitch_limit = 1.54f;
};

static ArcballCamera g_camera;

#if defined(HAS_OPENGL)
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_camera.is_dragging = true;
            glfwGetCursorPos(window, &g_camera.last_x, &g_camera.last_y);
        } else if (action == GLFW_RELEASE) {
            g_camera.is_dragging = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_camera.is_dragging) return;

    double dx = xpos - g_camera.last_x;
    double dy = ypos - g_camera.last_y;

    float sensitivity = 0.005f;
    g_camera.yaw   -= static_cast<float>(dx) * sensitivity;
    g_camera.pitch += static_cast<float>(dy) * sensitivity;
    g_camera.pitch = std::clamp(g_camera.pitch, -g_camera.pitch_limit, g_camera.pitch_limit);

    g_camera.last_x = xpos;
    g_camera.last_y = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    float zoom_sensitivity = 8.0f;
    g_camera.dist -= static_cast<float>(yoffset) * zoom_sensitivity;
    g_camera.dist = std::clamp(g_camera.dist, g_camera.min_dist, g_camera.max_dist);
}
#endif

// ----------------------------------------------------------------------------
// 2. CUDA Kernels
// ----------------------------------------------------------------------------

__global__ void Inject_Kronecker_Jamo_Seed_Kernel(float* d_p_curr, int3 dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim.x || y >= dim.y || z >= dim.z) return;

    // Cho 'ㄱ' (2x2) & Jung 'ㅏ' (2x2) Kronecker Product Matrix (4x4)
    const float A[2][2] = {{1.0f, 0.5f}, {0.0f, 1.0f}};
    const float B[2][2] = {{0.8f, 0.2f}, {1.0f, 0.4f}};

    int cx = dim.x / 2;
    int cy = dim.y / 2;
    int cz = dim.z / 2;

    if (abs(x - cx) < 2 && abs(y - cy) < 2 && abs(z - cz) < 2) {
        int i = (x - cx) + 2;
        int j = (y - cy) + 2;

        int a_i = i / 2, a_j = j / 2;
        int b_i = i % 2, b_j = j % 2;

        float k_val = A[a_i][a_j] * B[b_i][b_j];
        int idx = x + y * dim.x + z * (dim.x * dim.y);
        d_p_curr[idx] = k_val * 3.0f;
    }
}

__global__ void Inject_3Phase_Delta_Sources_Kernel(float* d_p_curr, int3 dim, float time) {
    int tid = threadIdx.x;
    if (tid >= 3) return;

    float radius = 24.0f;
    float phase_shift = tid * (2.0f * 3.14159265f / 3.0f); // 0, 120, 240 deg phase
    float angle = tid * (2.0f * 3.14159265f / 3.0f);

    int sx = static_cast<int>(dim.x * 0.5f + radius * cosf(angle));
    int sy = static_cast<int>(dim.y * 0.5f + radius * sinf(angle));
    int sz = dim.z / 2;

    int idx = sx + sy * dim.x + sz * (dim.x * dim.y);
    d_p_curr[idx] += sinf(time * 12.0f + phase_shift) * 2.5f;
}

__global__ void FDTD_Wave_Step_Kernel(const float* d_p_curr, float* d_p_next, int3 dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x <= 0 || x >= dim.x - 1 || y <= 0 || y >= dim.y - 1 || z <= 0 || z >= dim.z - 1) return;

    int idx = x + y * dim.x + z * (dim.x * dim.y);
    float p_c = d_p_curr[idx];

    float laplacian = d_p_curr[idx + 1] + d_p_curr[idx - 1] +
                      d_p_curr[idx + dim.x] + d_p_curr[idx - dim.x] +
                      d_p_curr[idx + dim.x * dim.y] + d_p_curr[idx - dim.x * dim.y] - 6.0f * p_c;

    d_p_next[idx] = p_c + 0.15f * laplacian;
}

__device__ inline float3 get_density_gradient(const float* grid, int3 pos, int3 dim) {
    int x = pos.x, y = pos.y, z = pos.z;
    if (x <= 0 || x >= dim.x - 1 || y <= 0 || y >= dim.y - 1 || z <= 0 || z >= dim.z - 1)
        return make_float3(0.0f, 0.0f, 0.0f);

    int stride_y = dim.x;
    int stride_z = dim.x * dim.y;
    int idx = x + y * stride_y + z * stride_z;

    float dx = fabsf(grid[idx + 1]) - fabsf(grid[idx - 1]);
    float dy = fabsf(grid[idx + stride_y]) - fabsf(grid[idx - stride_y]);
    float dz = fabsf(grid[idx + stride_z]) - fabsf(grid[idx - stride_z]);

    return make_float3(dx * 0.5f, dy * 0.5f, dz * 0.5f);
}

__global__ void CUDA_Unified_Raymarch_Kernel(
    const float* __restrict__ d_p_curr,
    uchar4*      __restrict__ d_pbo_frame,
    int2 res, int3 grid_dim, float time,
    float3 cam_pos, float3 view_u, float3 view_v, float3 view_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= res.x || y >= res.y) return;

    float u = (2.0f * x - res.x) / static_cast<float>(res.y);
    float v = (2.0f * y - res.y) / static_cast<float>(res.y);

    float fov_factor = 1.8f;
    float3 ray_dir;
    ray_dir.x = u * view_u.x + v * view_v.x + fov_factor * view_w.x;
    ray_dir.y = u * view_u.y + v * view_v.y + fov_factor * view_w.y;
    ray_dir.z = u * view_u.z + v * view_v.z + fov_factor * view_w.z;

    float inv_l = 1.0f / sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
    ray_dir.x *= inv_l; ray_dir.y *= inv_l; ray_dir.z *= inv_l;

    float3 ray_pos = cam_pos;
    float3 accum_color = make_float3(0.0f, 0.0f, 0.0f);
    float accum_transmittance = 1.0f;

    float3 light_pos = make_float3(40.0f * sinf(time), 50.0f, 40.0f * cosf(time));

    for (int step = 0; step < 140; ++step) {
        int gx = static_cast<int>(ray_pos.x + grid_dim.x * 0.5f);
        int gy = static_cast<int>(ray_pos.y + grid_dim.y * 0.5f);
        int gz = static_cast<int>(ray_pos.z + grid_dim.z * 0.5f);

        if (gx >= 0 && gx < grid_dim.x && gy >= 0 && gy < grid_dim.y && gz >= 0 && gz < grid_dim.z) {
            int idx = gx + gy * grid_dim.x + gz * (grid_dim.x * grid_dim.y);
            float wave_val = fabsf(d_p_curr[idx]);

            // 1. Clifford Gradient Ray Bending
            float3 grad = get_density_gradient(d_p_curr, make_int3(gx, gy, gz), grid_dim);
            float bend_strength = 0.35f;
            ray_dir.x += grad.x * bend_strength;
            ray_dir.y += grad.y * bend_strength;
            ray_dir.z += grad.z * bend_strength;

            float r_len = 1.0f / sqrtf(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
            ray_dir.x *= r_len; ray_dir.y *= r_len; ray_dir.z *= r_len;

            // 2. SDF Boundary Hybrid Coupling
            float dist_center = sqrtf(ray_pos.x * ray_pos.x + ray_pos.y * ray_pos.y + ray_pos.z * ray_pos.z);
            float sphere_sdf = dist_center - 18.0f;
            float sdf_alpha = fmaxf(0.0f, 1.0f - fabsf(sphere_sdf) * 0.1f);

            float total_density = (wave_val * 0.12f) + (sdf_alpha * 0.05f);

            if (total_density > 0.01f) {
                // 3. Volumetric Light Scattering
                float3 light_dir = make_float3(light_pos.x - ray_pos.x, light_pos.y - ray_pos.y, light_pos.z - ray_pos.z);
                float l_dist = sqrtf(light_dir.x * light_dir.x + light_dir.y * light_dir.y + light_dir.z * light_dir.z);
                float attenuation = 1.0f / (1.0f + 0.0005f * l_dist * l_dist);

                float absorption = total_density * 1.1f;
                accum_transmittance *= expf(-absorption * 1.2f);

                float3 scatter_col = make_float3(
                    0.2f + sinf(time + wave_val * 2.0f) * 0.3f,
                    0.5f + cosf(time * 0.8f + wave_val) * 0.4f,
                    0.9f
                );

                accum_color.x += scatter_col.x * absorption * accum_transmittance * attenuation;
                accum_color.y += scatter_col.y * absorption * accum_transmittance * attenuation;
                accum_color.z += scatter_col.z * absorption * accum_transmittance * attenuation;

                if (accum_transmittance < 0.02f) break;
            }
        }

        ray_pos.x += ray_dir.x * 1.1f;
        ray_pos.y += ray_dir.y * 1.1f;
        ray_pos.z += ray_dir.z * 1.1f;
    }

    if (d_pbo_frame) {
        int pixel_idx = x + y * res.x;
        d_pbo_frame[pixel_idx] = make_uchar4(
            static_cast<unsigned char>(fminf(accum_color.x * 255.0f, 255.0f)),
            static_cast<unsigned char>(fminf(accum_color.y * 255.0f, 255.0f)),
            static_cast<unsigned char>(fminf(accum_color.z * 255.0f, 255.0f)),
            255
        );
    }
}

inline float3 normalize_vec(float3 v) {
    float inv = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

inline float3 cross_vec(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// ----------------------------------------------------------------------------
// 3. Headless Offscreen Benchmark
// ----------------------------------------------------------------------------
void RunHeadlessBenchmark() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Elysia Unified Hologram Engine - Headless Mode Benchmark" << std::endl;
    std::cout << "==========================================================" << std::endl;

    int3 grid_dim = make_int3(GRID_DIM, GRID_DIM, GRID_DIM);
    size_t grid_voxels = grid_dim.x * grid_dim.y * grid_dim.z;
    size_t grid_bytes = grid_voxels * sizeof(float);

    int2 res = make_int2(WINDOW_WIDTH, WINDOW_HEIGHT);
    size_t frame_bytes = res.x * res.y * sizeof(uchar4);

    float *d_p_curr = nullptr, *d_p_next = nullptr;
    uchar4* d_pbo_frame = nullptr;

    cudaMalloc((void**)&d_p_curr, grid_bytes);
    cudaMalloc((void**)&d_p_next, grid_bytes);
    cudaMalloc((void**)&d_pbo_frame, frame_bytes);

    cudaMemset(d_p_curr, 0, grid_bytes);
    cudaMemset(d_p_next, 0, grid_bytes);

    dim3 grid_3d_block(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 grid_3d_grid(grid_dim.x / TILE_DIM, grid_dim.y / TILE_DIM, grid_dim.z / TILE_DIM);

    dim3 render_block(16, 16);
    dim3 render_grid((res.x + 15) / 16, (res.y + 15) / 16);

    LAUNCH_KERNEL(Inject_Kronecker_Jamo_Seed_Kernel, grid_3d_grid, grid_3d_block, d_p_curr, grid_dim);
    cudaDeviceSynchronize();

    float current_time = 0.0f;
    float3 cam_pos = make_float3(0.0f, 0.0f, -170.0f);
    float3 view_u  = make_float3(1.0f, 0.0f, 0.0f);
    float3 view_v  = make_float3(0.0f, 1.0f, 0.0f);
    float3 view_w  = make_float3(0.0f, 0.0f, 1.0f);

    const int test_frames = 50;
    std::cout << "Running " << test_frames << " simulation and raymarching frames..." << std::endl;

    for (int f = 0; f < test_frames; ++f) {
        current_time += 0.033f;

        LAUNCH_KERNEL(Inject_3Phase_Delta_Sources_Kernel, dim3(1,1,1), dim3(3,1,1), d_p_curr, grid_dim, current_time);

        LAUNCH_KERNEL(FDTD_Wave_Step_Kernel, grid_3d_grid, grid_3d_block, d_p_curr, d_p_next, grid_dim);
        std::swap(d_p_curr, d_p_next);

        LAUNCH_KERNEL(CUDA_Unified_Raymarch_Kernel, render_grid, render_block,
            d_p_curr, d_pbo_frame, res, grid_dim, current_time,
            cam_pos, view_u, view_v, view_w
        );
    }
    cudaDeviceSynchronize();

    std::cout << "Headless Benchmark Completed Successfully (" << test_frames << " frames executed)." << std::endl;

    cudaFree(d_p_curr);
    cudaFree(d_p_next);
    cudaFree(d_pbo_frame);
}

int main(int argc, char** argv) {
    bool headless = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--headless") {
            headless = true;
            break;
        }
    }

#if defined(HAS_OPENGL)
    if (headless) {
        RunHeadlessBenchmark();
        return 0;
    }

    if (!glfwInit()) {
        std::cout << "GLFW Initialization failed, falling back to Headless Benchmark Mode." << std::endl;
        RunHeadlessBenchmark();
        return 0;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Elysia Unified Hologram Engine", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cout << "Window Creation failed, falling back to Headless Benchmark Mode." << std::endl;
        RunHeadlessBenchmark();
        return 0;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cout << "GLEW Initialization failed, falling back to Headless Benchmark Mode." << std::endl;
        RunHeadlessBenchmark();
        return 0;
    }

    GLuint pbo_id, texture_id;
    size_t frame_bytes = WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(uchar4);

    glGenBuffers(1, &pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, frame_bytes, NULL, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cuda_pbo_resource = nullptr;
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_id, cudaGraphicsRegisterFlagsWriteDiscard);
#endif

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    int3 grid_dim = make_int3(GRID_DIM, GRID_DIM, GRID_DIM);
    size_t grid_bytes = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);

    float *d_p_curr = nullptr, *d_p_next = nullptr;
    cudaMalloc((void**)&d_p_curr, grid_bytes);
    cudaMalloc((void**)&d_p_next, grid_bytes);
    cudaMemset(d_p_curr, 0, grid_bytes);
    cudaMemset(d_p_next, 0, grid_bytes);

    dim3 grid_3d_block(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 grid_3d_grid(grid_dim.x / TILE_DIM, grid_dim.y / TILE_DIM, grid_dim.z / TILE_DIM);

    dim3 render_block(16, 16);
    dim3 render_grid((WINDOW_WIDTH + 15) / 16, (WINDOW_HEIGHT + 15) / 16);

    LAUNCH_KERNEL(Inject_Kronecker_Jamo_Seed_Kernel, grid_3d_grid, grid_3d_block, d_p_curr, grid_dim);

    float current_time = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        current_time += 0.033f;

        float cam_x = g_camera.dist * cosf(g_camera.pitch) * sinf(g_camera.yaw);
        float cam_y = g_camera.dist * sinf(g_camera.pitch);
        float cam_z = g_camera.dist * cosf(g_camera.pitch) * cosf(g_camera.yaw);

        float3 cam_pos = make_float3(cam_x, cam_y, cam_z);
        float3 target  = make_float3(0.0f, 0.0f, 0.0f);
        float3 world_up = make_float3(0.0f, 1.0f, 0.0f);

        float3 view_w = normalize_vec(make_float3(target.x - cam_pos.x, target.y - cam_pos.y, target.z - cam_pos.z));
        float3 view_u = normalize_vec(cross_vec(view_w, world_up));
        float3 view_v = cross_vec(view_u, view_w);

        LAUNCH_KERNEL(Inject_3Phase_Delta_Sources_Kernel, dim3(1,1,1), dim3(3,1,1), d_p_curr, grid_dim, current_time);

        LAUNCH_KERNEL(FDTD_Wave_Step_Kernel, grid_3d_grid, grid_3d_block, d_p_curr, d_p_next, grid_dim);
        std::swap(d_p_curr, d_p_next);

        uchar4* d_pbo_frame_ptr = nullptr;
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        size_t mapped_size = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_frame_ptr, &mapped_size, cuda_pbo_resource);
#endif

        LAUNCH_KERNEL(CUDA_Unified_Raymarch_Kernel, render_grid, render_block,
            d_p_curr, d_pbo_frame_ptr, make_int2(WINDOW_WIDTH, WINDOW_HEIGHT), grid_dim, current_time,
            cam_pos, view_u, view_v, view_w
        );

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
#endif

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();

        glfwSwapBuffers(window);
    }

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
#endif
    glDeleteBuffers(1, &pbo_id);
    glDeleteTextures(1, &texture_id);
    cudaFree(d_p_curr);
    cudaFree(d_p_next);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
#else
    RunHeadlessBenchmark();
    return 0;
#endif
}
