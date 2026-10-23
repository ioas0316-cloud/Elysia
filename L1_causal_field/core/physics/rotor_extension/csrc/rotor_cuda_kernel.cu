#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level parallel reduction helper
template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ------------------------------------------------------------------
// CUDA Kernel: Forward Pass (Batch-level & Grid Stride Loop)
// ------------------------------------------------------------------
template <typename scalar_t>
__global__ void rotor_sandwich_cuda_kernel(
    const scalar_t* __restrict__ v,     // [B, D]
    const scalar_t* __restrict__ u,     // [B, D]
    const scalar_t* __restrict__ w,     // [B, D]
    const scalar_t* __restrict__ theta, // [B]
    scalar_t* __restrict__ out,         // [B, D]
    int B, int D)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    const scalar_t* v_b = v + b * D;
    const scalar_t* u_b = u + b * D;
    const scalar_t* w_b = w + b * D;
    scalar_t* out_b = out + b * D;

    scalar_t local_dot_u = 0;
    scalar_t local_dot_w = 0;

    // 1-Pass: Grid Stride Loop to compute local dot products
    for (int i = tid; i < D; i += block_dim) {
        scalar_t v_val = v_b[i];
        local_dot_u += v_val * u_b[i];
        local_dot_w += v_val * w_b[i];
    }

    // Warp Reduction
    local_dot_u = warpReduceSum(local_dot_u);
    local_dot_w = warpReduceSum(local_dot_w);

    // Shared Memory to collect warp sums
    __shared__ scalar_t s_dot_u[32];
    __shared__ scalar_t s_dot_w[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        s_dot_u[warp_id] = local_dot_u;
        s_dot_w[warp_id] = local_dot_w;
    }
    __syncthreads();

    // Final reduction by thread 0
    scalar_t v_dot_u = 0;
    scalar_t v_dot_w = 0;
    int num_warps = (block_dim + 31) / 32;

    if (tid < 32) {
        scalar_t val_u = (tid < num_warps) ? s_dot_u[tid] : scalar_t(0);
        scalar_t val_w = (tid < num_warps) ? s_dot_w[tid] : scalar_t(0);
        v_dot_u = warpReduceSum(val_u);
        v_dot_w = warpReduceSum(val_w);
    }

    __shared__ scalar_t final_dot_u;
    __shared__ scalar_t final_dot_w;
    if (tid == 0) {
        final_dot_u = v_dot_u;
        final_dot_w = v_dot_w;
    }
    __syncthreads();

    // Scalar trigonometric calculations
    __shared__ scalar_t sin_t, one_minus_cos_t;
    if (tid == 0) {
        scalar_t th = theta[b];
        sin_t = sin(th);
        one_minus_cos_t = scalar_t(1.0) - cos(th);
    }
    __syncthreads();

    // 2-Pass: Apply rotation and write to global memory
    scalar_t dot_u = final_dot_u;
    scalar_t dot_w = final_dot_w;

    for (int i = tid; i < D; i += block_dim) {
        scalar_t v_val = v_b[i];
        scalar_t u_val = u_b[i];
        scalar_t w_val = w_b[i];

        // Av = w * (u . v) - u * (w . v)
        scalar_t Av = w_val * dot_u - u_val * dot_w;

        // A2v = -(u * (u . v) + w * (w . v))
        scalar_t A2v = -(u_val * dot_u + w_val * dot_w);

        // v' = v + sin(θ) * Av + (1 - cos(θ)) * A2v
        out_b[i] = v_val + sin_t * Av + one_minus_cos_t * A2v;
    }
}

// ------------------------------------------------------------------
// CUDA Kernel: Backward Pass (Simultaneous Gradient Calculation)
// ------------------------------------------------------------------
template <typename scalar_t>
__global__ void rotor_sandwich_cuda_backward_kernel(
    const scalar_t* __restrict__ g,     // [B, D] - Upstream Gradient
    const scalar_t* __restrict__ v,     // [B, D]
    const scalar_t* __restrict__ u,     // [B, D]
    const scalar_t* __restrict__ w,     // [B, D]
    const scalar_t* __restrict__ theta, // [B]
    scalar_t* __restrict__ grad_v,      // [B, D]
    scalar_t* __restrict__ grad_u,      // [B, D]
    scalar_t* __restrict__ grad_w,      // [B, D]
    scalar_t* __restrict__ grad_theta,  // [B]
    int B, int D)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    const scalar_t* g_b = g + b * D;
    const scalar_t* v_b = v + b * D;
    const scalar_t* u_b = u + b * D;
    const scalar_t* w_b = w + b * D;

    // 1-Pass: compute 4 local dot products simultaneously
    scalar_t l_alpha = 0, l_beta = 0, l_gamma = 0, l_delta = 0;

    for (int i = tid; i < D; i += block_dim) {
        scalar_t g_val = g_b[i];
        scalar_t v_val = v_b[i];
        scalar_t u_val = u_b[i];
        scalar_t w_val = w_b[i];

        l_alpha += u_val * v_val; // alpha = u . v
        l_beta  += w_val * v_val; // beta  = w . v
        l_gamma += u_val * g_val; // gamma = u . g
        l_delta += w_val * g_val; // delta = w . g
    }

    // Warp Reduction
    l_alpha = warpReduceSum(l_alpha);
    l_beta  = warpReduceSum(l_beta);
    l_gamma = warpReduceSum(l_gamma);
    l_delta = warpReduceSum(l_delta);

    __shared__ scalar_t s_alpha[32], s_beta[32], s_gamma[32], s_delta[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        s_alpha[warp_id] = l_alpha;
        s_beta[warp_id]  = l_beta;
        s_gamma[warp_id] = l_gamma;
        s_delta[warp_id] = l_delta;
    }
    __syncthreads();

    // Final reduction and scalar gradient coefficient calculation by thread 0
    __shared__ scalar_t C_u, C_w, D_u, D_w, d_th;
    if (tid == 0) {
        scalar_t a = 0, be = 0, ga = 0, de = 0;
        int num_warps = (block_dim + 31) / 32;
        for (int i = 0; i < num_warps; ++i) {
            a  += s_alpha[i]; be += s_beta[i];
            ga += s_gamma[i]; de += s_delta[i];
        }

        scalar_t th = theta[b];
        scalar_t s = sin(th);
        scalar_t c = scalar_t(1.0) - cos(th);
        scalar_t cos_t = cos(th);

        C_u = -be * s - a * c;
        C_w = a * s - be * c;
        D_u = -ga * c + de * s;
        D_w = -ga * s - de * c;

        // d_theta coefficient
        d_th = ga * (-be * cos_t - a * s) + de * (a * cos_t - be * s);
        grad_theta[b] = d_th;
    }
    __syncthreads();

    // 2-Pass: calculate and save gradient values for each input vector elements
    scalar_t cu = C_u, cw = C_w, du = D_u, dw = D_w;
    scalar_t* g_v_b = grad_v + b * D;
    scalar_t* g_u_b = grad_u + b * D;
    scalar_t* g_w_b = grad_w + b * D;

    for (int i = tid; i < D; i += block_dim) {
        scalar_t g_val = g_b[i];
        scalar_t v_val = v_b[i];
        scalar_t u_val = u_b[i];
        scalar_t w_val = w_b[i];

        g_v_b[i] = g_val + du * u_val + dw * w_val; // grad_v
        g_u_b[i] = cu * g_val + du * v_val;         // grad_u
        g_w_b[i] = cw * g_val + dw * v_val;         // grad_w
    }
}

// ------------------------------------------------------------------
// C++ Wrapper / CUDA Launcher interfaces
// ------------------------------------------------------------------
torch::Tensor rotor_sandwich_cuda_forward(
    torch::Tensor v, torch::Tensor u, torch::Tensor w, torch::Tensor theta)
{
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(u.is_contiguous(), "u must be contiguous");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(theta.is_cuda(), "theta must be a CUDA tensor");
    TORCH_CHECK(theta.is_contiguous(), "theta must be contiguous");

    int B = v.size(0);
    int D = v.size(1);

    auto out = torch::empty_like(v);

    int threads = 256;
    dim3 blocks(B);

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "rotor_sandwich_cuda_forward", ([&] {
        rotor_sandwich_cuda_kernel<scalar_t><<<blocks, threads>>>(
            v.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, D);
    }));

    return out;
}

std::vector<torch::Tensor> rotor_sandwich_cuda_backward(
    torch::Tensor g, torch::Tensor v, torch::Tensor u, torch::Tensor w, torch::Tensor theta)
{
    TORCH_CHECK(g.is_cuda(), "g must be a CUDA tensor");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(u.is_contiguous(), "u must be contiguous");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(theta.is_cuda(), "theta must be a CUDA tensor");
    TORCH_CHECK(theta.is_contiguous(), "theta must be contiguous");

    int B = v.size(0);
    int D = v.size(1);

    auto grad_v = torch::empty_like(v);
    auto grad_u = torch::empty_like(u);
    auto grad_w = torch::empty_like(w);
    auto grad_theta = torch::empty_like(theta);

    int threads = 256;
    dim3 blocks(B);

    AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "rotor_sandwich_cuda_backward", ([&] {
        rotor_sandwich_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            g.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            theta.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(),
            grad_u.data_ptr<scalar_t>(),
            grad_w.data_ptr<scalar_t>(),
            grad_theta.data_ptr<scalar_t>(),
            B, D);
    }));

    return {grad_v, grad_u, grad_w, grad_theta};
}
