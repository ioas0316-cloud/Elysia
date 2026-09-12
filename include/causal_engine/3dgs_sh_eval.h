#ifndef ELYSIA_3DGS_SH_EVAL_H
#define ELYSIA_3DGS_SH_EVAL_H

#include <cmath>
#include <vector>
#include <algorithm>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#define DEVICE_INLINE __device__ __forceinline__
#else
#define HOST_DEVICE
#define DEVICE_INLINE inline
#endif

namespace elysia {
namespace gaussian3d {

struct Float3 {
    float x, y, z;
    HOST_DEVICE Float3() : x(0.0f), y(0.0f), z(0.0f) {}
    HOST_DEVICE Float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Pre-computed Spherical Harmonics Constants
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

HOST_DEVICE inline float clamp_zero(float val) {
#if defined(__CUDA_ARCH__)
    return fmaxf(0.0f, val);
#else
    return std::max(0.0f, val);
#endif
}

// Evaluates SH basis functions for a single direction and SH parameters
HOST_DEVICE inline Float3 computeSH(int deg, Float3 dir, const float* sh) {
    Float3 result(
        0.5f + SH_C0 * sh[0],
        0.5f + SH_C0 * sh[1],
        0.5f + SH_C0 * sh[2]
    );

    if (deg < 1) {
        return Float3(
            clamp_zero(result.x),
            clamp_zero(result.y),
            clamp_zero(result.z)
        );
    }

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    // Degree 1
    float y1_0 = -SH_C1 * y;
    float y1_1 =  SH_C1 * z;
    float y1_2 = -SH_C1 * x;

    result.x += y1_0 * sh[3]  + y1_1 * sh[6]  + y1_2 * sh[9];
    result.y += y1_0 * sh[4]  + y1_1 * sh[7]  + y1_2 * sh[10];
    result.z += y1_0 * sh[5]  + y1_1 * sh[8]  + y1_2 * sh[11];

    if (deg < 2) {
        return Float3(
            clamp_zero(result.x),
            clamp_zero(result.y),
            clamp_zero(result.z)
        );
    }

    // Degree 2
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    float y2_0 = SH_C2_0 * xy;
    float y2_1 = SH_C2_1 * yz;
    float y2_2 = SH_C2_2 * (2.0f * zz - xx - yy);
    float y2_3 = SH_C2_3 * xz;
    float y2_4 = SH_C2_4 * (xx - yy);

    result.x += y2_0 * sh[12] + y2_1 * sh[15] + y2_2 * sh[18] + y2_3 * sh[21] + y2_4 * sh[24];
    result.y += y2_0 * sh[13] + y2_1 * sh[16] + y2_2 * sh[19] + y2_3 * sh[22] + y2_4 * sh[25];
    result.z += y2_0 * sh[14] + y2_1 * sh[17] + y2_2 * sh[20] + y2_3 * sh[23] + y2_4 * sh[26];

    if (deg < 3) {
        return Float3(
            clamp_zero(result.x),
            clamp_zero(result.y),
            clamp_zero(result.z)
        );
    }

    // Degree 3
    float y3_0 = SH_C3_0 * y * (3.0f * xx - yy);
    float y3_1 = SH_C3_1 * xy * z;
    float y3_2 = SH_C3_2 * y * (5.0f * zz - 1.0f);
    float y3_3 = SH_C3_3 * z * (5.0f * zz - 3.0f);
    float y3_4 = SH_C3_4 * x * (5.0f * zz - 1.0f);
    float y3_5 = SH_C3_5 * z * (xx - yy);
    float y3_6 = SH_C3_6 * x * (xx - 3.0f * yy);

    result.x += y3_0 * sh[27] + y3_1 * sh[30] + y3_2 * sh[33] + y3_3 * sh[36] + y3_4 * sh[39] + y3_5 * sh[42] + y3_6 * sh[45];
    result.y += y3_0 * sh[28] + y3_1 * sh[31] + y3_2 * sh[34] + y3_3 * sh[37] + y3_4 * sh[40] + y3_5 * sh[43] + y3_6 * sh[46];
    result.z += y3_0 * sh[29] + y3_1 * sh[32] + y3_2 * sh[35] + y3_3 * sh[38] + y3_4 * sh[41] + y3_5 * sh[44] + y3_6 * sh[47];

    return Float3(
        clamp_zero(result.x),
        clamp_zero(result.y),
        clamp_zero(result.z)
    );
}

// Host evaluation forward entry point
void computeColorFromSH(
    int P,
    int deg,
    int max_coeffs,
    const Float3* means3D,
    Float3 cam_pos,
    const float* shs,
    Float3* rgb_out
);

// Host evaluation backward entry point
void computeColorFromSHBackward(
    int P,
    int deg,
    int max_coeffs,
    const Float3* means3D,
    Float3 cam_pos,
    const float* shs,
    const Float3* rgb,
    const Float3* dL_drgb,
    float* dL_dsh,
    Float3* dL_dmeans3D
);

#if defined(__CUDACC__)
void launchComputeColorFromSHCUDA(
    int P, int deg, int max_coeffs,
    const Float3* means3D, Float3 cam_pos, const float* shs, Float3* rgb_out
);

void launchComputeColorFromSHBackwardCUDA(
    int P, int deg, int max_coeffs,
    const Float3* means3D, Float3 cam_pos, const float* shs,
    const Float3* rgb, const Float3* dL_drgb, float* dL_dsh, Float3* dL_dmeans3D
);
#endif

// Phys-3DGS Capsule Link vs Ellipsoid Gaussians Analytic Collision Gradient
struct CapsuleLink {
    Float3 pA;
    Float3 pB;
    int dof;
    const float* jacobianA;
    const float* jacobianB;
};

void computePhys3DGSCollisionGradient(
    int P,
    const Float3* means3D,
    const float* inv_cov3D,
    const float* opacity,
    CapsuleLink link,
    float* dC_dq
);

} // namespace gaussian3d
} // namespace elysia

#endif // ELYSIA_3DGS_SH_EVAL_H
