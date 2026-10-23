#include "causal_engine/3dgs_sh_eval.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace elysia {
namespace gaussian3d {

void computeColorFromSH(
    int P,
    int deg,
    int max_coeffs,
    const Float3* means3D,
    Float3 cam_pos,
    const float* shs,
    Float3* rgb_out
) {
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < P; ++idx) {
        Float3 p_orig = means3D[idx];
        Float3 dir_orig(p_orig.x - cam_pos.x, p_orig.y - cam_pos.y, p_orig.z - cam_pos.z);
        float dist = std::sqrt(dir_orig.x * dir_orig.x + dir_orig.y * dir_orig.y + dir_orig.z * dir_orig.z);

        Float3 dir(0.0f, 0.0f, 1.0f);
        if (dist > 0.0f) {
            dir.x = dir_orig.x / dist;
            dir.y = dir_orig.y / dist;
            dir.z = dir_orig.z / dist;
        }

        const float* sh_ptr = shs + idx * max_coeffs * 3;
        rgb_out[idx] = computeSH(deg, dir, sh_ptr);
    }
}

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
) {
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < P; ++idx) {
        Float3 p_orig = means3D[idx];
        Float3 dir_orig(p_orig.x - cam_pos.x, p_orig.y - cam_pos.y, p_orig.z - cam_pos.z);
        float dist = std::sqrt(dir_orig.x * dir_orig.x + dir_orig.y * dir_orig.y + dir_orig.z * dir_orig.z);

        if (dist == 0.0f) {
            dL_dmeans3D[idx] = Float3(0.0f, 0.0f, 0.0f);
            continue;
        }

        Float3 dir(dir_orig.x / dist, dir_orig.y / dist, dir_orig.z / dist);

        Float3 color = rgb[idx];
        Float3 dL_dcolor = dL_drgb[idx];

        if (color.x <= 0.0f) dL_dcolor.x = 0.0f;
        if (color.y <= 0.0f) dL_dcolor.y = 0.0f;
        if (color.z <= 0.0f) dL_dcolor.z = 0.0f;

        const float* sh = shs + idx * max_coeffs * 3;
        float* dL_dsh_ptr = dL_dsh + idx * max_coeffs * 3;

        // Degree 0 (DC)
        dL_dsh_ptr[0] = SH_C0 * dL_dcolor.x;
        dL_dsh_ptr[1] = SH_C0 * dL_dcolor.y;
        dL_dsh_ptr[2] = SH_C0 * dL_dcolor.z;

        Float3 dL_ddir(0.0f, 0.0f, 0.0f);

        if (deg > 0) {
            float x = dir.x, y = dir.y, z = dir.z;

            // Degree 1
            float y1_0 = -SH_C1 * y;
            float y1_1 =  SH_C1 * z;
            float y1_2 = -SH_C1 * x;

            dL_dsh_ptr[3]  = y1_0 * dL_dcolor.x; dL_dsh_ptr[4]  = y1_0 * dL_dcolor.y; dL_dsh_ptr[5]  = y1_0 * dL_dcolor.z;
            dL_dsh_ptr[6]  = y1_1 * dL_dcolor.x; dL_dsh_ptr[7]  = y1_1 * dL_dcolor.y; dL_dsh_ptr[8]  = y1_1 * dL_dcolor.z;
            dL_dsh_ptr[9]  = y1_2 * dL_dcolor.x; dL_dsh_ptr[10] = y1_2 * dL_dcolor.y; dL_dsh_ptr[11] = y1_2 * dL_dcolor.z;

            float dL_dy1_0 = dL_dcolor.x * sh[3] + dL_dcolor.y * sh[4] + dL_dcolor.z * sh[5];
            float dL_dy1_1 = dL_dcolor.x * sh[6] + dL_dcolor.y * sh[7] + dL_dcolor.z * sh[8];
            float dL_dy1_2 = dL_dcolor.x * sh[9] + dL_dcolor.y * sh[10] + dL_dcolor.z * sh[11];

            dL_ddir.x += -SH_C1 * dL_dy1_2;
            dL_ddir.y += -SH_C1 * dL_dy1_0;
            dL_ddir.z +=  SH_C1 * dL_dy1_1;

            if (deg > 1) {
                // Degree 2
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;

                float y2_0 = SH_C2_0 * xy;
                float y2_1 = SH_C2_1 * yz;
                float y2_2 = SH_C2_2 * (2.0f * zz - xx - yy);
                float y2_3 = SH_C2_3 * xz;
                float y2_4 = SH_C2_4 * (xx - yy);

                dL_dsh_ptr[12] = y2_0 * dL_dcolor.x; dL_dsh_ptr[13] = y2_0 * dL_dcolor.y; dL_dsh_ptr[14] = y2_0 * dL_dcolor.z;
                dL_dsh_ptr[15] = y2_1 * dL_dcolor.x; dL_dsh_ptr[16] = y2_1 * dL_dcolor.y; dL_dsh_ptr[17] = y2_1 * dL_dcolor.z;
                dL_dsh_ptr[18] = y2_2 * dL_dcolor.x; dL_dsh_ptr[19] = y2_2 * dL_dcolor.y; dL_dsh_ptr[20] = y2_2 * dL_dcolor.z;
                dL_dsh_ptr[21] = y2_3 * dL_dcolor.x; dL_dsh_ptr[22] = y2_3 * dL_dcolor.y; dL_dsh_ptr[23] = y2_3 * dL_dcolor.z;
                dL_dsh_ptr[24] = y2_4 * dL_dcolor.x; dL_dsh_ptr[25] = y2_4 * dL_dcolor.y; dL_dsh_ptr[26] = y2_4 * dL_dcolor.z;

                float dL_dy2_0 = dL_dcolor.x * sh[12] + dL_dcolor.y * sh[13] + dL_dcolor.z * sh[14];
                float dL_dy2_1 = dL_dcolor.x * sh[15] + dL_dcolor.y * sh[16] + dL_dcolor.z * sh[17];
                float dL_dy2_2 = dL_dcolor.x * sh[18] + dL_dcolor.y * sh[19] + dL_dcolor.z * sh[20];
                float dL_dy2_3 = dL_dcolor.x * sh[21] + dL_dcolor.y * sh[22] + dL_dcolor.z * sh[23];
                float dL_dy2_4 = dL_dcolor.x * sh[24] + dL_dcolor.y * sh[25] + dL_dcolor.z * sh[26];

                dL_ddir.x += SH_C2_0 * y * dL_dy2_0 + SH_C2_2 * (-2.0f * x) * dL_dy2_2 + SH_C2_3 * z * dL_dy2_3 + SH_C2_4 * (2.0f * x) * dL_dy2_4;
                dL_ddir.y += SH_C2_0 * x * dL_dy2_0 + SH_C2_1 * z * dL_dy2_1 + SH_C2_2 * (-2.0f * y) * dL_dy2_2 + SH_C2_4 * (-2.0f * y) * dL_dy2_4;
                dL_ddir.z += SH_C2_1 * y * dL_dy2_1 + SH_C2_2 * (4.0f * z) * dL_dy2_2 + SH_C2_3 * x * dL_dy2_3;

                if (deg > 2) {
                    // Degree 3
                    float y3_0 = SH_C3_0 * y * (3.0f * xx - yy);
                    float y3_1 = SH_C3_1 * xy * z;
                    float y3_2 = SH_C3_2 * y * (5.0f * zz - 1.0f);
                    float y3_3 = SH_C3_3 * z * (5.0f * zz - 3.0f);
                    float y3_4 = SH_C3_4 * x * (5.0f * zz - 1.0f);
                    float y3_5 = SH_C3_5 * z * (xx - yy);
                    float y3_6 = SH_C3_6 * x * (xx - 3.0f * yy);

                    dL_dsh_ptr[27] = y3_0 * dL_dcolor.x; dL_dsh_ptr[28] = y3_0 * dL_dcolor.y; dL_dsh_ptr[29] = y3_0 * dL_dcolor.z;
                    dL_dsh_ptr[30] = y3_1 * dL_dcolor.x; dL_dsh_ptr[31] = y3_1 * dL_dcolor.y; dL_dsh_ptr[32] = y3_1 * dL_dcolor.z;
                    dL_dsh_ptr[33] = y3_2 * dL_dcolor.x; dL_dsh_ptr[34] = y3_2 * dL_dcolor.y; dL_dsh_ptr[35] = y3_2 * dL_dcolor.z;
                    dL_dsh_ptr[36] = y3_3 * dL_dcolor.x; dL_dsh_ptr[37] = y3_3 * dL_dcolor.y; dL_dsh_ptr[38] = y3_3 * dL_dcolor.z;
                    dL_dsh_ptr[39] = y3_4 * dL_dcolor.x; dL_dsh_ptr[40] = y3_4 * dL_dcolor.y; dL_dsh_ptr[41] = y3_4 * dL_dcolor.z;
                    dL_dsh_ptr[42] = y3_5 * dL_dcolor.x; dL_dsh_ptr[43] = y3_5 * dL_dcolor.y; dL_dsh_ptr[44] = y3_5 * dL_dcolor.z;
                    dL_dsh_ptr[45] = y3_6 * dL_dcolor.x; dL_dsh_ptr[46] = y3_6 * dL_dcolor.y; dL_dsh_ptr[47] = y3_6 * dL_dcolor.z;

                    float dL_dy3_0 = dL_dcolor.x * sh[27] + dL_dcolor.y * sh[28] + dL_dcolor.z * sh[29];
                    float dL_dy3_1 = dL_dcolor.x * sh[30] + dL_dcolor.y * sh[31] + dL_dcolor.z * sh[32];
                    float dL_dy3_2 = dL_dcolor.x * sh[33] + dL_dcolor.y * sh[34] + dL_dcolor.z * sh[35];
                    float dL_dy3_3 = dL_dcolor.x * sh[36] + dL_dcolor.y * sh[37] + dL_dcolor.z * sh[38];
                    float dL_dy3_4 = dL_dcolor.x * sh[39] + dL_dcolor.y * sh[40] + dL_dcolor.z * sh[41];
                    float dL_dy3_5 = dL_dcolor.x * sh[42] + dL_dcolor.y * sh[43] + dL_dcolor.z * sh[44];
                    float dL_dy3_6 = dL_dcolor.x * sh[45] + dL_dcolor.y * sh[46] + dL_dcolor.z * sh[47];

                    dL_ddir.x += SH_C3_0 * (6.0f * x * y) * dL_dy3_0 + SH_C3_1 * (y * z) * dL_dy3_1 + SH_C3_4 * (5.0f * zz - 1.0f) * dL_dy3_4 + SH_C3_5 * (2.0f * x * z) * dL_dy3_5 + SH_C3_6 * (3.0f * xx - 3.0f * yy) * dL_dy3_6;
                    dL_ddir.y += SH_C3_0 * (3.0f * xx - 3.0f * yy) * dL_dy3_0 + SH_C3_1 * (x * z) * dL_dy3_1 + SH_C3_2 * (5.0f * zz - 1.0f) * dL_dy3_2 + SH_C3_5 * (-2.0f * y * z) * dL_dy3_5 + SH_C3_6 * (-6.0f * x * y) * dL_dy3_6;
                    dL_ddir.z += SH_C3_1 * (x * y) * dL_dy3_1 + SH_C3_2 * (10.0f * y * z) * dL_dy3_2 + SH_C3_3 * (15.0f * zz - 3.0f) * dL_dy3_3 + SH_C3_4 * (10.0f * x * z) * dL_dy3_4 + SH_C3_5 * (xx - yy) * dL_dy3_5;
                }
            }
        }

        // Projection Jacobian Chain Rule
        float dL_dot_dir = dL_ddir.x * dir.x + dL_ddir.y * dir.y + dL_ddir.z * dir.z;
        dL_dmeans3D[idx] = Float3(
            (dL_ddir.x - dL_dot_dir * dir.x) / dist,
            (dL_ddir.y - dL_dot_dir * dir.y) / dist,
            (dL_ddir.z - dL_dot_dir * dir.z) / dist
        );
    }
}

void computePhys3DGSCollisionGradient(
    int P,
    const Float3* means3D,
    const float* inv_cov3D,
    const float* opacity,
    CapsuleLink link,
    float* dC_dq
) {
    int dof = link.dof;
    std::fill(dC_dq, dC_dq + dof, 0.0f);

    Float3 ab(link.pB.x - link.pA.x, link.pB.y - link.pA.y, link.pB.z - link.pA.z);
    float ab_sq = ab.x * ab.x + ab.y * ab.y + ab.z * ab.z;

    std::vector<float> local_dC_dq(dof, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> thread_dC_dq(dof, 0.0f);

        #pragma omp for nowait
        for (int idx = 0; idx < P; ++idx) {
            Float3 mu = means3D[idx];
            float o = opacity[idx];

            // Projection onto capsule segment pA -> pB
            float t = 0.5f;
            if (ab_sq > 1e-8f) {
                Float3 a_mu(mu.x - link.pA.x, mu.y - link.pA.y, mu.z - link.pA.z);
                t = (a_mu.x * ab.x + a_mu.y * ab.y + a_mu.z * ab.z) / ab_sq;
                t = std::max(0.0f, std::min(1.0f, t));
            }

            Float3 x_star(
                link.pA.x + t * ab.x,
                link.pA.y + t * ab.y,
                link.pA.z + t * ab.z
            );

            Float3 diff(x_star.x - mu.x, x_star.y - mu.y, x_star.z - mu.z);

            // inv_cov3D symmetric matrix: xx, xy, xz, yy, yz, zz
            const float* ic = inv_cov3D + idx * 6;
            float mahalanobis_sq =
                diff.x * (ic[0] * diff.x + ic[1] * diff.y + ic[2] * diff.z) +
                diff.y * (ic[1] * diff.x + ic[3] * diff.y + ic[4] * diff.z) +
                diff.z * (ic[2] * diff.x + ic[4] * diff.y + ic[5] * diff.z);

            float rho = o * std::exp(-0.5f * mahalanobis_sq);

            // Gradient wrt x_star: dC / dx_star = -rho * inv_cov3D * diff
            Float3 dC_dx(
                -rho * (ic[0] * diff.x + ic[1] * diff.y + ic[2] * diff.z),
                -rho * (ic[1] * diff.x + ic[3] * diff.y + ic[4] * diff.z),
                -rho * (ic[2] * diff.x + ic[4] * diff.y + ic[5] * diff.z)
            );

            // Interpolated Jacobian J_x* = (1-t) J_A + t J_B
            for (int k = 0; k < dof; ++k) {
                float jx = (1.0f - t) * link.jacobianA[0 * dof + k] + t * link.jacobianB[0 * dof + k];
                float jy = (1.0f - t) * link.jacobianA[1 * dof + k] + t * link.jacobianB[1 * dof + k];
                float jz = (1.0f - t) * link.jacobianA[2 * dof + k] + t * link.jacobianB[2 * dof + k];

                thread_dC_dq[k] += dC_dx.x * jx + dC_dx.y * jy + dC_dx.z * jz;
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < dof; ++k) {
                dC_dq[k] += thread_dC_dq[k];
            }
        }
    }
}

} // namespace gaussian3d
} // namespace elysia
