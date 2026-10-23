#ifndef ELYSIA_CL31_STA_HPP
#define ELYSIA_CL31_STA_HPP

#include <vector>
#include <cmath>
#include <iostream>

namespace elysia {

// 16-component multivector in Space-Time Algebra Cl(3,1)
struct alignas(64) Multivector16 {
    float s{0.0f};           // 0D: Scalar (1)
    float v[4]{0,0,0,0};     // 1D: Vector (gamma_0, gamma_1, gamma_2, gamma_3)
    float b[6]{0,0,0,0,0,0}; // 2D: Bivector (e01, e02, e03, e12, e23, e31)
    float pv[4]{0,0,0,0};    // 3D: Pseudovector
    float ps{0.0f};          // 4D: Pseudoscalar (gamma_0123)
};

// Structure of Arrays (SoA) layout optimized for CXL / cache lines
struct Multivector16_SoA {
    std::vector<float> s;
    std::vector<float> v0, v1, v2, v3;
    std::vector<float> b0, b1, b2, b3, b4, b5;
    std::vector<float> pv0, pv1, pv2, pv3;
    std::vector<float> ps;

    size_t size() const { return s.size(); }

    void resize(size_t n) {
        s.resize(n, 0.0f);
        v0.resize(n, 0.0f); v1.resize(n, 0.0f); v2.resize(n, 0.0f); v3.resize(n, 0.0f);
        b0.resize(n, 0.0f); b1.resize(n, 0.0f); b2.resize(n, 0.0f);
        b3.resize(n, 0.0f); b4.resize(n, 0.0f); b5.resize(n, 0.0f);
        pv0.resize(n, 0.0f); pv1.resize(n, 0.0f); pv2.resize(n, 0.0f); pv3.resize(n, 0.0f);
        ps.resize(n, 0.0f);
    }
};

// CPU reference launcher for Cl(3,1) SoA Weave
inline void cl31_weave_soa_cpu(
    const float* a_s, const float* a_v0, const float* a_v1, const float* a_v2, const float* a_v3,
    const float* b_s, const float* b_v0, const float* b_v1, const float* b_v2, const float* b_v3,
    float* out_s, float* out_b0, float* out_b1, float* out_b2,
    float* out_b3, float* out_b4, float* out_b5,
    size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        float as = a_s[i], av0 = a_v0[i], av1 = a_v1[i], av2 = a_v2[i], av3 = a_v3[i];
        float bs = b_s[i], bv0 = b_v0[i], bv1 = b_v1[i], bv2 = b_v2[i], bv3 = b_v3[i];

        // Minkowski metric signature (+,-,-,-) scalar contraction:
        out_s[i] = as * bs + av0 * bv0 - av1 * bv1 - av2 * bv2 - av3 * bv3;

        // Wedge product: v_A ^ v_B -> 2D Bivector sheet
        // Time-space bivectors (e01, e02, e03)
        out_b0[i] = av0 * bv1 - av1 * bv0;
        out_b1[i] = av0 * bv2 - av2 * bv0;
        out_b2[i] = av0 * bv3 - av3 * bv0;

        // Space-space bivectors (e12, e23, e31)
        out_b3[i] = av1 * bv2 - av2 * bv1;
        out_b4[i] = av2 * bv3 - av3 * bv2;
        out_b5[i] = av3 * bv1 - av1 * bv3;
    }
}

} // namespace elysia

#endif // ELYSIA_CL31_STA_HPP
