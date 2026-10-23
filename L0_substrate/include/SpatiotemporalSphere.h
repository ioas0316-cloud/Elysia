#ifndef SPATIOTEMPORAL_SPHERE_H
#define SPATIOTEMPORAL_SPHERE_H

#if defined(__cplusplus)
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#define ALIGN64 alignas(64)
#else
#define ALIGN64
#endif

// 64-Byte Cache Line Aligned Spatiotemporal Sphere (Hyperspherical Bubble)
// Exactly 64 Bytes: Matches GPU L1/L2 Cache Line & Vector Register Alignment
struct ALIGN64 SpatiotemporalSphere {
    // Offset 0x00 [16 Bytes]: World Center (x, y, z) & Monopole Base Scale R_0 (l=0)
    float centerPos[3];
    float baseRadius;

    // Offset 0x10 [16 Bytes]: Physical Kinematics & Topological Surface Tension T
    float velocity[3];
    float surfaceTension;

    // Offset 0x20 [16 Bytes]: Dipole (l=1) & Low-Order Harmonic Coefficients (c_1^-1, c_1^0, c_1^1, c_2^0)
    float shCoeffs_Low[4];

    // Offset 0x30 [16 Bytes]: High-Order Deformation Waves (c_2^2, c_3^0, c_3^3) & Bit Flags
    float shCoeffs_High[3];
    uint32_t flags; // Bit 0: IsInfiltrated, Bit 1: IsActive, Bit 2: HasChildren
};

#if defined(__cplusplus)

constexpr float PI_F = 3.14159265358979323846f;

// Host-side evaluation of dynamic sphere radius R(theta, phi) using Spherical Harmonics up to l=2
inline float EvaluateSphereRadiusHost(const SpatiotemporalSphere& sphere, float theta, float phi) {
    float sinT = std::sin(theta);
    float cosT = std::cos(theta);
    float cosP = std::cos(phi);
    float sinP = std::sin(phi);

    // l=0 Monopole Base Radius
    float R = sphere.baseRadius;

    // l=1 Dipole Harmonics
    float k_Y1 = 0.5f * std::sqrt(3.0f / PI_F);
    float Y_1_m1 = k_Y1 * sinT * sinP;
    float Y_1_0  = k_Y1 * cosT;
    float Y_1_p1 = k_Y1 * sinT * cosP;

    R += sphere.shCoeffs_Low[0] * Y_1_m1 +
         sphere.shCoeffs_Low[1] * Y_1_0  +
         sphere.shCoeffs_Low[2] * Y_1_p1;

    // l=2 Quadrupole Harmonic
    float k_Y2 = 0.25f * std::sqrt(5.0f / PI_F);
    float Y_2_0 = k_Y2 * (3.0f * cosT * cosT - 1.0f);

    R += sphere.shCoeffs_Low[3] * Y_2_0;

    return std::max(R, 1e-6f);
}

// Host-side direction-vector evaluation of dynamic sphere radius (trig-free)
inline float EvaluateSphereRadiusDirectionHost(const SpatiotemporalSphere& sphere, const float n[3]) {
    float R = sphere.baseRadius;

    float k_Y1 = 0.5f * std::sqrt(3.0f / PI_F);
    R += sphere.shCoeffs_Low[0] * (k_Y1 * n[1]) +
         sphere.shCoeffs_Low[1] * (k_Y1 * n[2]) +
         sphere.shCoeffs_Low[2] * (k_Y1 * n[0]);

    float k_Y2 = 0.25f * std::sqrt(5.0f / PI_F);
    R += sphere.shCoeffs_Low[3] * (k_Y2 * (3.0f * n[2] * n[2] - 1.0f));

    return std::max(R, 1e-6f);
}

// Host-side Boundary Crossing and Hysteresis Filtering logic
inline void ProcessBoundaryCrossingHost(
    const float obsPos[3],
    SpatiotemporalSphere& parentSphere,
    std::vector<SpatiotemporalSphere>& childSpheres,
    float hysteresisEpsilon
) {
    float relPos[3] = {
        obsPos[0] - parentSphere.centerPos[0],
        obsPos[1] - parentSphere.centerPos[1],
        obsPos[2] - parentSphere.centerPos[2]
    };

    float dist = std::sqrt(relPos[0] * relPos[0] + relPos[1] * relPos[1] + relPos[2] * relPos[2]);
    float n[3] = { 0.0f, 0.0f, 1.0f };
    if (dist > 1e-6f) {
        n[0] = relPos[0] / dist;
        n[1] = relPos[1] / dist;
        n[2] = relPos[2] / dist;
    }

    float dynamicRadius = EvaluateSphereRadiusDirectionHost(parentSphere, n);

    bool isCurrentlyInside = (parentSphere.flags & 0x01) != 0;

    // 1. Boundary Infiltration (Unrolling Child Bubbles)
    if (!isCurrentlyInside && (dist <= dynamicRadius - hysteresisEpsilon)) {
        parentSphere.flags |= 0x01; // Set IsInfiltrated = true

        for (auto& child : childSpheres) {
            child.flags |= 0x02; // Set IsActive = true
            child.surfaceTension += parentSphere.surfaceTension * 0.15f; // Boundary coupling
        }
    }
    // 2. Boundary Egress (Pruning Child Bubbles)
    else if (isCurrentlyInside && (dist > dynamicRadius + hysteresisEpsilon)) {
        parentSphere.flags &= ~0x01; // Set IsInfiltrated = false

        for (auto& child : childSpheres) {
            child.flags &= ~0x02; // Set IsActive = false
        }
    }
}

#endif // __cplusplus

#endif // SPATIOTEMPORAL_SPHERE_H
