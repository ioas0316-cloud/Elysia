#include "phase_transformer.h"
#include <bit>
#include <numbers>

namespace elysia {

PhaseSignature PhaseTransformer::transform_to_wave(const uint64_t block[8]) {
    float amp = calculate_amplitude(block);
    float freq = calculate_frequency(block);
    Quaternion phase = calculate_phase_angle(amp, freq, block);

    return PhaseSignature{amp, freq, phase};
}

float PhaseTransformer::calculate_amplitude(const uint64_t block[8]) {
    int total_set_bits = 0;
    for (int i = 0; i < 8; ++i) {
        total_set_bits += std::popcount(block[i]);
    }
    // Return density [0, 1] as amplitude
    return static_cast<float>(total_set_bits) / 512.0f;
}

float PhaseTransformer::calculate_frequency(const uint64_t block[8]) {
    int transitions = 0;

    // Check transitions within each 64-bit chunk
    for (int i = 0; i < 8; ++i) {
        uint64_t val = block[i];
        // XOR with shifted value to find bit flips.
        // A bit flip results in a 1 in the XOR result.
        uint64_t flips = val ^ (val >> 1);

        // Count set bits, subtract 1 if the MSB was set (since shift introduces 0)
        transitions += std::popcount(flips) - (val >> 63);

        // Check transition between chunks
        if (i < 7) {
            uint64_t last_bit_current = block[i] >> 63;
            uint64_t first_bit_next = block[i+1] & 1;
            if (last_bit_current != first_bit_next) {
                transitions++;
            }
        }
    }

    // Normalize frequency [0, 1]
    return static_cast<float>(transitions) / 511.0f;
}

Quaternion PhaseTransformer::calculate_phase_angle(float amplitude, float frequency, const uint64_t block[8]) {
    // Generate a deterministic spatial orientation based on data distribution
    // This maps the data density (mass) and frequency to a Quaternion

    // Simplistic deterministic mapping for PoC
    float theta = amplitude * std::numbers::pi_v<float>; // Angle based on mass
    float phi = frequency * std::numbers::pi_v<float>;   // Angle based on frequency

    // Add some spatial variance based on the actual bit layout
    float spatial_variance = 0.0f;
    for (int i = 0; i < 4; ++i) {
         spatial_variance += std::popcount(block[i]);
    }
    spatial_variance /= (4.0f * 64.0f); // Ratio of first half

    float gamma = spatial_variance * std::numbers::pi_v<float>;

    // Euler to Quaternion (XYZ sequence roughly)
    float cy = cos(gamma * 0.5);
    float sy = sin(gamma * 0.5);
    float cp = cos(phi * 0.5);
    float sp = sin(phi * 0.5);
    float cr = cos(theta * 0.5);
    float sr = sin(theta * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

} // namespace elysia
