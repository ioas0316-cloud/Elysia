#include "phase_transformer.h"
#include <bit>
#include <numbers>

namespace elysia {

PhaseSignature PhaseTransformer::transform_to_wave(const uint64_t block[8]) {
    float amp = calculate_amplitude(block);
    float freq = calculate_frequency(block);
    Quaternion phase = calculate_phase_angle(amp, freq, block);

    // Dilute data into the 4 Associative Axes (Variable Cognition - Yang)
    float rel = calculate_relationship(amp, freq);
    float conn = calculate_connectivity(block);
    float kin = calculate_kinematics(freq);
    float dir = calculate_directionality(phase);

    // Extract the absolute hardware constant (Immutable Record - Yin)
    uint64_t anchor = generate_immutable_anchor(block);

    return PhaseSignature{
        amp, freq, phase,
        rel, conn, kin, dir,
        anchor
    };
}

float PhaseTransformer::calculate_relationship(float amplitude, float frequency) {
    // Relationship: How intense is the wave compared to the baseline?
    // Calculated as the harmony between mass (amplitude) and chaos (frequency).
    return std::abs(amplitude - frequency);
}

float PhaseTransformer::calculate_connectivity(const uint64_t block[8]) {
    // Connectivity: Resonance factor that dictates adjacent memory linking.
    // We derive this from the symmetry of the bit block (Left vs Right halves).
    int left_mass = 0;
    int right_mass = 0;
    for (int i=0; i<4; ++i) left_mass += std::popcount(block[i]);
    for (int i=4; i<8; ++i) right_mass += std::popcount(block[i]);

    float total_mass = static_cast<float>(left_mass + right_mass);
    if (total_mass == 0.0f) return 0.0f;

    return static_cast<float>(std::min(left_mass, right_mass)) / total_mass;
}

float PhaseTransformer::calculate_kinematics(float frequency) {
    // Kinematics: The kinetic energy / rhythm of the wave.
    // Driven heavily by the frequency of bit transitions.
    return std::sqrt(frequency);
}

float PhaseTransformer::calculate_directionality(const Quaternion& angle) {
    // Directionality: The causal trajectory vector length
    // Extracted from the spatial orientation elements (x, y, z)
    return std::sqrt(angle.x * angle.x + angle.y * angle.y + angle.z * angle.z);
}

uint64_t PhaseTransformer::generate_immutable_anchor(const uint64_t block[8]) {
    // The Immutable Anchor: A non-variable constant derived purely from the physical silicon state.
    // Acts as the Yin baseline preventing memory corruption from the Yang variable dials.
    // Uses a deterministic hash (FNV-1a 64-bit basis) of the raw 512-bit block.
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (int i = 0; i < 8; ++i) {
        hash ^= block[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
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
