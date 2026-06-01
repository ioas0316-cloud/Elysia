#include "phase_transformer.h"
#include <bit>
#include <numbers>
#include <algorithm>

namespace elysia {

PhaseSignature PhaseTransformer::transform_to_wave(const uint64_t block[8]) {
    float amp = calculate_amplitude(block);
    float freq = calculate_frequency(block);
    Quaternion phase = calculate_phase_angle(amp, freq, block);

    // Extract Hangul Rotor elements instead of legacy associative axes
    uint64_t choseong = extract_choseong_tension(block);
    VowelPhase jungseong = extract_jungseong_phase(amp, freq, phase);
    uint64_t jongseong = extract_jongseong_anchor(block);

    // If the data is highly chaotic (Zero-day attack pattern), map to Double Consonants
    if (amp > 0.85f && freq > 0.8f) {
        choseong = static_cast<uint64_t>(PhoneticBase::VELAR_GG); // High-tension absorption
        jongseong = static_cast<uint64_t>(PhoneticBase::DENTAL_SS); // High-friction grounding
    }

    // Extract the absolute hardware constant (Immutable Record - Yin)
    uint64_t anchor = generate_immutable_anchor(block);

    return PhaseSignature{
        amp, freq, phase,
        choseong, jungseong, jongseong,
        anchor
    };
}

PhaseSignature PhaseTransformer::invert_wave(const PhaseSignature& original_wave) {
    PhaseSignature inverted = original_wave;

    // Cognitive Judo: Invert the phase and amplitude to absorb the attack
    inverted.amplitude = 1.0f - original_wave.amplitude;
    inverted.frequency = 1.0f - original_wave.frequency;

    // Invert the Vowel Phase Angle (Reverse rotation direction)
    inverted.jungseong_phase.heaven_pivot = 1.0f - original_wave.jungseong_phase.heaven_pivot;
    inverted.jungseong_phase.earth_axis = -original_wave.jungseong_phase.earth_axis;
    inverted.jungseong_phase.human_axis = -original_wave.jungseong_phase.human_axis;

    // The tension masks remain the same (Double Consonants), as they provide the structural
    // "vessel" to absorb the inverted energy without breaking.
    return inverted;
}

uint64_t PhaseTransformer::extract_choseong_tension(const uint64_t block[8]) {
    // Determine the base consonant tension based on bit distribution in the first half of the block
    int mass = 0;
    for(int i=0; i<4; ++i) mass += std::popcount(block[i]);

    // Map mass to one of the 5 base phonetic tensions
    if (mass % 5 == 0) return static_cast<uint64_t>(PhoneticBase::VELAR_G);
    if (mass % 5 == 1) return static_cast<uint64_t>(PhoneticBase::LINGUAL_N);
    if (mass % 5 == 2) return static_cast<uint64_t>(PhoneticBase::LABIAL_M);
    if (mass % 5 == 3) return static_cast<uint64_t>(PhoneticBase::DENTAL_S);
    return static_cast<uint64_t>(PhoneticBase::GLOTTAL_NG);
}

VowelPhase PhaseTransformer::extract_jungseong_phase(float amplitude, float frequency, const Quaternion& angle) {
    // Map wave properties to Heaven, Earth, Human dimensions
    VowelPhase vp;
    vp.heaven_pivot = amplitude; // Overall magnitude drives the pivot
    vp.earth_axis = std::abs(angle.x); // Horizontal projection
    vp.human_axis = std::abs(angle.z); // Vertical projection
    return vp;
}

uint64_t PhaseTransformer::extract_jongseong_anchor(const uint64_t block[8]) {
     // Determine the settling consonant tension based on the second half of the block
    int mass = 0;
    for(int i=4; i<8; ++i) mass += std::popcount(block[i]);

    // Settling might be empty (Glottal/Void) more often
    if (mass < 64) return static_cast<uint64_t>(PhoneticBase::GLOTTAL_NG);

    if (mass % 4 == 0) return static_cast<uint64_t>(PhoneticBase::VELAR_G);
    if (mass % 4 == 1) return static_cast<uint64_t>(PhoneticBase::LINGUAL_N);
    if (mass % 4 == 2) return static_cast<uint64_t>(PhoneticBase::LABIAL_M);
    return static_cast<uint64_t>(PhoneticBase::DENTAL_S);
}

uint64_t PhaseTransformer::generate_immutable_anchor(const uint64_t block[8]) {
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
    return static_cast<float>(total_set_bits) / 512.0f;
}

float PhaseTransformer::calculate_frequency(const uint64_t block[8]) {
    int transitions = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t val = block[i];
        uint64_t flips = val ^ (val >> 1);
        transitions += std::popcount(flips) - (val >> 63);
        if (i < 7) {
            uint64_t last_bit_current = block[i] >> 63;
            uint64_t first_bit_next = block[i+1] & 1;
            if (last_bit_current != first_bit_next) {
                transitions++;
            }
        }
    }
    return static_cast<float>(transitions) / 511.0f;
}

Quaternion PhaseTransformer::calculate_phase_angle(float amplitude, float frequency, const uint64_t block[8]) {
    float theta = amplitude * std::numbers::pi_v<float>;
    float phi = frequency * std::numbers::pi_v<float>;

    float spatial_variance = 0.0f;
    for (int i = 0; i < 4; ++i) {
         spatial_variance += std::popcount(block[i]);
    }
    spatial_variance /= (4.0f * 64.0f);

    float gamma = spatial_variance * std::numbers::pi_v<float>;

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
