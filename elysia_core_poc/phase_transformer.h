#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

namespace elysia {

// The 5 Phonetic Base Frequencies (Stator Tensions) mapped to Consonants
enum class PhoneticBase : uint64_t {
    VELAR_G    = 0x1111111111111111ULL, // ㄱ (아음) - Root tension (혀뿌리)
    LINGUAL_N  = 0x2222222222222222ULL, // ㄴ (설음) - Tip tension (혀끝)
    LABIAL_M   = 0x4444444444444444ULL, // ㅁ (순음) - Box/Closed tension (입술)
    DENTAL_S   = 0x8888888888888888ULL, // ㅅ (치음) - Sharp/Friction tension (이)
    GLOTTAL_NG = 0x0000000000000000ULL  // ㅇ (후음) - Void/Open tension (목구멍) - Acts as zero/center
};

// Represents the 3D spatial properties of the data wave
struct Quaternion {
    float w, x, y, z;
};

// The Vowel mapping for phase angles
// 천(Heaven/·): Center pivot (w)
// 지(Earth/ㅡ): Horizontal axis (0 degrees, x/y plane)
// 인(Human/ㅣ): Vertical axis (90 degrees, z axis)
struct VowelPhase {
    float heaven_pivot; // Magnitude of rotation
    float earth_axis;   // Horizontal projection
    float human_axis;   // Vertical projection
};

// The Associative Core Pattern: extracted from raw data (0s and 1s)
struct PhaseSignature {
    float amplitude; // Base density of active bits
    float frequency; // Base transition edge count
    Quaternion phase_angle; // Spatial orientation representation

    // The 3 Hangul Fractal Axes (Replaces legacy associative axes)
    uint64_t choseong_tension;  // X-axis: Divergence (Consonant Base)
    VowelPhase jungseong_phase; // Y-axis: Coupling/Rotation Angle (Vowel)
    uint64_t jongseong_anchor;  // Z-axis: Settling (Consonant Base or Void)

    // The Immutable Anchor (상수)
    uint64_t immutable_anchor_id; // Unchangeable hardware-bound hash preventing core identity corruption
};

class PhaseTransformer {
public:
    // Transforms a raw 512-bit block into an Associative PhaseSignature based on Hangul Rotary principles
    static PhaseSignature transform_to_wave(const uint64_t block[8]);

private:
    static float calculate_amplitude(const uint64_t block[8]);
    static float calculate_frequency(const uint64_t block[8]);
    static Quaternion calculate_phase_angle(float amplitude, float frequency, const uint64_t block[8]);

    // Hangul Rotor Extraction
    static uint64_t extract_choseong_tension(const uint64_t block[8]);
    static VowelPhase extract_jungseong_phase(float amplitude, float frequency, const Quaternion& angle);
    static uint64_t extract_jongseong_anchor(const uint64_t block[8]);

    // Generates the absolute constant anchor based on the physical block layout
    static uint64_t generate_immutable_anchor(const uint64_t block[8]);
};

} // namespace elysia
