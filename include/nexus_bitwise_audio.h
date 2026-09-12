#ifndef NEXUS_BITWISE_AUDIO_H
#define NEXUS_BITWISE_AUDIO_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace CausalNexus {

struct AudioVisualVoltageFrame {
    // Visual Voltage Output Buffer (RGB 3-channel voltage [0.0, 1.0])
    std::vector<float> visual_voltage_rgb;
    // Spatial Audio Surround DSP Voltage Registers (Left, Right, Center, LFE, Surround L, Surround R)
    float audio_dsp_registers[6];
};

class BitwiseCausalSimulator {
public:
    /**
     * Executes 1-Bit BNN XNOR + POPCNT SIMD step between weight packed bitmask and input bitmask
     */
    static std::vector<uint8_t> ExecuteBitwiseXnorPopcnt(
        const std::vector<uint64_t>& input_bit_blocks,
        const std::vector<uint64_t>& weight_bit_blocks
    ) {
        size_t n_blocks = std::min(input_bit_blocks.size(), weight_bit_blocks.size());
        std::vector<uint8_t> bit_activations(n_blocks * 64, 0);

        for (size_t i = 0; i < n_blocks; ++i) {
            // XNOR logic: ~(A ^ B)
            uint64_t xnor_result = ~(input_bit_blocks[i] ^ weight_bit_blocks[i]);

            for (int b = 0; b < 64; ++b) {
                bit_activations[i * 64 + b] = (xnor_result & (1ULL << b)) ? 1 : 0;
            }
        }
        return bit_activations;
    }

    /**
     * Splits causal bitmask voltage signal into both visual voltage buffer and audio DSP surround registers
     */
    static AudioVisualVoltageFrame SplitAudioVisualVoltage(
        const std::vector<uint8_t>& trajectory_bits,
        const std::vector<uint8_t>& hitbox_bits,
        int height,
        int width
    ) {
        AudioVisualVoltageFrame frame;
        frame.visual_voltage_rgb.resize(height * width * 3, 0.0f);

        size_t total_pixels = height * width;
        double active_traj_count = 0;
        double active_hitbox_count = 0;
        double traj_center_x = 0;

        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                size_t idx = r * width + c;
                uint8_t traj = (idx < trajectory_bits.size()) ? trajectory_bits[idx] : 0;
                uint8_t hit = (idx < hitbox_bits.size()) ? hitbox_bits[idx] : 0;

                float v_red = traj ? 1.0f : 0.0f;
                float v_green = hit ? 1.0f : 0.0f;
                float v_blue = (traj || hit) ? 1.0f : 0.0f;

                frame.visual_voltage_rgb[idx * 3 + 0] = v_red;
                frame.visual_voltage_rgb[idx * 3 + 1] = v_green;
                frame.visual_voltage_rgb[idx * 3 + 2] = v_blue;

                if (traj) {
                    active_traj_count += 1.0;
                    traj_center_x += static_cast<double>(c) / width;
                }
                if (hit) {
                    active_hitbox_count += 1.0;
                }
            }
        }

        double norm_x = (active_traj_count > 0) ? (traj_center_x / active_traj_count) : 0.5;
        float intensity = static_cast<float>(active_traj_count / std::max(1.0, static_cast<double>(total_pixels)));
        float hit_intensity = static_cast<float>(active_hitbox_count / std::max(1.0, static_cast<double>(total_pixels)));

        // Audio DSP 5.1 Surround voltage channels
        frame.audio_dsp_registers[0] = static_cast<float>((1.0 - norm_x) * intensity * 2.0); // Left
        frame.audio_dsp_registers[1] = static_cast<float>(norm_x * intensity * 2.0);         // Right
        frame.audio_dsp_registers[2] = (intensity + hit_intensity) * 0.5f;                  // Center
        frame.audio_dsp_registers[3] = hit_intensity * 3.0f;                                // LFE (Low Frequency Impact)
        frame.audio_dsp_registers[4] = static_cast<float>((1.0 - norm_x) * hit_intensity);   // Surround L
        frame.audio_dsp_registers[5] = static_cast<float>(norm_x * hit_intensity);           // Surround R

        return frame;
    }
};

} // namespace CausalNexus

#endif // NEXUS_BITWISE_AUDIO_H
