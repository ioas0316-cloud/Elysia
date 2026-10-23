#ifndef CAUSAL_ENGINE_HYPER_PIXEL_H
#define CAUSAL_ENGINE_HYPER_PIXEL_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace causal_engine {

// Morton Code (Z-Order Curve) 2D/3D Bit Interleaving Utilities
class MortonUtils {
public:
    // Expand an 8-bit integer into 16-bit by inserting 0s between bits
    static inline uint32_t expandBits(uint32_t v) {
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        return v;
    }

    // Encode 2D coordinates (x, y) into a 32-bit Morton code
    static inline uint32_t encode2D(uint16_t x, uint16_t y) {
        return (expandBits(y) << 1) | expandBits(x);
    }

    // Encode 2D 32-bit coordinates into 64-bit Morton code
    static inline uint64_t encode2D_64(uint32_t x, uint32_t y) {
        uint64_t x64 = x;
        uint64_t y64 = y;
        x64 = (x64 | (x64 << 16)) & 0x0000FFFF0000FFFFULL;
        x64 = (x64 | (x64 << 8))  & 0x00FF00FF00FF00FFULL;
        x64 = (x64 | (x64 << 4))  & 0x0F0F0F0F0F0F0F0FULL;
        x64 = (x64 | (x64 << 2))  & 0x3333333333333333ULL;
        x64 = (x64 | (x64 << 1))  & 0x5555555555555555ULL;

        y64 = (y64 | (y64 << 16)) & 0x0000FFFF0000FFFFULL;
        y64 = (y64 | (y64 << 8))  & 0x00FF00FF00FF00FFULL;
        y64 = (y64 | (y64 << 4))  & 0x0F0F0F0F0F0F0F0FULL;
        y64 = (y64 | (y64 << 2))  & 0x3333333333333333ULL;
        y64 = (y64 | (y64 << 1))  & 0x5555555555555555ULL;

        return (y64 << 1) | x64;
    }
};

// 256-bit AVX2/AVX-512 register aligned Hyper-Pixel Node
struct alignas(32) HyperPixelNode {
    uint64_t spatial_morton_code; // Morton Code compressed space-time coordinates
    uint64_t causal_edge_mask;    // Direct topology binding bits with neighboring nodes
    uint64_t physics_state_flags; // Physics, hitbox, trajectory voltage register
    uint64_t shader_voltage_data; // GPU display direct-drive voltage & spectral mask

    // Evaluate 4 x 64-bit attributes in a single 256-bit AVX2 instruction O(1)
    inline void EvaluateCausalState(const HyperPixelNode& input_node) {
#if defined(__AVX2__)
        __m256i* self_reg = reinterpret_cast<__m256i*>(this);
        const __m256i* in_reg = reinterpret_cast<const __m256i*>(&input_node);
        *self_reg = _mm256_and_si256(*self_reg, *in_reg);
#else
        spatial_morton_code &= input_node.spatial_morton_code;
        causal_edge_mask &= input_node.causal_edge_mask;
        physics_state_flags &= input_node.physics_state_flags;
        shader_voltage_data &= input_node.shader_voltage_data;
#endif
    }
};

// Morton Quadtree Bitmask Early-Culling Engine for high resolution (4K/8K)
struct MortonQuadtreeCuller {
    // 64x64 pixel block root bit mask (4096 pixels)
    // If block mask == 0, O(1) skip 4,096 pixels
    static inline bool ShouldCullBlock(uint64_t block_causal_mask) {
        return block_causal_mask == 0ULL;
    }
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_HYPER_PIXEL_H
