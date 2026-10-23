#ifndef CAUSAL_ENGINE_CAUSAL_BIT_MATRIX_H
#define CAUSAL_ENGINE_CAUSAL_BIT_MATRIX_H

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <chrono>
#include <omp.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace causal_engine {

// 64-Byte aligned Zero-Abstraction Causal Bit Matrix
template <size_t WIDTH = 512, size_t HEIGHT = 512>
struct alignas(64) CausalBitMatrix {
    static constexpr size_t GRID_SIZE = WIDTH * HEIGHT;
    static constexpr size_t WORD_COUNT = GRID_SIZE / 64; // uint64_t words count

    uint64_t hitbox_bits[WORD_COUNT];         // Physics collision domain
    uint64_t trajectory_bits[WORD_COUNT];     // Motion vector trajectory domain
    uint64_t shader_voltage_bits[WORD_COUNT]; // GPU shader activation voltage mask

    uint64_t frame_tick;

    CausalBitMatrix() : frame_tick(0) {
        std::memset(hitbox_bits, 0, sizeof(hitbox_bits));
        std::memset(trajectory_bits, 0, sizeof(trajectory_bits));
        std::memset(shader_voltage_bits, 0, sizeof(shader_voltage_bits));
    }
};

template <size_t WIDTH = 512, size_t HEIGHT = 512>
class ZeroAbstractionCausalEngine {
public:
    using MatrixType = CausalBitMatrix<WIDTH, HEIGHT>;

private:
    MatrixType state;
    void* vram_shared_ptr;

public:
    ZeroAbstractionCausalEngine() : vram_shared_ptr(nullptr) {}

    // Direct binding to shared CPU/GPU UMA memory ($O(1)$ zero-copy allocation)
    void BindVRAMBuffer(void* external_vram_ptr) {
        vram_shared_ptr = external_vram_ptr;
    }

    // O(1) Bitwise Flip
    inline void FlipBit(size_t x, size_t y, uint8_t channel) {
        size_t pixel_idx = y * WIDTH + x;
        size_t word_idx = pixel_idx / 64;
        uint64_t bit_mask = 1ULL << (pixel_idx % 64);

        switch (channel) {
            case 0: state.hitbox_bits[word_idx] ^= bit_mask; break;
            case 1: state.trajectory_bits[word_idx] ^= bit_mask; break;
            case 2: state.shader_voltage_bits[word_idx] ^= bit_mask; break;
        }
    }

    // Vectorized SIMD Multi-Core Tick Execution
    void Tick() {
        state.frame_tick++;

#if defined(__AVX512F__)
        constexpr size_t AVX512_STEP = MatrixType::WORD_COUNT / 8;
        const __m512i* traj = reinterpret_cast<const __m512i*>(state.trajectory_bits);
        const __m512i* hit  = reinterpret_cast<const __m512i*>(state.hitbox_bits);
        __m512i* volt       = reinterpret_cast<__m512i*>(state.shader_voltage_bits);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(AVX512_STEP); ++i) {
            __m512i v_traj = _mm512_load_si512(&traj[i]);
            __m512i v_hit  = _mm512_load_si512(&hit[i]);
            __m512i v_res  = _mm512_and_si512(v_traj, v_hit);
            _mm512_store_si512(&volt[i], v_res);
        }

#elif defined(__AVX2__)
        constexpr size_t AVX2_STEP = MatrixType::WORD_COUNT / 4;
        const __m256i* traj = reinterpret_cast<const __m256i*>(state.trajectory_bits);
        const __m256i* hit  = reinterpret_cast<const __m256i*>(state.hitbox_bits);
        __m256i* volt       = reinterpret_cast<__m256i*>(state.shader_voltage_bits);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(AVX2_STEP); ++i) {
            __m256i v_traj = _mm256_load_si256(&traj[i]);
            __m256i v_hit  = _mm256_load_si256(&hit[i]);
            __m256i v_res  = _mm256_and_si256(v_traj, v_hit);
            _mm256_store_si256(&volt[i], v_res);
        }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        constexpr size_t NEON_STEP = MatrixType::WORD_COUNT / 2;
        const uint64_t* traj = state.trajectory_bits;
        const uint64_t* hit  = state.hitbox_bits;
        uint64_t* volt       = state.shader_voltage_bits;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(NEON_STEP); ++i) {
            uint64x2_t v_traj = vld1q_u64(&traj[i * 2]);
            uint64x2_t v_hit  = vld1q_u64(&hit[i * 2]);
            uint64x2_t v_res  = vandq_u64(v_traj, v_hit);
            vst1q_u64(&volt[i * 2], v_res);
        }

#else
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(MatrixType::WORD_COUNT); ++i) {
            state.shader_voltage_bits[i] = state.trajectory_bits[i] & state.hitbox_bits[i];
        }
#endif

        if (vram_shared_ptr) {
            std::memcpy(vram_shared_ptr, &state, sizeof(MatrixType));
        }
    }

    const MatrixType& GetState() const { return state; }
    MatrixType& GetMutableState() { return state; }
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CAUSAL_BIT_MATRIX_H
