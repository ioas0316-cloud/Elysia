#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__CUDACC__)
#define DEVICE_HOST __device__ __host__
#define DEVICE_INLINE __device__ inline
#else
#define DEVICE_HOST inline
#define DEVICE_INLINE inline
#endif

// 64-bit hardware 1:1 mapped Causal Node Layout (Substrate Layout)
// Bit 0 ~ 7: Phase State (00: Gas, 01: Fluid, 11: Crystal)
// Bit 8 ~ 31: Scalar Potential (24-bit integer Psi)
// Bit 32 ~ 47: Bonding Operator ID (16-bit tensor code)
// Bit 48 ~ 63: Topology Relative Offset (16-bit signed int offset)

#define PHASE_GAS     0x00
#define PHASE_FLUID   0x01
#define PHASE_CRYSTAL 0x03

struct alignas(8) CausalNode {
    uint64_t raw;

    DEVICE_HOST uint8_t getPhaseState() const {
        return static_cast<uint8_t>(raw & 0xFFULL);
    }

    DEVICE_HOST uint32_t getPotential() const {
        return static_cast<uint32_t>((raw >> 8) & 0xFFFFFFULL);
    }

    DEVICE_HOST uint16_t getBondOperator() const {
        return static_cast<uint16_t>((raw >> 32) & 0xFFFFULL);
    }

    DEVICE_HOST int16_t getTopoOffset() const {
        return static_cast<int16_t>((raw >> 48) & 0xFFFFULL);
    }

    DEVICE_HOST void updatePhaseState(uint8_t newPhase) {
        raw = (raw & ~0xFFULL) | (static_cast<uint64_t>(newPhase) & 0xFFULL);
    }

    DEVICE_HOST void updatePotential(uint32_t potential) {
        raw = (raw & ~(0xFFFFFFULL << 8)) | ((static_cast<uint64_t>(potential) & 0xFFFFFFULL) << 8);
    }

    DEVICE_HOST void updateBondOperator(uint16_t bondOp) {
        raw = (raw & ~(0xFFFFULL << 32)) | ((static_cast<uint64_t>(bondOp) & 0xFFFFULL) << 32);
    }

    DEVICE_HOST void updateTopoOffset(int16_t offset) {
        raw = (raw & ~(0xFFFFULL << 48)) | ((static_cast<uint64_t>(static_cast<uint16_t>(offset)) & 0xFFFFULL) << 48);
    }
};

// Zero-Copy 3D Gaussian Splatting + 64-bit CausalNode Layout Interop Structure
struct alignas(16) CausalGaussianSplat {
    float posX, posY, posZ;
    float rotW, rotX, rotY, rotZ;
    float scaleX, scaleY, scaleZ;
    float opacity;
    uint64_t causalRaw;
};
