#ifndef CAUSAL_ROTOR_TENSOR_HPP
#define CAUSAL_ROTOR_TENSOR_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <memory>
#include <string>

// CUDA 또는 HLSL과의 상호운용성을 위한 4D, 3D float 벡터 타입 정의 (C++ 단독 빌드 지원)
#ifndef __CUDACC__
struct Float4 {
    float x, y, z, w;
    Float4() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
    Float4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

struct Float3 {
    float x, y, z;
    Float3() : x(0.0f), y(0.0f), z(0.0f) {}
    Float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
#else
#include <cuda_runtime.h>
using Float4 = float4;
using Float3 = float3;
#endif

namespace elysia {

// ============================================================================
// 128바이트 수평 정렬 RotorNodeData 구조체
// CUDA Warp Memory Coalescing & VRAM Slab Allocator & HLSL Compute Shader 호환
// ============================================================================
struct alignas(128) RotorNodeData {
    // 1. 위상 및 동역학 상태 (32 Bytes)
    Float4   quaternion;       // 16 Bytes: q = (x, y, z, w) - 맥락적 위상 상태
    Float3   angularVelocity;  // 12 Bytes: omega = (wx, wy, wz) - so(3) 각속도
    float    dampingBeta;      // 4 Bytes: 시스템 마찰/감쇄 계수

    // 2. 공간 및 인과 식별자 (32 Bytes)
    uint64_t spatialHashKey;   // 8 Bytes: Morton Code 64-bit 공간 해시 키
    uint32_t usageFrequency;   // 4 Bytes: LRU Eviction 방어용 사용 빈도
    uint32_t isAttractor;      // 4 Bytes: Phase-Lock 달성 여부 (1: Attractor Basin, 0: Free Rotor)
    uint64_t vramSlabAddress;  // 8 Bytes: DirectStorage / D3D12 Causal Page Allocator VRAM 주소
    uint32_t reserved0;        // 4 Bytes: alignment 패딩
    uint32_t reserved1;        // 4 Bytes: 추가 패딩

    // 3. 기어 바인딩 텐서 인덱스 (64 Bytes: 최대 8개 기어 커플링 엣지)
    uint32_t connectedNodeIndices[8]; // 32 Bytes: 연결된 주변 노드 Index (0xFFFFFFFF = 연결 없음)
    float    gearRatios[8];           // 32 Bytes: 연결 기어비 (Gamma_ij)

    RotorNodeData() {
        quaternion = Float4(0.0f, 0.0f, 0.0f, 1.0f);
        angularVelocity = Float3(0.0f, 0.0f, 0.0f);
        dampingBeta = 0.1f;
        spatialHashKey = 0;
        usageFrequency = 0;
        isAttractor = 0;
        vramSlabAddress = 0;
        reserved0 = 0;
        reserved1 = 0;
        for (int i = 0; i < 8; ++i) {
            connectedNodeIndices[i] = 0xFFFFFFFF;
            gearRatios[i] = 1.0f;
        }
    }
};

static_assert(sizeof(RotorNodeData) == 128, "RotorNodeData must be exactly 128 bytes for VRAM Slab alignment.");

// ============================================================================
// Morton 3D Code (Z-order Curve) 64-bit 비트 인터리빙 인코딩/디코딩 함수
// ============================================================================
class Morton3D {
public:
    // 21비트 정수 좌표 (x, y, z) -> 64비트 Morton Code 변환
    static uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        return (splitBy3(x) << 0) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
    }

    // 64비트 Morton Code -> 21비트 정수 좌표 (x, y, z) 복원
    static void decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
        x = compactBy3(code >> 0);
        y = compactBy3(code >> 1);
        z = compactBy3(code >> 2);
    }

private:
    static uint64_t splitBy3(uint32_t a) {
        uint64_t x = a & 0x1fffff; // 21 bits
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
        x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
        x = (x | (x << 2))  & 0x1249249249249249ULL;
        return x;
    }

    static uint32_t compactBy3(uint64_t code) {
        uint64_t x = code & 0x1249249249249249ULL;
        x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3ULL;
        x = (x ^ (x >> 4))  & 0x100f00f00f00f00fULL;
        x = (x ^ (x >> 8))  & 0x1f0000ff0000ffULL;
        x = (x ^ (x >> 16)) & 0x1f00000000ffffULL;
        x = (x ^ (x >> 32)) & 0x1fffffULL;
        return static_cast<uint32_t>(x);
    }
};

// ============================================================================
// Dynamic Causal Rotor Engine Class Interface
// ============================================================================
class CausalRotorTensorSystem {
public:
    CausalRotorTensorSystem(uint32_t maxNodes = 65536);
    ~CausalRotorTensorSystem();

    // 로터 노드 추가 및 연결
    uint32_t addRotorNode(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          float dampingBeta = 0.05f);

    bool connectNodes(uint32_t sourceIdx, uint32_t targetIdx, float gearRatio);

    // 각속도 동력 주입 (Trigger Impulse / Continuous Torque)
    void injectImpulse(uint32_t nodeIdx, float wx, float wy, float wz);

    // Causal Rotor Kinematics & Phase-Lock Dynamics 적분 실행 (CPU or CUDA)
    void stepSimulation(float deltaTime, float lockTolerance = 0.01f);

    // Attractor Basin 상태 및 VRAM Hot-Cache Eviction 상태 읽기
    uint32_t getAttractorCount() const;
    std::vector<RotorNodeData> getNodeBuffer() const;
    const RotorNodeData* getRawDataPointer() const;

    uint32_t getNodeCount() const { return m_nodeCount; }

private:
    uint32_t m_maxNodes;
    uint32_t m_nodeCount;
    std::vector<RotorNodeData> m_nodes;

    // CUDA Device Pointer (CUDA 빌드 시 사용)
    RotorNodeData* d_nodesBuffer;
    bool m_cudaInitialized;

    void initializeCUDA();
    void freeCUDA();
};

} // namespace elysia

#endif // CAUSAL_ROTOR_TENSOR_HPP
