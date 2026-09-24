#ifndef ELYSIA_CAUSAL_LUT_HEADER_H
#define ELYSIA_CAUSAL_LUT_HEADER_H

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>
#else
#ifndef VECTOR_TYPES_H
struct float4 {
    float x, y, z, w;
};
inline float4 make_float4(float x, float y, float z, float w) {
    float4 f = {x, y, z, w};
    return f;
}
#endif
#endif

#pragma pack(push, 1)

// SSD 파일 최상단에 기록되는 전역 메타 헤더 (4KB Sector Aligned)
struct alignas(4096) DirectStorageFileHeader {
    char     magic_bytes[8];      // "ELYSIAN1" 식별자
    uint32_t schema_version;      // 엔진 아키텍처 버전
    uint64_t total_baked_pages;   // NVMe SSD에 최종 고착화된 Causal Page 개수
    uint64_t page_table_offset;   // 파일 내 Page Table 시작 오프셋
    uint64_t data_region_offset;  // 파일 내 실시간 Stream Data 시작 오프셋
    uint32_t codebook_size;       // VQ Codebook 엔트리 수 (예: 256 또는 1024)
    uint8_t  reserved[4048];      // 4096 바이트 섹터 정렬용 패딩
};

// 64바이트 CPU/GPU 캐시라인에 최적화된 개별 인과 페이지 엔트리
struct alignas(64) CausalPageEntry {
    uint64_t spatial_hash_key;    // 3D 공간 해시 버킷 고유 키
    uint64_t nvme_sector_offset;  // NVMe SSD 물리 오프셋 (DirectStorage 읽기 지점)
    uint32_t payload_size_bytes;  // 베이킹된 VAT 데이터 바이트 크기
    uint16_t vq_codebook_idx;     // 추상화된 원형 메타 코드북 인덱스

    // 비트 필드 플래그 (상태 제어)
    uint8_t  is_baked        : 1; // 1: SSD 베이킹 완료, 0: VRAM 링버퍼 상주 중
    uint8_t  vram_pinned     : 1; // 1: Track A Hot Cache에 고정 할당됨
    uint8_t  is_attractor    : 1; // 1: 끌개 계곡(Attractor Valley), 0: 일반 궤적
    uint8_t  has_bifurcation : 1; // 1: 임계점(Ridge) 분기 이력 존재
    uint8_t  reserved_flags  : 4;

    uint32_t last_access_tick;   // LRU 캐시 교체 알고리즘용 타임스탬프
    uint32_t usage_frequency;    // 인과 소환 빈도 (Attractor 깊이 측정용)

    // DirectStorage 및 CUDA 메모리 포인터 주소
    uint64_t vram_page_address;   // Hot-Cache 상태일 때의 VRAM 가상 주소 (Void*)
    uint8_t  padding[20];         // 64바이트 정렬을 위한 패딩
};

#pragma pack(pop)

// 16-Channel 통합 감각 데이터 구조체
struct UnifiedSensoryFrame {
    float4 visual_depth;    // R, G, B, Depth
    float4 acoustic_phase;  // Amp, Freq, Sin(Phase), Cos(Phase)
    float4 physical_force;  // Fx, Fy, Fz, Viscosity
    float4 contextual_meta; // Entropy, ContextID, Prior_1, Prior_2
};

struct TrajectoryNode {
    float4 position_phase;  // x, y, z, phase
    float4 velocity_energy; // vx, vy, vz, kinetic_energy
};

struct BakeMetaData {
    uint32_t node_id;
    uint32_t ring_buffer_offset;
    uint32_t frame_count;
    bool is_phase_locked;
};

// CPU VRAM 캐시 페이지 테이블 관리자
struct CausalPageTable {
    CausalPageEntry* entries;
    uint32_t capacity;
    uint32_t active_count;

    // O(1) 공간 해시 탐색 함수
    inline CausalPageEntry* LookupEntry(uint64_t hashKey) {
        if (!entries || capacity == 0) return nullptr;
        uint32_t slot = hashKey % capacity;
        uint32_t startSlot = slot;
        // Linear Probing으로 충돌 해쇄
        while (entries[slot].spatial_hash_key != 0) {
            if (entries[slot].spatial_hash_key == hashKey) {
                return &entries[slot];
            }
            slot = (slot + 1) % capacity;
            if (slot == startSlot) break;
        }
        return nullptr;
    }
};

#endif // ELYSIA_CAUSAL_LUT_HEADER_H
