#ifndef ELYSIA_VRAM_SLAB_ALLOCATOR_H
#define ELYSIA_VRAM_SLAB_ALLOCATOR_H

#include <vector>
#include <cstdint>
#include <stdexcept>

// VRAM 고정 크기 블록(Slab) 할당자
class VRAMSlabAllocator {
private:
    uint8_t*  m_vramBaseAddress;   // VRAM 거대 Pool의 시작 GPU/가상 주소
    uint32_t  m_slabSizeBytes;     // 슬랩 1개의 고정 크기 (예: 64KB)
    uint32_t  m_totalSlabs;        // 전체 슬랩 개수

    std::vector<uint32_t> m_freeSlabStack; // O(1) 할당/해제를 위한 Free-List 스택

public:
    VRAMSlabAllocator(void* vramBasePtr, uint32_t totalSlabs, uint32_t slabSizeBytes)
        : m_vramBaseAddress(static_cast<uint8_t*>(vramBasePtr)),
          m_totalSlabs(totalSlabs),
          m_slabSizeBytes(slabSizeBytes)
    {
        m_freeSlabStack.reserve(totalSlabs);
        // 초기화 시 모든 슬랩 인덱스를 Free 스택에 푸시
        for (int32_t i = static_cast<int32_t>(totalSlabs) - 1; i >= 0; --i) {
            m_freeSlabStack.push_back(static_cast<uint32_t>(i));
        }
    }

    // O(1) 시간 복잡도의 VRAM 슬랩 할당 (드라이버 API 호출 0)
    uint8_t* AllocateSlab(uint32_t& outSlabIndex) {
        if (m_freeSlabStack.empty()) {
            return nullptr; // VRAM Pool 꽉 참 (Eviction 필요)
        }

        outSlabIndex = m_freeSlabStack.back();
        m_freeSlabStack.pop_back();

        // Base 주소에서 오프셋 계산하여 반환 (단편화 발생 불가능)
        return m_vramBaseAddress + (outSlabIndex * m_slabSizeBytes);
    }

    // O(1) 시간 복잡도의 VRAM 슬랩 반환
    void FreeSlab(uint32_t slabIndex) {
        if (slabIndex >= m_totalSlabs) {
            throw std::out_of_range("Invalid Slab Index returned.");
        }
        m_freeSlabStack.push_back(slabIndex);
    }

    // 현재 남은 가용 슬랩 수
    uint32_t GetAvailableSlabs() const {
        return static_cast<uint32_t>(m_freeSlabStack.size());
    }

    uint32_t GetSlabSizeBytes() const {
        return m_slabSizeBytes;
    }

    uint32_t GetTotalSlabs() const {
        return m_totalSlabs;
    }
};

#endif // ELYSIA_VRAM_SLAB_ALLOCATOR_H
