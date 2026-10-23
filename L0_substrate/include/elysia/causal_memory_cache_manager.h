#ifndef ELYSIA_CAUSAL_MEMORY_CACHE_MANAGER_H
#define ELYSIA_CAUSAL_MEMORY_CACHE_MANAGER_H

#include <vector>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "elysia/causal_lut_header.h"
#include "elysia/d3d12_causal_page_allocator.h"

class CausalMemoryCacheManager {
private:
    CausalPageTable            m_pageTable;
    D3D12CausalPageAllocator*  m_pageAllocator;

    uint32_t                   m_maxVramPages;     // Hot-Cache 최대 슬롯 수
    uint32_t                   m_currentTick = 0;  // LRU 관리용 타임스탬프

    // VRAM 슬롯 관리 테이블 (Slot Index -> Entry Pointer)
    std::vector<CausalPageEntry*> m_vramSlotTable;

public:
    CausalMemoryCacheManager(D3D12CausalPageAllocator* allocator, uint32_t maxPages, uint32_t hashCapacity);
    ~CausalMemoryCacheManager();

    CausalPageTable& GetPageTable() { return m_pageTable; }

    uint64_t FetchOrPageInCausalPage(uint64_t spatialHashKey);

    // Helper to register new entry in Page Table
    void RegisterEntry(uint64_t spatialHashKey, uint64_t nvmeSectorOffset, uint32_t payloadSizeBytes, bool isAttractor = false);

private:
    int FindFreeVramSlot();
    int EvictLRUPage();
};

#endif // ELYSIA_CAUSAL_MEMORY_CACHE_MANAGER_H
