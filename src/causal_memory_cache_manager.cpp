#include "elysia/causal_memory_cache_manager.h"
#include <climits>
#include <iostream>

CausalMemoryCacheManager::CausalMemoryCacheManager(D3D12CausalPageAllocator* allocator, uint32_t maxPages, uint32_t hashCapacity)
    : m_pageAllocator(allocator), m_maxVramPages(maxPages)
{
    m_pageTable.capacity = hashCapacity;
    m_pageTable.active_count = 0;
    m_pageTable.entries = new CausalPageEntry[hashCapacity]();

    m_vramSlotTable.resize(maxPages, nullptr);
}

CausalMemoryCacheManager::~CausalMemoryCacheManager() {
    delete[] m_pageTable.entries;
}

void CausalMemoryCacheManager::RegisterEntry(uint64_t spatialHashKey, uint64_t nvmeSectorOffset, uint32_t payloadSizeBytes, bool isAttractor) {
    if (!m_pageTable.entries || m_pageTable.capacity == 0) return;

    uint32_t slot = spatialHashKey % m_pageTable.capacity;
    uint32_t startSlot = slot;

    while (m_pageTable.entries[slot].spatial_hash_key != 0 && m_pageTable.entries[slot].spatial_hash_key != spatialHashKey) {
        slot = (slot + 1) % m_pageTable.capacity;
        if (slot == startSlot) throw std::runtime_error("CausalPageTable capacity full.");
    }

    CausalPageEntry& entry = m_pageTable.entries[slot];
    if (entry.spatial_hash_key == 0) {
        m_pageTable.active_count++;
    }

    entry.spatial_hash_key = spatialHashKey;
    entry.nvme_sector_offset = nvmeSectorOffset;
    entry.payload_size_bytes = payloadSizeBytes;
    entry.is_baked = 1;
    entry.is_attractor = isAttractor ? 1 : 0;
}

uint64_t CausalMemoryCacheManager::FetchOrPageInCausalPage(uint64_t spatialHashKey) {
    m_currentTick++;

    CausalPageEntry* entry = m_pageTable.LookupEntry(spatialHashKey);
    if (!entry) {
        throw std::runtime_error("Spatial Hash Key not found in SSD Page Table Index.");
    }

    // 1. [Cache Hit] 이미 VRAM Hot-Cache에 Pinned된 상태
    if (entry->vram_pinned && entry->vram_page_address != 0) {
        entry->last_access_tick = m_currentTick;
        entry->usage_frequency++;
        return entry->vram_page_address; // 즉시 O(1) 반환
    }

    // 2. [Cache Miss] VRAM으로 로드 필요 -> 가용 슬롯 탐색
    int targetSlot = FindFreeVramSlot();

    // 3. VRAM 용량이 가득 찬 경우: LRU Eviction (Page-Out) 실행
    if (targetSlot == -1) {
        targetSlot = EvictLRUPage();
    }

    // 4. 신규 페이지 Paging-In (DirectStorage Enqueue)
    m_vramSlotTable[targetSlot] = entry;
    entry->last_access_tick = m_currentTick;
    entry->usage_frequency++;

    m_pageAllocator->PageInAsync(entry, static_cast<uint32_t>(targetSlot));

    return entry->vram_page_address;
}

int CausalMemoryCacheManager::FindFreeVramSlot() {
    for (size_t i = 0; i < m_vramSlotTable.size(); ++i) {
        if (m_vramSlotTable[i] == nullptr) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int CausalMemoryCacheManager::EvictLRUPage() {
    uint32_t minTick = UINT32_MAX;
    int lruSlot = -1;

    for (size_t i = 0; i < m_vramSlotTable.size(); ++i) {
        CausalPageEntry* candidate = m_vramSlotTable[i];
        if (candidate && candidate->vram_pinned) {
            if (candidate->is_attractor && candidate->usage_frequency > 1000) {
                continue;
            }

            if (candidate->last_access_tick < minTick) {
                minTick = candidate->last_access_tick;
                lruSlot = static_cast<int>(i);
            }
        }
    }

    if (lruSlot == -1) {
        lruSlot = 0;
    }

    CausalPageEntry* evictEntry = m_vramSlotTable[lruSlot];
    if (evictEntry) {
        evictEntry->vram_pinned = 0;
        evictEntry->vram_page_address = 0;
    }

    m_vramSlotTable[lruSlot] = nullptr;
    return lruSlot;
}
