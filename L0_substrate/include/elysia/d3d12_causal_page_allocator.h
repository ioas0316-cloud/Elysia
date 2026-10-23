#ifndef ELYSIA_D3D12_CAUSAL_PAGE_ALLOCATOR_H
#define ELYSIA_D3D12_CAUSAL_PAGE_ALLOCATOR_H

#include <cstdint>
#include <string>
#include <vector>
#include "elysia/causal_lut_header.h"
#include "elysia/d3d12_shims.h"

class D3D12CausalPageAllocator {
private:
    ID3D12Device* m_d3d12Device;
    uint64_t      m_pageSizeBytes;
    uint64_t      m_poolTotalSize;
    uint32_t      m_maxPages;

    HANDLE        m_ssdFileHandle;
    uint64_t      m_fenceValue = 0;

    std::vector<uint8_t> m_simulatedVramPool; // Fallback VRAM buffer for host/Linux

#if defined(_WIN32)
    ComPtr<ID3D12Resource>   m_vramHotCachePool;
    ComPtr<IDStorageFactory> m_dsFactory;
    ComPtr<IDStorageQueue1>  m_dsQueue;
    ComPtr<ID3D12Fence>      m_dsFence;
#endif

public:
    D3D12CausalPageAllocator(ID3D12Device* device, uint32_t maxPages, uint64_t pageSizeBytes);
    ~D3D12CausalPageAllocator();

    void OpenCausalStorageFile(const wchar_t* filePath);
    void OpenCausalStorageFile(const char* filePath);
    void PageInAsync(CausalPageEntry* entry, uint32_t poolSlotIndex);
    bool IsPagingComplete(uint64_t fenceValue) const;

    uint8_t* GetSimulatedVramBase() { return m_simulatedVramPool.data(); }
    uint64_t GetPageSizeBytes() const { return m_pageSizeBytes; }
};

#endif // ELYSIA_D3D12_CAUSAL_PAGE_ALLOCATOR_H
