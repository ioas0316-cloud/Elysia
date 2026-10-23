#include "elysia/d3d12_causal_page_allocator.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#endif

D3D12CausalPageAllocator::D3D12CausalPageAllocator(ID3D12Device* device, uint32_t maxPages, uint64_t pageSizeBytes)
    : m_d3d12Device(device), m_pageSizeBytes(pageSizeBytes), m_maxPages(maxPages)
{
    m_poolTotalSize = maxPages * pageSizeBytes;
    m_simulatedVramPool.resize(m_poolTotalSize, 0);

#if defined(_WIN32)
    if (m_d3d12Device) {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = m_poolTotalSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        if (FAILED(m_d3d12Device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(&m_vramHotCachePool)))) {
            std::cerr << "Warning: Failed to allocate D3D12 VRAM Hot-Cache Pool, fallback active.\n";
        }

        DStorageGetFactory(IID_PPV_ARGS(&m_dsFactory));
        if (m_dsFactory) {
            DSTORAGE_QUEUE_DESC queueDesc = {};
            queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
            queueDesc.Priority = DSTORAGE_PRIORITY_HIGH;
            queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            queueDesc.Device = m_d3d12Device;

            m_dsFactory->CreateQueue(&queueDesc, IID_PPV_ARGS(&m_dsQueue));
            m_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_dsFence));
        }
    }
#endif
}

D3D12CausalPageAllocator::~D3D12CausalPageAllocator() {
#if defined(_WIN32)
    if (m_ssdFileHandle) {
        CloseHandle(m_ssdFileHandle);
        m_ssdFileHandle = nullptr;
    }
#endif
}

void D3D12CausalPageAllocator::OpenCausalStorageFile(const wchar_t* filePath) {
#if defined(_WIN32)
    if (m_dsFactory) {
        m_dsFactory->OpenFile(filePath, IID_PPV_ARGS(&m_ssdFileHandle));
    }
#else
    (void)filePath;
#endif
}

void D3D12CausalPageAllocator::OpenCausalStorageFile(const char* filePath) {
#if defined(_WIN32)
    wchar_t wpath[1024];
    mbstowcs(wpath, filePath, 1024);
    OpenCausalStorageFile(wpath);
#else
    (void)filePath;
#endif
}

void D3D12CausalPageAllocator::PageInAsync(CausalPageEntry* entry, uint32_t poolSlotIndex) {
    if (!entry || !entry->is_baked || entry->vram_pinned) return;

    uint64_t vramOffset = poolSlotIndex * m_pageSizeBytes;

#if defined(_WIN32)
    if (m_dsQueue && m_ssdFileHandle) {
        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;

        request.File.Source = m_ssdFileHandle;
        request.File.Offset = entry->nvme_sector_offset;
        request.File.Size   = entry->payload_size_bytes;

        request.Buffer.Resource = m_vramHotCachePool.Get();
        request.Buffer.Offset   = vramOffset;
        request.Buffer.Size     = entry->payload_size_bytes;

        m_dsQueue->EnqueueRequest(&request);

        m_fenceValue++;
        m_dsQueue->EnqueueSignal(m_dsFence.Get(), m_fenceValue);
        m_dsQueue->Submit();

        entry->vram_pinned = 1;
        entry->vram_page_address = m_vramHotCachePool->GetGPUVirtualAddress() + vramOffset;
        return;
    }
#endif

    // CPU / Host fallback
    entry->vram_pinned = 1;
    entry->vram_page_address = reinterpret_cast<uint64_t>(m_simulatedVramPool.data() + vramOffset);
}

bool D3D12CausalPageAllocator::IsPagingComplete(uint64_t fenceValue) const {
#if defined(_WIN32)
    if (m_dsFence) {
        return m_dsFence->GetCompletedValue() >= fenceValue;
    }
#endif
    (void)fenceValue;
    return true;
}
