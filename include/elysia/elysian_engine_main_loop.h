#ifndef ELYSIAN_ENGINE_MAIN_LOOP_H
#define ELYSIAN_ENGINE_MAIN_LOOP_H

#include <cstdint>
#include "elysia/d3d12_causal_page_allocator.h"
#include "elysia/d3d12_cuda_interop.h"
#include "elysia/d3d12_cuda_semaphore.h"
#include "elysia/causal_memory_cache_manager.h"
#include "elysia/vram_slab_allocator.h"

class ElysianEngineLoop {
private:
    D3D12CausalPageAllocator*   m_dsAllocator = nullptr;
    D3D12CUDAInteropBuffer*     m_vramInteropBuffer = nullptr;
    D3D12CUDASemaphoreSync*     m_gpuSyncSemaphore = nullptr;
    CausalMemoryCacheManager*   m_cacheManager = nullptr;
    VRAMSlabAllocator*          m_slabAllocator = nullptr;

    void*                       m_cudaComputeStream = nullptr;
    uint32_t                    m_maxPages = 1024;
    uint64_t                    m_pageSizeBytes = 64 * 1024; // 64KB Causal Page

    float*                      m_d_masterCodebook = nullptr;
    float*                      m_d_clusterSums = nullptr;
    uint32_t*                   m_d_clusterCounts = nullptr;

    void Init(ID3D12Device* d3d12Device, const wchar_t* ssdFilePath);

public:
    ElysianEngineLoop(ID3D12Device* d3d12Device, const wchar_t* ssdFilePath);
    ElysianEngineLoop(ID3D12Device* d3d12Device, const char* ssdFilePath);
    ~ElysianEngineLoop();

    CausalMemoryCacheManager* GetCacheManager() { return m_cacheManager; }
    D3D12CausalPageAllocator* GetAllocator() { return m_dsAllocator; }
    VRAMSlabAllocator* GetSlabAllocator() { return m_slabAllocator; }

    void RegisterCausalPage(uint64_t hashKey, uint64_t offset, uint32_t size, bool isAttractor = false);

    void Tick(uint64_t targetSpatialHashKey, float* d_sensoryTensorBatch, uint32_t batchSize);
};

#endif // ELYSIAN_ENGINE_MAIN_LOOP_H
