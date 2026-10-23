#include "elysia/elysian_engine_main_loop.h"
#include "elysia/sensory_quantizer.h"
#include "elysia/codebook_online_trainer.h"
#include <iostream>
#include <vector>
#include <cstring>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>
#endif

void ElysianEngineLoop::Init(ID3D12Device* d3d12Device, const wchar_t* ssdFilePath) {
    m_dsAllocator = new D3D12CausalPageAllocator(d3d12Device, m_maxPages, m_pageSizeBytes);
    if (ssdFilePath) {
        m_dsAllocator->OpenCausalStorageFile(ssdFilePath);
    }

    m_vramInteropBuffer = new D3D12CUDAInteropBuffer(d3d12Device, m_maxPages * m_pageSizeBytes);
    m_gpuSyncSemaphore = new D3D12CUDASemaphoreSync(d3d12Device);

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    m_cudaComputeStream = (void*)stream;

    cudaMalloc(&m_d_masterCodebook, sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM);
    cudaMalloc(&m_d_clusterSums, sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM);
    cudaMalloc(&m_d_clusterCounts, sizeof(uint32_t) * CODEBOOK_SIZE);
    cudaMemset(m_d_masterCodebook, 0, sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM);
#else
    m_d_masterCodebook = new float[CODEBOOK_SIZE * VECTOR_DIM]();
    m_d_clusterSums = new float[CODEBOOK_SIZE * VECTOR_DIM]();
    m_d_clusterCounts = new uint32_t[CODEBOOK_SIZE]();
#endif

    m_cacheManager = new CausalMemoryCacheManager(m_dsAllocator, m_maxPages, 4096);
    m_slabAllocator = new VRAMSlabAllocator(m_vramInteropBuffer->GetCUDADevicePointer(), m_maxPages, static_cast<uint32_t>(m_pageSizeBytes));
}

ElysianEngineLoop::ElysianEngineLoop(ID3D12Device* d3d12Device, const wchar_t* ssdFilePath) {
    Init(d3d12Device, ssdFilePath);
}

ElysianEngineLoop::ElysianEngineLoop(ID3D12Device* d3d12Device, const char* ssdFilePath) {
#if defined(_WIN32)
    wchar_t wpath[1024];
    mbstowcs(wpath, ssdFilePath, 1024);
    Init(d3d12Device, wpath);
#else
    (void)ssdFilePath;
    Init(d3d12Device, nullptr);
#endif
}

ElysianEngineLoop::~ElysianEngineLoop() {
    delete m_cacheManager;
    delete m_slabAllocator;
    delete m_gpuSyncSemaphore;
    delete m_vramInteropBuffer;
    delete m_dsAllocator;

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
    if (m_cudaComputeStream) {
        cudaStreamDestroy((cudaStream_t)m_cudaComputeStream);
    }
    if (m_d_masterCodebook) cudaFree(m_d_masterCodebook);
    if (m_d_clusterSums) cudaFree(m_d_clusterSums);
    if (m_d_clusterCounts) cudaFree(m_d_clusterCounts);
#else
    delete[] m_d_masterCodebook;
    delete[] m_d_clusterSums;
    delete[] m_d_clusterCounts;
#endif
}

void ElysianEngineLoop::RegisterCausalPage(uint64_t hashKey, uint64_t offset, uint32_t size, bool isAttractor) {
    if (m_cacheManager) {
        m_cacheManager->RegisterEntry(hashKey, offset, size, isAttractor);
    }
}

void ElysianEngineLoop::Tick(uint64_t targetSpatialHashKey, float* d_sensoryTensorBatch, uint32_t batchSize) {
    uint64_t vramAddress = m_cacheManager->FetchOrPageInCausalPage(targetSpatialHashKey);
    (void)vramAddress;

    ID3D12Fence* d3d12Fence = nullptr;
    uint64_t signalValue = m_gpuSyncSemaphore->GetNextFenceSignalValue(&d3d12Fence);
    (void)signalValue;

    m_gpuSyncSemaphore->SignalCUDAToWait(m_cudaComputeStream, signalValue);

    std::vector<uint8_t> compressedIndices(batchSize, 0);

    LaunchSensoryQuantizer(
        d_sensoryTensorBatch,
        compressedIndices.data(),
        m_d_masterCodebook,
        batchSize,
        m_cudaComputeStream
    );

    ExecuteOnlineCodebookUpdate(
        d_sensoryTensorBatch,
        compressedIndices.data(),
        m_d_masterCodebook,
        m_d_clusterSums,
        m_d_clusterCounts,
        batchSize,
        0.05f,
        m_cudaComputeStream
    );
}
