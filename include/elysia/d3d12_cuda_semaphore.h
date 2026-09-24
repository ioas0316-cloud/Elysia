#ifndef ELYSIA_D3D12_CUDA_SEMAPHORE_H
#define ELYSIA_D3D12_CUDA_SEMAPHORE_H

#include <cstdint>
#include "elysia/d3d12_shims.h"

class D3D12CUDASemaphoreSync {
private:
    HANDLE                  m_sharedFenceHandle = nullptr;
    void*                   m_cudaExtSemaphore = nullptr;
    uint64_t                m_currentFenceValue = 0;

#if defined(_WIN32)
    ComPtr<ID3D12Fence>     m_d3d12Fence;
#endif

public:
    D3D12CUDASemaphoreSync(ID3D12Device* d3d12Device);
    ~D3D12CUDASemaphoreSync();

    uint64_t GetNextFenceSignalValue(ID3D12Fence** ppFenceOut);
    void SignalCUDAToWait(void* stream, uint64_t fenceValue);
};

#endif // ELYSIA_D3D12_CUDA_SEMAPHORE_H
