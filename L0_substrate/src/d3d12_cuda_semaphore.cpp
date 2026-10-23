#include "elysia/d3d12_cuda_semaphore.h"
#include <iostream>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <cuda_d3d12.h>
#endif
#endif

D3D12CUDASemaphoreSync::D3D12CUDASemaphoreSync(ID3D12Device* d3d12Device) {
#if defined(_WIN32) && defined(__CUDACC__)
    if (d3d12Device) {
        if (SUCCEEDED(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_d3d12Fence)))) {
            d3d12Device->CreateSharedHandle(m_d3d12Fence.Get(), nullptr, GENERIC_ALL, nullptr, &m_sharedFenceHandle);

            cudaExternalSemaphoreHandleDesc semHandleDesc = {};
            semHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
            semHandleDesc.handle.win32.handle = m_sharedFenceHandle;
            semHandleDesc.flags = 0;

            cudaExternalSemaphore_t sem = nullptr;
            if (cudaImportExternalSemaphore(&sem, &semHandleDesc) == cudaSuccess) {
                m_cudaExtSemaphore = sem;
            }
        }
    }
#else
    (void)d3d12Device;
#endif
}

D3D12CUDASemaphoreSync::~D3D12CUDASemaphoreSync() {
#if defined(_WIN32) && defined(__CUDACC__)
    if (m_cudaExtSemaphore) {
        cudaDestroyExternalSemaphore((cudaExternalSemaphore_t)m_cudaExtSemaphore);
    }
    if (m_sharedFenceHandle) {
        CloseHandle(m_sharedFenceHandle);
    }
#endif
}

uint64_t D3D12CUDASemaphoreSync::GetNextFenceSignalValue(ID3D12Fence** ppFenceOut) {
#if defined(_WIN32)
    if (ppFenceOut) *ppFenceOut = m_d3d12Fence.Get();
#else
    if (ppFenceOut) *ppFenceOut = nullptr;
#endif
    return ++m_currentFenceValue;
}

void D3D12CUDASemaphoreSync::SignalCUDAToWait(void* stream, uint64_t fenceValue) {
#if defined(_WIN32) && defined(__CUDACC__)
    if (m_cudaExtSemaphore) {
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.params.fence.value = fenceValue;
        waitParams.flags = 0;
        cudaStream_t custream = (cudaStream_t)stream;
        cudaExternalSemaphore_t sem = (cudaExternalSemaphore_t)m_cudaExtSemaphore;
        cudaWaitExternalSemaphoresAsync(&sem, &waitParams, 1, custream);
        return;
    }
#endif
    (void)stream;
    (void)fenceValue;
}
