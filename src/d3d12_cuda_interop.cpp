#include "elysia/d3d12_cuda_interop.h"
#include <iostream>
#include <stdexcept>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <cuda_d3d12.h>
#endif
#endif

D3D12CUDAInteropBuffer::D3D12CUDAInteropBuffer(ID3D12Device* d3d12Device, uint64_t sizeBytes)
    : m_d3d12Device(d3d12Device), m_bufferSizeBytes(sizeBytes)
{
    m_fallbackBuffer.resize(sizeBytes, 0);

#if defined(_WIN32) && defined(__CUDACC__)
    if (d3d12Device) {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC resourceDesc = {};
        resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resourceDesc.Width = sizeBytes;
        resourceDesc.Height = 1;
        resourceDesc.DepthOrArraySize = 1;
        resourceDesc.MipLevels = 1;
        resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
        resourceDesc.SampleDesc.Count = 1;
        resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        if (SUCCEEDED(d3d12Device->CreateCommittedResource(
                &heapProps, D3D12_HEAP_FLAG_SHARED, &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_d3d12Resource)))) {

            d3d12Device->CreateSharedHandle(m_d3d12Resource.Get(), nullptr, GENERIC_ALL, nullptr, IID_PPV_ARGS(&m_sharedNtHandle));

            cudaExternalMemoryHandleDesc extMemHandleDesc = {};
            extMemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
            extMemHandleDesc.handle.win32.handle = m_sharedNtHandle;
            extMemHandleDesc.size = sizeBytes;
            extMemHandleDesc.flags = cudaExternalMemoryDedicated;

            cudaExternalMemory_t extMem = nullptr;
            if (cudaImportExternalMemory(&extMem, &extMemHandleDesc) == cudaSuccess) {
                m_cudaExtMemHandle = extMem;
                cudaExternalMemoryBufferDesc bufferDesc = {};
                bufferDesc.offset = 0;
                bufferDesc.size = sizeBytes;
                bufferDesc.flags = 0;
                cudaExternalMemoryGetMappedBuffer(&m_d_cudaMappedPtr, extMem, &bufferDesc);
            }
        }
    }
#endif

    if (!m_d_cudaMappedPtr) {
        m_d_cudaMappedPtr = m_fallbackBuffer.data();
    }
}

D3D12CUDAInteropBuffer::~D3D12CUDAInteropBuffer() {
#if defined(_WIN32) && defined(__CUDACC__)
    if (m_d_cudaMappedPtr && m_d_cudaMappedPtr != m_fallbackBuffer.data()) {
        cudaFree(m_d_cudaMappedPtr);
    }
    if (m_cudaExtMemHandle) {
        cudaDestroyExternalMemory((cudaExternalMemory_t)m_cudaExtMemHandle);
    }
    if (m_sharedNtHandle) {
        CloseHandle(m_sharedNtHandle);
    }
#endif
}

ID3D12Resource* D3D12CUDAInteropBuffer::GetD3D12Resource() const {
#if defined(_WIN32)
    return m_d3d12Resource.Get();
#else
    return nullptr;
#endif
}

void* D3D12CUDAInteropBuffer::GetCUDADevicePointer() const {
    return m_d_cudaMappedPtr;
}
