#ifndef ELYSIA_D3D12_CUDA_INTEROP_H
#define ELYSIA_D3D12_CUDA_INTEROP_H

#include <cstdint>
#include <vector>
#include "elysia/d3d12_shims.h"

class D3D12CUDAInteropBuffer {
private:
    ID3D12Device*          m_d3d12Device = nullptr;
    HANDLE                 m_sharedNtHandle = nullptr;
    void*                  m_cudaExtMemHandle = nullptr;
    void*                  m_d_cudaMappedPtr = nullptr;
    uint64_t               m_bufferSizeBytes = 0;

    std::vector<uint8_t>   m_fallbackBuffer;

#if defined(_WIN32)
    ComPtr<ID3D12Resource> m_d3d12Resource;
#endif

public:
    D3D12CUDAInteropBuffer(ID3D12Device* d3d12Device, uint64_t sizeBytes);
    ~D3D12CUDAInteropBuffer();

    ID3D12Resource* GetD3D12Resource() const;
    void* GetCUDADevicePointer() const;
    uint64_t GetBufferSizeBytes() const { return m_bufferSizeBytes; }
};

#endif // ELYSIA_D3D12_CUDA_INTEROP_H
