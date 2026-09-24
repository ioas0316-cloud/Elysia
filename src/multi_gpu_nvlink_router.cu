#include "elysia/multi_gpu_nvlink_router.h"
#include <cstring>
#include <iostream>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void BroadcastCodebookKernel(
    const float* __restrict__ masterCodebook,
    float*       __restrict__ slaveCodebook,
    uint32_t numElements)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    slaveCodebook[idx] = masterCodebook[idx];
}
#endif

MultiGPUNVLinkCacheRouter::MultiGPUNVLinkCacheRouter() {
#if defined(__CUDACC__)
    cudaGetDeviceCount(&m_deviceCount);
    if (m_deviceCount <= 0) m_deviceCount = 1;
    m_p2pMatrix.resize(m_deviceCount * m_deviceCount, false);

    for (int i = 0; i < m_deviceCount; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < m_deviceCount; ++j) {
            if (i == j) continue;

            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess) {
                cudaDeviceEnablePeerAccess(j, 0);
                m_p2pMatrix[i * m_deviceCount + j] = true;
            }
        }
    }
#else
    m_deviceCount = 1;
    m_p2pMatrix.resize(1, false);
#endif
}

void MultiGPUNVLinkCacheRouter::FetchCausalTensorP2P(
    int targetGpuId,
    const DistributedPageLocation& loc,
    float* d_destTensorBuffer,
    uint32_t tensorSizeBytes,
    void* stream)
{
#if defined(__CUDACC__)
    cudaStream_t custream = (cudaStream_t)stream;
    if (targetGpuId == loc.gpuId) {
        cudaMemcpyAsync(
            d_destTensorBuffer,
            loc.vramDevicePointer,
            tensorSizeBytes,
            cudaMemcpyDeviceToDevice,
            custream
        );
    } else if (targetGpuId < m_deviceCount && loc.gpuId < m_deviceCount &&
               m_p2pMatrix[targetGpuId * m_deviceCount + loc.gpuId]) {
        cudaMemcpyPeerAsync(
            d_destTensorBuffer, targetGpuId,
            loc.vramDevicePointer, loc.gpuId,
            tensorSizeBytes,
            custream
        );
    } else {
        cudaMemcpyAsync(
            d_destTensorBuffer,
            loc.vramDevicePointer,
            tensorSizeBytes,
            cudaMemcpyDeviceToDevice,
            custream
        );
    }
#else
    (void)targetGpuId;
    (void)stream;
    std::memcpy(d_destTensorBuffer, loc.vramDevicePointer, tensorSizeBytes);
#endif
}

void MultiGPUNVLinkCacheRouter::BroadcastCodebook(
    const float* masterCodebook,
    float* slaveCodebook,
    uint32_t numElements,
    void* stream)
{
#if defined(__CUDACC__)
    cudaStream_t custream = (cudaStream_t)stream;
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    BroadcastCodebookKernel<<<gridSize, blockSize, 0, custream>>>(
        masterCodebook, slaveCodebook, numElements
    );
#else
    (void)stream;
    std::memcpy(slaveCodebook, masterCodebook, sizeof(float) * numElements);
#endif
}
