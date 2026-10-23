#ifndef ELYSIA_MULTI_GPU_NVLINK_ROUTER_H
#define ELYSIA_MULTI_GPU_NVLINK_ROUTER_H

#include <cstdint>
#include <vector>

struct DistributedPageLocation {
    int      gpuId;               // 해당 인과 페이지가 물리적으로 할당된 GPU ID
    void*    vramDevicePointer;   // 해당 GPU 내 VRAM 가상 주소
    uint32_t usageFrequency;      // global 접근 빈도
};

class MultiGPUNVLinkCacheRouter {
private:
    int m_deviceCount;
    std::vector<bool> m_p2pMatrix;

public:
    MultiGPUNVLinkCacheRouter();
    ~MultiGPUNVLinkCacheRouter() = default;

    int GetDeviceCount() const { return m_deviceCount; }

    void FetchCausalTensorP2P(
        int targetGpuId,
        const DistributedPageLocation& loc,
        float* d_destTensorBuffer,
        uint32_t tensorSizeBytes,
        void* stream = nullptr
    );

    void BroadcastCodebook(
        const float* masterCodebook,
        float* slaveCodebook,
        uint32_t numElements,
        void* stream = nullptr
    );
};

#endif // ELYSIA_MULTI_GPU_NVLINK_ROUTER_H
