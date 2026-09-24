#include "elysia/sensory_tensor_encoder.h"
#include <cmath>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline uint32_t ComputeSpatialHashKey(int x, int y, int z, int timeSlot, uint32_t hashSize) {
    const uint32_t p1 = 73856093;
    const uint32_t p2 = 19349663;
    const uint32_t p3 = 83492791;
    const uint32_t p4 = 25165843;
    return ((x * p1) ^ (y * p2) ^ (z * p3) ^ (timeSlot * p4)) % hashSize;
}

__global__ void InjectSensoryTensorKernel(
    const UnifiedSensoryFrame* __restrict__ rawSensoryStream,
    float4* __restrict__ spatialTensorMap,
    uint32_t streamCount,
    uint32_t hashSize)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= streamCount) return;

    UnifiedSensoryFrame frame = rawSensoryStream[idx];

    int gridX = __float2int_rn(frame.visual_depth.w * 100.0f);
    int gridY = __float2int_rn(frame.physical_force.x * 10.0f);
    int gridZ = __float2int_rn(frame.physical_force.y * 10.0f);
    int timeSlot = __float2int_rn(frame.acoustic_phase.z * 10.0f);

    uint32_t hashKey = ComputeSpatialHashKey(gridX, gridY, gridZ, timeSlot, hashSize);

    uint32_t baseSlot = hashKey * 4; // float4 * 4 = 16 Channels

    atomicAdd(&spatialTensorMap[baseSlot + 0].x, frame.visual_depth.x);
    atomicAdd(&spatialTensorMap[baseSlot + 0].y, frame.visual_depth.y);
    atomicAdd(&spatialTensorMap[baseSlot + 0].z, frame.visual_depth.z);
    atomicAdd(&spatialTensorMap[baseSlot + 0].w, frame.visual_depth.w);

    atomicAdd(&spatialTensorMap[baseSlot + 1].x, frame.acoustic_phase.x);
    atomicAdd(&spatialTensorMap[baseSlot + 1].y, frame.acoustic_phase.y);
    atomicAdd(&spatialTensorMap[baseSlot + 1].z, frame.acoustic_phase.z);
    atomicAdd(&spatialTensorMap[baseSlot + 1].w, frame.acoustic_phase.w);

    atomicAdd(&spatialTensorMap[baseSlot + 2].x, frame.physical_force.x);
    atomicAdd(&spatialTensorMap[baseSlot + 2].y, frame.physical_force.y);
    atomicAdd(&spatialTensorMap[baseSlot + 2].z, frame.physical_force.z);
    atomicAdd(&spatialTensorMap[baseSlot + 2].w, frame.physical_force.w);

    atomicAdd(&spatialTensorMap[baseSlot + 3].x, frame.contextual_meta.x);
    atomicAdd(&spatialTensorMap[baseSlot + 3].y, frame.contextual_meta.y);
    atomicAdd(&spatialTensorMap[baseSlot + 3].z, frame.contextual_meta.z);
    atomicAdd(&spatialTensorMap[baseSlot + 3].w, frame.contextual_meta.w);
}
#else
static inline uint32_t ComputeSpatialHashKeyHost(int x, int y, int z, int timeSlot, uint32_t hashSize) {
    const uint32_t p1 = 73856093;
    const uint32_t p2 = 19349663;
    const uint32_t p3 = 83492791;
    const uint32_t p4 = 25165843;
    return ((x * p1) ^ (y * p2) ^ (z * p3) ^ (timeSlot * p4)) % hashSize;
}
#endif

extern "C" void LaunchInjectSensoryTensor(
    const UnifiedSensoryFrame* rawSensoryStream,
    float4* spatialTensorMap,
    uint32_t streamCount,
    uint32_t hashSize,
    void* stream)
{
#if defined(__CUDACC__)
    int blockSize = 256;
    int gridSize = (streamCount + blockSize - 1) / blockSize;
    cudaStream_t custream = (cudaStream_t)stream;
    InjectSensoryTensorKernel<<<gridSize, blockSize, 0, custream>>>(
        rawSensoryStream, spatialTensorMap, streamCount, hashSize
    );
#else
    (void)stream;
    for (uint32_t idx = 0; idx < streamCount; ++idx) {
        UnifiedSensoryFrame frame = rawSensoryStream[idx];

        int gridX = static_cast<int>(std::round(frame.visual_depth.w * 100.0f));
        int gridY = static_cast<int>(std::round(frame.physical_force.x * 10.0f));
        int gridZ = static_cast<int>(std::round(frame.physical_force.y * 10.0f));
        int timeSlot = static_cast<int>(std::round(frame.acoustic_phase.z * 10.0f));

        uint32_t hashKey = ComputeSpatialHashKeyHost(gridX, gridY, gridZ, timeSlot, hashSize);
        uint32_t baseSlot = hashKey * 4;

        spatialTensorMap[baseSlot + 0].x += frame.visual_depth.x;
        spatialTensorMap[baseSlot + 0].y += frame.visual_depth.y;
        spatialTensorMap[baseSlot + 0].z += frame.visual_depth.z;
        spatialTensorMap[baseSlot + 0].w += frame.visual_depth.w;

        spatialTensorMap[baseSlot + 1].x += frame.acoustic_phase.x;
        spatialTensorMap[baseSlot + 1].y += frame.acoustic_phase.y;
        spatialTensorMap[baseSlot + 1].z += frame.acoustic_phase.z;
        spatialTensorMap[baseSlot + 1].w += frame.acoustic_phase.w;

        spatialTensorMap[baseSlot + 2].x += frame.physical_force.x;
        spatialTensorMap[baseSlot + 2].y += frame.physical_force.y;
        spatialTensorMap[baseSlot + 2].z += frame.physical_force.z;
        spatialTensorMap[baseSlot + 2].w += frame.physical_force.w;

        spatialTensorMap[baseSlot + 3].x += frame.contextual_meta.x;
        spatialTensorMap[baseSlot + 3].y += frame.contextual_meta.y;
        spatialTensorMap[baseSlot + 3].z += frame.contextual_meta.z;
        spatialTensorMap[baseSlot + 3].w += frame.contextual_meta.w;
    }
#endif
}
