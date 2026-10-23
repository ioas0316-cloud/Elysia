#include "elysia/phase_lock_evaluator.h"
#include <cmath>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void EvaluatePhaseLockKernel(
    const TrajectoryNode* __restrict__ ringBuffer,
    BakeMetaData* __restrict__ bakeData,
    float energyThreshold,
    float varianceThreshold,
    uint32_t nodeCount,
    uint32_t windowSize)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;

    float totalEnergy = 0.0f;
    float4 meanVel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t f = 0; f < windowSize; ++f) {
        uint32_t slot = (idx * windowSize) + f;
        TrajectoryNode node = ringBuffer[slot];

        totalEnergy += node.velocity_energy.w;
        meanVel.x += node.velocity_energy.x;
        meanVel.y += node.velocity_energy.y;
        meanVel.z += node.velocity_energy.z;
    }

    float avgEnergy = totalEnergy / (float)windowSize;

    float variance = 0.0f;
    for (uint32_t f = 0; f < windowSize; ++f) {
        uint32_t slot = (idx * windowSize) + f;
        float4 vel = ringBuffer[slot].velocity_energy;
        float diffX = vel.x - (meanVel.x / (float)windowSize);
        float diffY = vel.y - (meanVel.y / (float)windowSize);
        float diffZ = vel.z - (meanVel.z / (float)windowSize);
        variance += (diffX*diffX + diffY*diffY + diffZ*diffZ);
    }
    variance /= (float)windowSize;

    if (avgEnergy < energyThreshold && variance < varianceThreshold) {
        bakeData[idx].is_phase_locked = true;
        bakeData[idx].frame_count = windowSize;
    } else {
        bakeData[idx].is_phase_locked = false;
    }
}
#endif

extern "C" void LaunchPhaseLockEvaluation(
    const TrajectoryNode* ringBuffer,
    BakeMetaData* bakeData,
    float energyThreshold,
    float varianceThreshold,
    uint32_t nodeCount,
    uint32_t windowSize,
    void* stream)
{
#if defined(__CUDACC__)
    int blockSize = 256;
    int gridSize = (nodeCount + blockSize - 1) / blockSize;
    cudaStream_t custream = (cudaStream_t)stream;
    EvaluatePhaseLockKernel<<<gridSize, blockSize, 0, custream>>>(
        ringBuffer, bakeData, energyThreshold, varianceThreshold, nodeCount, windowSize
    );
#else
    (void)stream;
    for (uint32_t idx = 0; idx < nodeCount; ++idx) {
        float totalEnergy = 0.0f;
        float meanVelX = 0.0f, meanVelY = 0.0f, meanVelZ = 0.0f;

        for (uint32_t f = 0; f < windowSize; ++f) {
            uint32_t slot = (idx * windowSize) + f;
            TrajectoryNode node = ringBuffer[slot];

            totalEnergy += node.velocity_energy.w;
            meanVelX += node.velocity_energy.x;
            meanVelY += node.velocity_energy.y;
            meanVelZ += node.velocity_energy.z;
        }

        float variance = 0.0f;
        for (uint32_t f = 0; f < windowSize; ++f) {
            uint32_t slot = (idx * windowSize) + f;
            float4 vel = ringBuffer[slot].velocity_energy;
            float diffX = vel.x - (meanVelX / (float)windowSize);
            float diffY = vel.y - (meanVelY / (float)windowSize);
            float diffZ = vel.z - (meanVelZ / (float)windowSize);
            variance += (diffX*diffX + diffY*diffY + diffZ*diffZ);
        }
        variance /= (float)windowSize;

        float avgEnergy = totalEnergy / (float)windowSize;
        if (avgEnergy < energyThreshold && variance < varianceThreshold) {
            bakeData[idx].is_phase_locked = true;
            bakeData[idx].frame_count = windowSize;
        } else {
            bakeData[idx].is_phase_locked = false;
        }
    }
#endif
}
