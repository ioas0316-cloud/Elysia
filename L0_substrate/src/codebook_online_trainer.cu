#include "elysia/codebook_online_trainer.h"
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void AccumulateClusterCentroidsKernel(
    const float*   __restrict__ batchTensors,
    const uint8_t* __restrict__ assignedIndices,
    float*         __restrict__ clusterSums,
    uint32_t*      __restrict__ clusterCounts,
    uint32_t batchSize)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    uint8_t k = assignedIndices[idx];
    uint32_t tensorOffset = idx * VECTOR_DIM;

    atomicAdd(&clusterCounts[k], 1);

    uint32_t sumOffset = k * VECTOR_DIM;
    #pragma unroll
    for (int c = 0; c < VECTOR_DIM; ++c) {
        atomicAdd(&clusterSums[sumOffset + c], batchTensors[tensorOffset + c]);
    }
}

__global__ void UpdateCentroidsEMAKernel(
    float*          __restrict__ masterCodebook,
    const float*    __restrict__ clusterSums,
    const uint32_t* __restrict__ clusterCounts,
    float learningRate_eta)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= CODEBOOK_SIZE) return;

    uint32_t count = clusterCounts[k];
    uint32_t offset = k * VECTOR_DIM;

    if (count > 0) {
        float invCount = 1.0f / static_cast<float>(count);

        #pragma unroll
        for (int c = 0; c < VECTOR_DIM; ++c) {
            float batchMean = clusterSums[offset + c] * invCount;
            float currentCentroid = masterCodebook[offset + c];

            masterCodebook[offset + c] = (1.0f - learningRate_eta) * currentCentroid + (learningRate_eta * batchMean);
        }
    }
}
#endif

extern "C" void ExecuteOnlineCodebookUpdate(
    const float* d_batchTensors,
    const uint8_t* d_assignedIndices,
    float* d_masterCodebook,
    float* d_clusterSums,
    uint32_t* d_clusterCounts,
    uint32_t batchSize,
    float eta,
    void* stream)
{
#if defined(__CUDACC__)
    cudaStream_t custream = (cudaStream_t)stream;
    cudaMemsetAsync(d_clusterSums, 0, sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM, custream);
    cudaMemsetAsync(d_clusterCounts, 0, sizeof(uint32_t) * CODEBOOK_SIZE, custream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    AccumulateClusterCentroidsKernel<<<blocksPerGrid, threadsPerBlock, 0, custream>>>(
        d_batchTensors, d_assignedIndices, d_clusterSums, d_clusterCounts, batchSize
    );

    UpdateCentroidsEMAKernel<<<1, CODEBOOK_SIZE, 0, custream>>>(
        d_masterCodebook, d_clusterSums, d_clusterCounts, eta
    );
#else
    (void)stream;
    std::memset(d_clusterSums, 0, sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM);
    std::memset(d_clusterCounts, 0, sizeof(uint32_t) * CODEBOOK_SIZE);

    for (uint32_t idx = 0; idx < batchSize; ++idx) {
        uint8_t k = d_assignedIndices[idx];
        d_clusterCounts[k]++;
        uint32_t tensorOffset = idx * VECTOR_DIM;
        uint32_t sumOffset = k * VECTOR_DIM;
        for (int c = 0; c < VECTOR_DIM; ++c) {
            d_clusterSums[sumOffset + c] += d_batchTensors[tensorOffset + c];
        }
    }

    for (uint32_t k = 0; k < CODEBOOK_SIZE; ++k) {
        uint32_t count = d_clusterCounts[k];
        uint32_t offset = k * VECTOR_DIM;
        if (count > 0) {
            float invCount = 1.0f / static_cast<float>(count);
            for (int c = 0; c < VECTOR_DIM; ++c) {
                float batchMean = d_clusterSums[offset + c] * invCount;
                float currentCentroid = d_masterCodebook[offset + c];
                d_masterCodebook[offset + c] = (1.0f - eta) * currentCentroid + (eta * batchMean);
            }
        }
    }
#endif
}
