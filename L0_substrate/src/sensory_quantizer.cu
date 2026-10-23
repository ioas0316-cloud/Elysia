#include "elysia/sensory_quantizer.h"
#include <cstring>
#include <cmath>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ float c_CodebookCentroids[CODEBOOK_SIZE * VECTOR_DIM];

__global__ void QuantizeSensoryTensorKernel(
    const float* __restrict__ spatialTensorMap,
    uint8_t*     __restrict__ compressedIndices,
    uint32_t numNodes)
{
    uint32_t nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeIdx >= numNodes) return;

    float localTensor[VECTOR_DIM];
    uint32_t baseOffset = nodeIdx * VECTOR_DIM;

    #pragma unroll
    for (int c = 0; c < VECTOR_DIM; ++c) {
        localTensor[c] = spatialTensorMap[baseOffset + c];
    }

    float minDistance = 1e30f;
    uint8_t bestCodebookIdx = 0;

    for (int k = 0; k < CODEBOOK_SIZE; ++k) {
        float currentDist = 0.0f;
        uint32_t codebookOffset = k * VECTOR_DIM;

        #pragma unroll
        for (int c = 0; c < VECTOR_DIM; ++c) {
            float diff = localTensor[c] - c_CodebookCentroids[codebookOffset + c];
            currentDist += diff * diff;
        }

        if (currentDist < minDistance) {
            minDistance = currentDist;
            bestCodebookIdx = static_cast<uint8_t>(k);
        }
    }

    compressedIndices[nodeIdx] = bestCodebookIdx;
}
#endif

extern "C" void LaunchSensoryQuantizer(
    const float* d_spatialTensorMap,
    uint8_t* d_compressedIndices,
    const float* h_codebookCentroids,
    uint32_t numNodes,
    void* stream)
{
#if defined(__CUDACC__)
    cudaStream_t custream = (cudaStream_t)stream;
    cudaMemcpyToSymbolAsync(
        c_CodebookCentroids,
        h_codebookCentroids,
        sizeof(float) * CODEBOOK_SIZE * VECTOR_DIM,
        0,
        cudaMemcpyHostToDevice,
        custream
    );

    int blockSize = 256;
    int gridSize = (numNodes + blockSize - 1) / blockSize;

    QuantizeSensoryTensorKernel<<<gridSize, blockSize, 0, custream>>>(
        d_spatialTensorMap,
        d_compressedIndices,
        numNodes
    );
#else
    (void)stream;
    for (uint32_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        float minDistance = 1e30f;
        uint8_t bestCodebookIdx = 0;
        uint32_t baseOffset = nodeIdx * VECTOR_DIM;

        for (int k = 0; k < CODEBOOK_SIZE; ++k) {
            float currentDist = 0.0f;
            uint32_t codebookOffset = k * VECTOR_DIM;

            for (int c = 0; c < VECTOR_DIM; ++c) {
                float diff = d_spatialTensorMap[baseOffset + c] - h_codebookCentroids[codebookOffset + c];
                currentDist += diff * diff;
            }

            if (currentDist < minDistance) {
                minDistance = currentDist;
                bestCodebookIdx = static_cast<uint8_t>(k);
            }
        }
        d_compressedIndices[nodeIdx] = bestCodebookIdx;
    }
#endif
}
