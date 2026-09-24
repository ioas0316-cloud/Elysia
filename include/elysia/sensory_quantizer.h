#ifndef ELYSIA_SENSORY_QUANTIZER_H
#define ELYSIA_SENSORY_QUANTIZER_H

#include <cstdint>

#define CODEBOOK_SIZE 256
#define VECTOR_DIM 16

#ifdef __cplusplus
extern "C" {
#endif

void LaunchSensoryQuantizer(
    const float* d_spatialTensorMap,
    uint8_t* d_compressedIndices,
    const float* h_codebookCentroids,
    uint32_t numNodes,
    void* stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_SENSORY_QUANTIZER_H
