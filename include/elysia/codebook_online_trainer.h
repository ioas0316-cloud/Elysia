#ifndef ELYSIA_CODEBOOK_ONLINE_TRAINER_H
#define ELYSIA_CODEBOOK_ONLINE_TRAINER_H

#include <cstdint>
#include "elysia/sensory_quantizer.h"

#ifdef __cplusplus
extern "C" {
#endif

void ExecuteOnlineCodebookUpdate(
    const float* d_batchTensors,
    const uint8_t* d_assignedIndices,
    float* d_masterCodebook,
    float* d_clusterSums,
    uint32_t* d_clusterCounts,
    uint32_t batchSize,
    float eta,
    void* stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_CODEBOOK_ONLINE_TRAINER_H
