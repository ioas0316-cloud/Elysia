#ifndef ELYSIA_PHASE_LOCK_EVALUATOR_H
#define ELYSIA_PHASE_LOCK_EVALUATOR_H

#include <cstdint>
#include "elysia/causal_lut_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void LaunchPhaseLockEvaluation(
    const TrajectoryNode* ringBuffer,
    BakeMetaData* bakeData,
    float energyThreshold,
    float varianceThreshold,
    uint32_t nodeCount,
    uint32_t windowSize,
    void* stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_PHASE_LOCK_EVALUATOR_H
