#ifndef ELYSIA_SENSORY_TENSOR_ENCODER_H
#define ELYSIA_SENSORY_TENSOR_ENCODER_H

#include <cstdint>
#include "elysia/causal_lut_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void LaunchInjectSensoryTensor(
    const UnifiedSensoryFrame* rawSensoryStream,
    float4* spatialTensorMap,
    uint32_t streamCount,
    uint32_t hashSize,
    void* stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_SENSORY_TENSOR_ENCODER_H
