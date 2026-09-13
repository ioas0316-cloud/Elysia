// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * Raw blittable node data for cache-efficient Task Graph / ParallelFor iteration.
 */
struct CAUSALENGINE_API FCCNodeRawData
{
    FName NodeID;
    float TensionField = 0.0f;       // V_t
    float CriticalThreshold = 0.85f; // V_critical
    bool bIsQuarantined = false;
};

/**
 * FCausalParallelEvaluator
 *
 * Parallel evaluator leveraging UE5 Task Graph / ParallelFor CPU core partitioning
 * to process tens of thousands of CC-Nodes without blocking the main game thread.
 */
class CAUSALENGINE_API FCausalParallelEvaluator
{
public:
    /**
     * Executes parallel tension evaluation over continuous raw memory chunks.
     */
    static void ParallelEvaluateTension(TArray<FCCNodeRawData>& NodeArray, float CriticalLimit);
};
