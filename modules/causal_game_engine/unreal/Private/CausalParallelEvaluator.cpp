// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#include "CausalParallelEvaluator.h"
#include "Async/ParallelFor.h"

void FCausalParallelEvaluator::ParallelEvaluateTension(TArray<FCCNodeRawData>& NodeArray, float CriticalLimit)
{
    const int32 TotalNodes = NodeArray.Num();
    if (TotalNodes == 0) return;

    // Parallel partitioning across available CPU threads via Task Graph
    ParallelFor(TotalNodes, [&](int32 Index)
    {
        FCCNodeRawData& Node = NodeArray[Index];
        if (Node.bIsQuarantined) return;

        // V_t causal tension decay / relaxation operation
        Node.TensionField *= 0.98f;

        // Critical threshold check for causal rupture & quarantine
        if (Node.TensionField > CriticalLimit)
        {
            Node.bIsQuarantined = true;
            // Lock-free atomic push to SealedAttractor queue can be dispatched here
        }
    }, EParallelForFlags::BackgroundPriority);
}
