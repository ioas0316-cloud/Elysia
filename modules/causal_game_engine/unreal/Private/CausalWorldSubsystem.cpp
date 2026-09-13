// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#include "CausalWorldSubsystem.h"
#include "CausalParallelEvaluator.h"
#include "GameFramework/Actor.h"

void UCausalWorldSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    CurrentWorldTension = 0.0f;
    ActiveCCNodes.Empty();
}

void UCausalWorldSubsystem::Deinitialize()
{
    ActiveCCNodes.Empty();
    Super::Deinitialize();
}

void UCausalWorldSubsystem::RegisterPlayerAction(FName ActionType, AActor* TargetActor)
{
    if (!TargetActor) return;

    FName TargetID = TargetActor->GetFName();
    FCCGameNode* NodePtr = ActiveCCNodes.Find(TargetID);

    if (NodePtr)
    {
        // Increase node tension based on player interaction
        NodePtr->TensionField += 0.25f;
        if (ActionType == FName("OnActorKilled") || ActionType == FName("OnResourceExhausted"))
        {
            NodePtr->bIsActive = false;
            NodePtr->TensionField += 0.50f;
        }
    }
    else
    {
        FCCGameNode NewNode;
        NewNode.NodeID = TargetID;
        NewNode.Scale = EScaleLevel::Micro_Actor;
        NewNode.TensionField = 0.30f;
        NewNode.bIsActive = true;
        NewNode.bIsQuarantined = false;

        ActiveCCNodes.Add(TargetID, NewNode);
    }

    EvaluateWorldTension();
}

void UCausalWorldSubsystem::UpdateCCNode(const FCCGameNode& InNode)
{
    ActiveCCNodes.FindOrAdd(InNode.NodeID) = InNode;
    EvaluateWorldTension();
}

bool UCausalWorldSubsystem::GetCCNode(FName NodeID, FCCGameNode& OutNode) const
{
    const FCCGameNode* Found = ActiveCCNodes.Find(NodeID);
    if (Found)
    {
        OutNode = *Found;
        return true;
    }
    return false;
}

void UCausalWorldSubsystem::EvaluateWorldTension()
{
    if (ActiveCCNodes.Num() == 0) return;

    // Pack raw node data for CPU multithreaded Task Graph / ParallelFor evaluation
    TArray<FCCNodeRawData> RawNodeArray;
    RawNodeArray.Reserve(ActiveCCNodes.Num());

    for (const auto& Pair : ActiveCCNodes)
    {
        FCCNodeRawData Raw;
        Raw.NodeID = Pair.Value.NodeID;
        Raw.TensionField = Pair.Value.TensionField;
        Raw.CriticalThreshold = Pair.Value.CriticalThreshold;
        Raw.bIsQuarantined = Pair.Value.bIsQuarantined;
        RawNodeArray.Add(Raw);
    }

    // Execute Task Graph ParallelFor evaluation
    FCausalParallelEvaluator::ParallelEvaluateTension(RawNodeArray, CriticalTensionThreshold);

    // Write back evaluated tension values & quarantine flags
    float TotalTension = 0.0f;
    for (const FCCNodeRawData& Raw : RawNodeArray)
    {
        if (FCCGameNode* NodePtr = ActiveCCNodes.Find(Raw.NodeID))
        {
            NodePtr->TensionField = Raw.TensionField;
            if (Raw.bIsQuarantined && !NodePtr->bIsQuarantined)
            {
                NodePtr->bIsQuarantined = true;
                ExecuteSealedAttractorRestructure(Raw.NodeID, Raw.TensionField);
            }
        }
        TotalTension += Raw.TensionField;
    }

    CurrentWorldTension = TotalTension / ActiveCCNodes.Num();

    // Scale shift trigger if tension crosses critical macro boundary
    if (CurrentWorldTension > CriticalTensionThreshold)
    {
        OnPerceptualScaleShifted.Broadcast(EScaleLevel::Macro_Kingdom);
    }
}

void UCausalWorldSubsystem::ExecuteSealedAttractorRestructure(FName RupturedNodeID, float PeakTension)
{
    FName AnomalyID = *FString::Printf(TEXT("ANOMALY_%s_%f"), *RupturedNodeID.ToString(), PeakTension);
    OnSealedAttractorTriggered.Broadcast(AnomalyID, PeakTension);
}
