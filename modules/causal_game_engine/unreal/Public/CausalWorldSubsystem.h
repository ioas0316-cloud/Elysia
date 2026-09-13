// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "CausalWorldSubsystem.generated.h"

/**
 * Perception scale level for macro/micro causal field dynamics.
 */
UENUM(BlueprintType)
enum class EScaleLevel : uint8
{
    Micro_Actor,
    Meso_Region,
    Macro_Kingdom
};

/**
 * Causal Conservation Node representation in Unreal Engine 5.
 */
USTRUCT(BlueprintType)
struct FCCGameNode
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    FName NodeID;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    EScaleLevel Scale = EScaleLevel::Micro_Actor;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    float TensionField = 0.0f; // V_t

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    float CriticalThreshold = 0.85f; // V_critical

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    bool bIsActive = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Causal Engine")
    bool bIsQuarantined = false;
};

// Dynamic Multicast Delegate fired when critical V_t is exceeded and SealedAttractor isolation triggers
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnSealedAttractorTriggered, FName, AnomalyID, float, PeakTension);

// Delegate fired when observation scale shifts dynamically
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPerceptualScaleShifted, EScaleLevel, NewScale);

/**
 * UCausalWorldSubsystem
 *
 * Auto-bound to UWorld lifecycle in UE5.
 * Converts gameplay actor signals into causal tension vectors, executes multi-threaded
 * V_t evaluation via Task Graph / ParallelFor, and dispatches SealedAttractor events.
 */
UCLASS()
class CAUSALENGINE_API UCausalWorldSubsystem : public UWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    /** Registers a gameplay action as a causal signal into the tension field */
    UFUNCTION(BlueprintCallable, Category = "Causal Engine")
    void RegisterPlayerAction(FName ActionType, AActor* TargetActor);

    /** Manually registers or updates a CC-Node in the world subsystem */
    UFUNCTION(BlueprintCallable, Category = "Causal Engine")
    void UpdateCCNode(const FCCGameNode& InNode);

    /** Retrieves node state by ID */
    UFUNCTION(BlueprintCallable, Category = "Causal Engine")
    bool GetCCNode(FName NodeID, FCCGameNode& OutNode) const;

    /** Dynamic event dispatcher when SealedAttractor isolation is triggered */
    UPROPERTY(BlueprintAssignable, Category = "Causal Engine|Events")
    FOnSealedAttractorTriggered OnSealedAttractorTriggered;

    /** Dynamic event dispatcher when macro/micro perceptual scale shifts */
    UPROPERTY(BlueprintAssignable, Category = "Causal Engine|Events")
    FOnPerceptualScaleShifted OnPerceptualScaleShifted;

private:
    float CurrentWorldTension = 0.0f;
    float CriticalTensionThreshold = 0.85f;

    TMap<FName, FCCGameNode> ActiveCCNodes;

    /** Evaluates global tension and triggers ParallelFor evaluator */
    void EvaluateWorldTension();

    /** Executes asynchronous narrative restructuring when a node ruptures */
    void ExecuteSealedAttractorRestructure(FName RupturedNodeID, float PeakTension);
};
