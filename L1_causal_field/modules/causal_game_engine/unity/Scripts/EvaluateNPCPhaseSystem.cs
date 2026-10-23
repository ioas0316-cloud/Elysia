// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Burst;
using Unity.Collections;
using Unity.Entities;

namespace CausalEngine.Unity.DOTS
{
    public enum AttractorTypeEnum : byte
    {
        Equilibrium = 0,
        Defensive = 1,
        Obsessive = 2,
        Panic = 3
    }

    public struct NPCPhaseComponent : IComponentData
    {
        public float NodeVt;                  // CC-Node tension
        public float VtGradientMagnitude;     // Tension gradient magnitude
        public float InternalDrive;           // Internal drive / motivation
        public AttractorTypeEnum CurrentAttractor;
        public AttractorTypeEnum NewAttractor;
        public bool TransitionOccurred;
        public float MinPotentialEnergy;
    }

    public struct NPCGoalComponent : IComponentData
    {
        public AttractorTypeEnum CurrentGoal;
        public bool NeedsBTRebind;
    }

    public struct AttractorBasinBlobData
    {
        public AttractorTypeEnum Type;
        public float CenterVt;
        public float DepthWeight;
        public float HysteresisThreshold;
    }

    public struct AttractorConfigurationBlob
    {
        public BlobArray<AttractorBasinBlobData> Basins;
    }

    public struct AttractorConfigSingleton : IComponentData
    {
        public BlobAssetReference<AttractorConfigurationBlob> BlobRef;
    }

    [BurstCompile]
    public partial struct EvaluateNPCPhaseSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<AttractorConfigSingleton>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var configSingleton = SystemAPI.GetSingleton<AttractorConfigSingleton>();

            var evaluateJob = new EvaluateNPCPhaseJob
            {
                BasinsBlob = configSingleton.BlobRef
            };

            state.Dependency = evaluateJob.ScheduleParallel(state.Dependency);
        }

        [BurstCompile]
        public void OnDestroy(ref SystemState state) { }
    }

    [BurstCompile]
    public partial struct EvaluateNPCPhaseJob : IJobEntity
    {
        [ReadOnly] public BlobAssetReference<AttractorConfigurationBlob> BasinsBlob;

        void Execute(ref NPCPhaseComponent phase, ref NPCGoalComponent goal)
        {
            ref var basins = ref BasinsBlob.Value.Basins;

            AttractorTypeEnum bestBasin = phase.CurrentAttractor;
            float minPotentialEnergy = float.MaxValue;

            int basinCount = basins.Length;
            for (int i = 0; i < basinCount; i++)
            {
                ref readonly var basin = ref basins[i];
                float vtDiff = phase.NodeVt - basin.CenterVt;

                float potentialEnergy = basin.DepthWeight * (vtDiff * vtDiff);
                potentialEnergy -= (phase.VtGradientMagnitude * 0.2f) + (phase.InternalDrive * 0.1f);

                if (basin.Type == phase.CurrentAttractor)
                {
                    potentialEnergy -= basin.HysteresisThreshold;
                }

                if (potentialEnergy < minPotentialEnergy)
                {
                    minPotentialEnergy = potentialEnergy;
                    bestBasin = basin.Type;
                }
            }

            bool transition = (bestBasin != phase.CurrentAttractor);
            phase.NewAttractor = bestBasin;
            phase.TransitionOccurred = transition;
            phase.MinPotentialEnergy = minPotentialEnergy;

            if (transition)
            {
                phase.CurrentAttractor = bestBasin;
                goal.CurrentGoal = bestBasin;
                goal.NeedsBTRebind = true;
            }
        }
    }
}
