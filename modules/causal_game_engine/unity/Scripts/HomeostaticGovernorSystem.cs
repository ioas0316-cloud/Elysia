// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;

namespace CausalEngine.Unity.Homeostasis
{
    public struct CCNodeHomeostasisComponent : IComponentData
    {
        public float CurrentWeight;     // L_ij current
        public float CoreWeight;        // L_ij core
        public float RawVtGradient;     // Raw V_t gradient
        public float ClampedVtGradient; // Clamped V_t* gradient
        public float LocalPotential;    // U_core local
    }

    public struct HomeostasisConfigSingleton : IComponentData
    {
        public float KappaCore;  // Stiffness kappa_core
        public float UMax;       // U_max threshold
        public float Lambda;     // Damping lambda
        public float PowerP;     // Power p
        public float Gamma;      // Restorative gamma
        public float TotalUCore; // Accumulated U_core
    }

    [BurstCompile]
    public partial struct HomeostaticGovernorSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<HomeostasisConfigSingleton>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var configEntity = SystemAPI.GetSingletonEntity<HomeostasisConfigSingleton>();
            var config = SystemAPI.GetComponent<HomeostasisConfigSingleton>(configEntity);

            var accumulatedUCore = new NativeReference<float>(0.0f, Allocator.TempJob);

            var calculatePotentialJob = new CalculateCorePotentialJob
            {
                KappaCore = config.KappaCore,
                TotalUCoreRef = accumulatedUCore
            };
            state.Dependency = calculatePotentialJob.ScheduleParallel(state.Dependency);

            var clampTensionJob = new ClampVtGradientJob
            {
                KappaCore = config.KappaCore,
                UMax = config.UMax,
                Lambda = config.Lambda,
                PowerP = config.PowerP,
                Gamma = config.Gamma,
                TotalUCoreRef = accumulatedUCore
            };
            state.Dependency = clampTensionJob.ScheduleParallel(state.Dependency);

            accumulatedUCore.Dispose(state.Dependency);
        }
    }

    [BurstCompile]
    public partial struct CalculateCorePotentialJob : IJobEntity
    {
        public float KappaCore;
        [NativeDisableParallelForRestriction] public NativeReference<float> TotalUCoreRef;

        void Execute(ref CCNodeHomeostasisComponent node)
        {
            float deltaL = node.CurrentWeight - node.CoreWeight;
            float localU = 0.5f * KappaCore * (deltaL * deltaL);
            node.LocalPotential = localU;

            unsafe
            {
                float* ptr = (float*)TotalUCoreRef.GetUnsafePtr();
                AtomicAddFloat(ptr, localU);
            }
        }

        private static void AtomicAddFloat(unsafe float* ptr, float value)
        {
            int* intPtr = (int*)ptr;
            int oldInt, newInt;
            do
            {
                oldInt = *intPtr;
                float oldValue = math.asfloat(oldInt);
                float newValue = oldValue + value;
                newInt = math.asint(newValue);
            }
            while (System.Threading.Interlocked.CompareExchange(ref *intPtr, newInt, oldInt) != oldInt);
        }
    }

    [BurstCompile]
    public partial struct ClampVtGradientJob : IJobEntity
    {
        public float KappaCore;
        public float UMax;
        public float Lambda;
        public float PowerP;
        public float Gamma;
        [ReadOnly] public NativeReference<float> TotalUCoreRef;

        void Execute(ref CCNodeHomeostasisComponent node)
        {
            float totalUCore = TotalUCoreRef.Value;

            float normalizedU = math.saturate(totalUCore / math.max(UMax, 0.0001f));
            float sigmaH = math.exp(-Lambda * math.pow(normalizedU, PowerP));

            float gradUCore = KappaCore * (node.CurrentWeight - node.CoreWeight);

            float clampedVt = (sigmaH * node.RawVtGradient) - (Gamma * gradUCore);

            node.ClampedVtGradient = math.clamp(clampedVt, -100.0f, 100.0f);

            if (totalUCore >= UMax)
            {
                node.ClampedVtGradient = -Gamma * gradUCore;
            }
        }
    }
}
