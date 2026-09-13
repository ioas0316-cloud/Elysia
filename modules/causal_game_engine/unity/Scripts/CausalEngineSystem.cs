// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;

namespace CausalEngine.Unity.DOTS
{
    // Memory aligned component data for Data-Oriented Design (DOD)
    public struct CCNodeComponent : IComponentData
    {
        public float TensionField;       // V_t (friction tension)
        public float CriticalThreshold;  // V_critical
        public bool IsQuarantined;      // SealedAttractor isolation flag
    }

    // Buffer element for topological causal graph links
    public struct CausalLinkElement : IBufferElementData
    {
        public Entity TargetEntity;
        public float Weight;
    }

    public struct SealedAttractorTag : IComponentData {}

    // Burst-compiled parallel job evaluating tension fields via CPU SIMD
    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
    public partial struct EvaluateTensionJob : IJobEntity
    {
        [ReadOnly] public ComponentLookup<CCNodeComponent> NodeLookup;

        public void Execute(ref CCNodeComponent node, in DynamicBuffer<CausalLinkElement> links)
        {
            if (node.IsQuarantined) return;

            float accumulatedTension = 0.0f;

            for (int i = 0; i < links.Length; i++)
            {
                var link = links[i];
                if (NodeLookup.HasComponent(link.TargetEntity))
                {
                    var targetNode = NodeLookup[link.TargetEntity];
                    float tensionDiff = math.abs(node.TensionField - targetNode.TensionField);
                    accumulatedTension += tensionDiff * link.Weight;
                }
            }

            node.TensionField = math.lerp(node.TensionField, accumulatedTension, 0.1f);
        }
    }

    [BurstCompile]
    public partial struct CausalEngineSystem : ISystem
    {
        private ComponentLookup<CCNodeComponent> m_NodeLookup;

        public void OnCreate(ref SystemState state)
        {
            m_NodeLookup = state.GetComponentLookup<CCNodeComponent>(true);
        }

        public void OnUpdate(ref SystemState state)
        {
            m_NodeLookup.Update(ref state);

            var tensionJob = new EvaluateTensionJob
            {
                NodeLookup = m_NodeLookup
            };
            state.Dependency = tensionJob.ScheduleParallel(state.Dependency);

            var ecbSingleton = SystemAPI.GetSingleton<EndSimulationEntityCommandBufferSystem.Singleton>();
            var ecb = ecbSingleton.CreateCommandBuffer(state.WorldUnmanaged).AsParallelWriter();

            state.Dependency = Entities.ForEach((Entity entity, int entityInQueryIndex, ref CCNodeComponent node) =>
            {
                if (!node.IsQuarantined && node.TensionField > node.CriticalThreshold)
                {
                    node.IsQuarantined = true;
                    ecb.AddComponent<SealedAttractorTag>(entityInQueryIndex, entity);
                }
            }).ScheduleParallel(state.Dependency);
        }
    }
}
