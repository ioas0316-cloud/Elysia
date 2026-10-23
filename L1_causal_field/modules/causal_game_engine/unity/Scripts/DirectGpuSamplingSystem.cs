// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

namespace CausalEngine.Unity.GPU
{
    [BurstCompile]
    public partial struct DirectGpuSamplingSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<DirectGpuTensionBufferSingleton>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var gpuBufferData = SystemAPI.GetSingleton<DirectGpuTensionBufferSingleton>();

            if (!gpuBufferData.MappedTensionData.IsCreated) return;

            var sampleJob = new SampleGpuFieldJob
            {
                TensionBuffer = gpuBufferData.MappedTensionData,
                GridWidth = gpuBufferData.GridWidth,
                GridHeight = gpuBufferData.GridHeight,
                CellSize = gpuBufferData.CellSize
            };

            state.Dependency = sampleJob.ScheduleParallel(state.Dependency);
        }
    }

    [BurstCompile]
    public partial struct SampleGpuFieldJob : IJobEntity
    {
        [ReadOnly] public NativeArray<float> TensionBuffer;
        public int GridWidth;
        public int GridHeight;
        public float CellSize;

        void Execute(in LocalTransform transform, ref DOTS.NPCPhaseComponent phase)
        {
            float3 pos = transform.Position;
            int x = math.clamp((int)(pos.x / CellSize), 0, GridWidth - 1);
            int z = math.clamp((int)(pos.z / CellSize), 0, GridHeight - 1);

            int bufferIndex = x + (z * GridWidth);
            float currentVt = TensionBuffer[bufferIndex];

            phase.NodeVt = currentVt;
        }
    }
}
