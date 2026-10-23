// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using UnityEngine;

namespace CausalEngine.Unity.GPU
{
    public struct DirectGpuTensionBufferSingleton : IComponentData
    {
        public NativeArray<float> MappedTensionData;
        public int GridWidth;
        public int GridHeight;
        public float CellSize;
        public int CurrentBufferIndex;
    }

    public class GpuTensionBufferProvider : MonoBehaviour
    {
        [Header("Field Configuration")]
        [SerializeField] private ComputeShader tensionShader;
        [SerializeField] private int gridWidth = 256;
        [SerializeField] private int gridHeight = 256;
        [SerializeField] private float cellSize = 2.0f;

        private const int RING_BUFFER_SIZE = 3;
        private GraphicsBuffer[] gpuBuffers = new GraphicsBuffer[RING_BUFFER_SIZE];
        private NativeArray<float>[] nativePointers = new NativeArray<float>[RING_BUFFER_SIZE];

        private int frameCounter = 0;
        private Entity singletonEntity;
        private EntityManager entityManager;

        private void Awake()
        {
            int totalElements = gridWidth * gridHeight;
            int stride = sizeof(float);

            if (World.DefaultGameObjectInjectionWorld != null)
            {
                entityManager = World.DefaultGameObjectInjectionWorld.EntityManager;
                singletonEntity = entityManager.CreateEntity(typeof(DirectGpuTensionBufferSingleton));
            }

            for (int i = 0; i < RING_BUFFER_SIZE; i++)
            {
                gpuBuffers[i] = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured,
                    GraphicsBuffer.UsageFlags.None,
                    totalElements,
                    stride
                );

                unsafe
                {
                    IntPtr bufferPtr = gpuBuffers[i].GetNativeBufferPtr();
                    nativePointers[i] = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<float>(
                        (void*)bufferPtr,
                        totalElements,
                        Allocator.None
                    );

#if UNITY_EDITOR
                    NativeArrayUnsafeUtility.SetAtomicSafetyHandle(
                        ref nativePointers[i],
                        AtomicSafetyHandle.GetEmptyHandle()
                    );
#endif
                }
            }
        }

        private void Update()
        {
            if (tensionShader == null) return;

            int gpuWriteIdx = frameCounter % RING_BUFFER_SIZE;
            int cpuReadIdx = (frameCounter + 1) % RING_BUFFER_SIZE;

            int kernel = tensionShader.FindKernel("CSMain_EvaluateTension");
            tensionShader.SetBuffer(kernel, "_TensionOutputBuffer", gpuBuffers[gpuWriteIdx]);
            tensionShader.Dispatch(kernel, gridWidth / 8, gridHeight / 8, 1);

            if (entityManager != null && entityManager.Exists(singletonEntity))
            {
                entityManager.SetComponentData(singletonEntity, new DirectGpuTensionBufferSingleton
                {
                    MappedTensionData = nativePointers[cpuReadIdx],
                    GridWidth = gridWidth,
                    GridHeight = gridHeight,
                    CellSize = cellSize,
                    CurrentBufferIndex = cpuReadIdx
                });
            }

            frameCounter++;
        }

        private void OnDestroy()
        {
            for (int i = 0; i < RING_BUFFER_SIZE; i++)
            {
                gpuBuffers[i]?.Release();
            }
        }
    }
}
