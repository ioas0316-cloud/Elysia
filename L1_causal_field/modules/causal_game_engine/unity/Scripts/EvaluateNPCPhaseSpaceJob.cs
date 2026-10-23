// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace CausalEngine.Unity.AI
{
    public enum AttractorTypeByte : byte
    {
        Equilibrium = 0,
        Defensive = 1,
        Obsessive = 2,
        Panic = 3
    }

    public struct AttractorBasinData
    {
        public AttractorTypeByte type;
        public float centerVt;
        public float depthWeight;
        public float hysteresisThreshold;
    }

    public struct NPCPhaseInput
    {
        public float nodeVt;
        public float vtGradientMagnitude;
        public float internalDrive;
        public AttractorTypeByte currentAttractor;
    }

    public struct NPCPhaseOutput
    {
        public AttractorTypeByte newAttractor;
        public byte transitionOccurred; // 1: true, 0: false
        public float minPotentialEnergy;
    }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    public struct EvaluateNPCPhaseSpaceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<NPCPhaseInput> inputs;
        [ReadOnly] public NativeArray<AttractorBasinData> basins;
        [WriteOnly] public NativeArray<NPCPhaseOutput> outputs;

        public void Execute(int index)
        {
            NPCPhaseInput input = inputs[index];
            AttractorTypeByte bestBasin = input.currentAttractor;
            float minPotentialEnergy = float.MaxValue;

            int basinCount = basins.Length;
            for (int i = 0; i < basinCount; i++)
            {
                AttractorBasinData basin = basins[i];
                float vtDiff = input.nodeVt - basin.centerVt;

                float potentialEnergy = basin.depthWeight * (vtDiff * vtDiff);
                potentialEnergy -= (input.vtGradientMagnitude * 0.2f) + (input.internalDrive * 0.1f);

                if (basin.type == input.currentAttractor)
                {
                    potentialEnergy -= basin.hysteresisThreshold;
                }

                if (potentialEnergy < minPotentialEnergy)
                {
                    minPotentialEnergy = potentialEnergy;
                    bestBasin = basin.type;
                }
            }

            NPCPhaseOutput output;
            output.newAttractor = bestBasin;
            output.transitionOccurred = (byte)(bestBasin != input.currentAttractor ? 1 : 0);
            output.minPotentialEnergy = minPotentialEnergy;

            outputs[index] = output;
        }
    }

    public class NPCPhaseSpaceManager : MonoBehaviour
    {
        [Header("Simulation Settings")]
        [SerializeField] private int npcCount = 10000;
        [SerializeField] private int batchSize = 64;

        private NativeArray<NPCPhaseInput> inputs;
        private NativeArray<AttractorBasinData> basins;
        private NativeArray<NPCPhaseOutput> outputs;

        private JobHandle phaseJobHandle;

        private void Awake()
        {
            inputs = new NativeArray<NPCPhaseInput>(npcCount, Allocator.Persistent);
            outputs = new NativeArray<NPCPhaseOutput>(npcCount, Allocator.Persistent);

            InitializeBasins();
            InitializeSampleNPCs();
        }

        private void InitializeBasins()
        {
            basins = new NativeArray<AttractorBasinData>(4, Allocator.Persistent);
            basins[0] = new AttractorBasinData { type = AttractorTypeByte.Equilibrium, centerVt = 0.15f, depthWeight = 10.0f, hysteresisThreshold = 0.05f };
            basins[1] = new AttractorBasinData { type = AttractorTypeByte.Defensive,   centerVt = 0.45f, depthWeight = 8.0f,  hysteresisThreshold = 0.08f };
            basins[2] = new AttractorBasinData { type = AttractorTypeByte.Obsessive,   centerVt = 0.75f, depthWeight = 12.0f, hysteresisThreshold = 0.04f };
            basins[3] = new AttractorBasinData { type = AttractorTypeByte.Panic,       centerVt = 0.95f, depthWeight = 15.0f, hysteresisThreshold = 0.02f };
        }

        private void InitializeSampleNPCs()
        {
            for (int i = 0; i < npcCount; i++)
            {
                inputs[i] = new NPCPhaseInput
                {
                    nodeVt = UnityEngine.Random.Range(0.0f, 1.0f),
                    vtGradientMagnitude = UnityEngine.Random.Range(0.0f, 0.5f),
                    internalDrive = UnityEngine.Random.Range(0.0f, 1.0f),
                    currentAttractor = AttractorTypeByte.Equilibrium
                };
            }
        }

        private void Update()
        {
            phaseJobHandle.Complete();

            EvaluateNPCPhaseSpaceJob job = new EvaluateNPCPhaseSpaceJob
            {
                inputs = inputs,
                basins = basins,
                outputs = outputs
            };

            phaseJobHandle = job.Schedule(npcCount, batchSize);
        }

        private void LateUpdate()
        {
            phaseJobHandle.Complete();

            for (int i = 0; i < npcCount; i++)
            {
                if (outputs[i].transitionOccurred == 1)
                {
                    ApplyPhaseTransitionToNPC(i, outputs[i].newAttractor);

                    var updatedInput = inputs[i];
                    updatedInput.currentAttractor = outputs[i].newAttractor;
                    inputs[i] = updatedInput;
                }
            }
        }

        private void ApplyPhaseTransitionToNPC(int npcIndex, AttractorTypeByte newAttractor)
        {
            // BT Blackboard rebind dispatcher hook
        }

        private void OnDestroy()
        {
            phaseJobHandle.Complete();
            if (inputs.IsCreated) inputs.Dispose();
            if (basins.IsCreated) basins.Dispose();
            if (outputs.IsCreated) outputs.Dispose();
        }
    }
}
