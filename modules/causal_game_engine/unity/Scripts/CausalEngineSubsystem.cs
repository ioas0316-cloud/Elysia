// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace CausalEngine.Unity
{
    public enum ScaleLevel
    {
        MicroActor,
        MesoRegion,
        MacroKingdom
    }

    [Serializable]
    public struct CCGameNodeData
    {
        public string nodeId;
        public ScaleLevel scale;
        public float tensionField; // V_t
        public float criticalThreshold; // V_critical
        public bool isActive;
        public bool isQuarantined;
    }

    /// <summary>
    /// MonoBehaviour Subsystem pattern for Unity environment.
    /// Acts as state bridge for event interception, tension evaluation, and SealedAttractor dispatching.
    /// </summary>
    public class CausalEngineSubsystem : MonoBehaviour
    {
        public static CausalEngineSubsystem Instance { get; private set; }

        public event Action<string, float> OnSealedAttractorTriggered;
        public event Action<ScaleLevel> OnPerceptualScaleShifted;

        [SerializeField] private float criticalTensionThreshold = 0.85f;
        private Dictionary<string, CCGameNodeData> activeCCNodes = new Dictionary<string, CCGameNodeData>();

        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
            }
        }

        /// <summary>
        /// Registers or updates a CC-Node in the active tracking map.
        /// </summary>
        public void RegisterOrUpdateNode(CCGameNodeData nodeData)
        {
            activeCCNodes[nodeData.nodeId] = nodeData;
        }

        /// <summary>
        /// Dispatches game signals (e.g. OnActorKilled, OnResourceExhausted).
        /// </summary>
        public void DispatchGameSignal(string actionType, string targetEntityId)
        {
            if (activeCCNodes.TryGetValue(targetEntityId, out var node))
            {
                node.isActive = false;
                node.tensionField += 0.50f;
                activeCCNodes[targetEntityId] = node;
            }
            else
            {
                node = new CCGameNodeData
                {
                    nodeId = targetEntityId,
                    scale = ScaleLevel.MicroActor,
                    tensionField = 0.35f,
                    criticalThreshold = criticalTensionThreshold,
                    isActive = true,
                    isQuarantined = false
                };
                activeCCNodes[targetEntityId] = node;
            }

            EvaluateCausalTension(actionType, targetEntityId);
        }

        private void EvaluateCausalTension(string actionType, string targetEntityId)
        {
            float calculatedVt = CalculateCurrentTension(targetEntityId);

            if (calculatedVt > criticalTensionThreshold)
            {
                string anomalyId = $"ANOMALY_{targetEntityId}_{Time.time}";

                if (activeCCNodes.TryGetValue(targetEntityId, out var node))
                {
                    node.isQuarantined = true;
                    activeCCNodes[targetEntityId] = node;
                }

                OnSealedAttractorTriggered?.Invoke(anomalyId, calculatedVt);
                OnPerceptualScaleShifted?.Invoke(ScaleLevel.MacroKingdom);
            }
        }

        private float CalculateCurrentTension(string entityId)
        {
            if (activeCCNodes.TryGetValue(entityId, out var node))
            {
                return node.tensionField;
            }
            return 0.85f;
        }

        public bool TryGetNode(string nodeId, out CCGameNodeData nodeData)
        {
            return activeCCNodes.TryGetValue(nodeId, out nodeData);
        }
    }
}
