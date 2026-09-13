// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using System;
using UnityEngine;

namespace CausalEngine.Unity.AI
{
    public enum AttractorType
    {
        Equilibrium,
        Defensive,
        Obsessive,
        Panic
    }

    [Serializable]
    public struct AttractorBasin
    {
        public AttractorType type;
        public float centerVt;            // Minimum potential energy point (Center V_t)
        public float depthWeight;          // Potential energy depth weight
        public float hysteresisThreshold;  // Friction threshold to prevent chattering
    }

    public class BTBlackboard : MonoBehaviour
    {
        public void SetVariable(string key, object value) { }
        public void SetGoal(string goalName) { }
    }

    public class AttractorAIController : MonoBehaviour
    {
        [Header("Attractor Space Configuration")]
        [SerializeField] private AttractorBasin[] basins;
        [SerializeField] private AttractorType currentAttractor = AttractorType.Equilibrium;

        [Header("Internal NPC State")]
        [SerializeField] private float internalDrive = 0.0f;

        private BTBlackboard blackboard;

        private void Awake()
        {
            blackboard = GetComponent<BTBlackboard>();
            if (basins == null || basins.Length == 0)
            {
                basins = new AttractorBasin[]
                {
                    new AttractorBasin { type = AttractorType.Equilibrium, centerVt = 0.15f, depthWeight = 10.0f, hysteresisThreshold = 0.05f },
                    new AttractorBasin { type = AttractorType.Defensive,   centerVt = 0.45f, depthWeight = 8.0f,  hysteresisThreshold = 0.08f },
                    new AttractorBasin { type = AttractorType.Obsessive,   centerVt = 0.75f, depthWeight = 12.0f, hysteresisThreshold = 0.04f },
                    new AttractorBasin { type = AttractorType.Panic,       centerVt = 0.95f, depthWeight = 15.0f, hysteresisThreshold = 0.02f }
                };
            }
        }

        public void EvaluatePhaseSpace(float nodeVt, float vtGradientMagnitude)
        {
            AttractorType bestBasin = currentAttractor;
            float minPotentialEnergy = float.MaxValue;

            foreach (var basin in basins)
            {
                float vtDiff = nodeVt - basin.centerVt;

                // Potential energy quadratic well calculation: E(p) = depth * (V_t - center)^2
                float potentialEnergy = basin.depthWeight * (vtDiff * vtDiff);

                // Deformation via gradient and internal drive
                potentialEnergy -= (vtGradientMagnitude * 0.2f) + (internalDrive * 0.1f);

                // Hysteresis friction against chattering
                if (basin.type == currentAttractor)
                {
                    potentialEnergy -= basin.hysteresisThreshold;
                }

                if (potentialEnergy < minPotentialEnergy)
                {
                    minPotentialEnergy = potentialEnergy;
                    bestBasin = basin.type;
                }
            }

            if (bestBasin != currentAttractor)
            {
                TransitionToAttractor(bestBasin, nodeVt);
            }
        }

        private void TransitionToAttractor(AttractorType newAttractor, float currentVt)
        {
            currentAttractor = newAttractor;
            if (blackboard != null)
            {
                blackboard.SetVariable("CurrentAttractor", newAttractor);
                blackboard.SetVariable("LastTransitionVt", currentVt);

                switch (newAttractor)
                {
                    case AttractorType.Equilibrium:
                        blackboard.SetGoal("Goal_SocializeAndTeach");
                        break;
                    case AttractorType.Defensive:
                        blackboard.SetGoal("Goal_FortifyAndPatrol");
                        break;
                    case AttractorType.Obsessive:
                        blackboard.SetGoal("Goal_ResearchCausalAnomalies");
                        break;
                    case AttractorType.Panic:
                        blackboard.SetGoal("Goal_FleeOrSelfQuarantine");
                        break;
                }
            }
        }
    }
}
