using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace Elysia.GameMechanics
{
    public class TrajectoryFeedbackBridge : MonoBehaviour
    {
        [SerializeField] private ComputeShader feedbackKernel;

        private ComputeBuffer densityBuffer;
        private ComputeBuffer fluxBuffer;
        private ComputeBuffer metricTensorBuffer;

        private const int GRID_SIZE = 64;
        private const int TOTAL_CELLS = GRID_SIZE * GRID_SIZE * GRID_SIZE;

        private void Awake()
        {
            densityBuffer = new ComputeBuffer(TOTAL_CELLS, sizeof(float));
            fluxBuffer = new ComputeBuffer(TOTAL_CELLS, sizeof(float) * 3);
            metricTensorBuffer = new ComputeBuffer(TOTAL_CELLS, sizeof(float) * 9); // 3x3 Metric Tensor g_ij

            ClearBuffers();
        }

        private void ClearBuffers()
        {
            float[] emptyDensity = new float[TOTAL_CELLS];
            Vector3[] emptyFlux = new Vector3[TOTAL_CELLS];
            densityBuffer.SetData(emptyDensity);
            fluxBuffer.SetData(emptyFlux);
        }

        // Collect agent trajectory density & flux to feedback into metric tensor field g_ij
        public void DispatchFeedbackLoop(NativeArray<float3> agentPositions, NativeArray<float3> agentVelocities, float deltaTime)
        {
            if (feedbackKernel == null || !agentPositions.IsCreated || agentPositions.Length == 0) return;

            // Step 1: Splat agent trajectories onto 3D grid
            int kernelSplat = feedbackKernel.FindKernel("SplatAgentTrajectory");

            ComputeBuffer posBuffer = new ComputeBuffer(agentPositions.Length, sizeof(float) * 3);
            ComputeBuffer velBuffer = new ComputeBuffer(agentVelocities.Length, sizeof(float) * 3);
            posBuffer.SetData(agentPositions);
            velBuffer.SetData(agentVelocities);

            feedbackKernel.SetBuffer(kernelSplat, "b_Positions", posBuffer);
            feedbackKernel.SetBuffer(kernelSplat, "b_Velocities", velBuffer);
            feedbackKernel.SetBuffer(kernelSplat, "b_DensityGrid", densityBuffer);
            feedbackKernel.SetBuffer(kernelSplat, "b_FluxGrid", fluxBuffer);
            feedbackKernel.SetInt("u_AgentCount", agentPositions.Length);

            int threadGroupsX = Mathf.CeilToInt(agentPositions.Length / 64.0f);
            feedbackKernel.Dispatch(kernelSplat, threadGroupsX, 1, 1);

            // Step 2: Evolve metric tensor field g_ij from density gradients and momentum flux
            int kernelUpdateMetric = feedbackKernel.FindKernel("UpdateMetricTensorField");

            feedbackKernel.SetBuffer(kernelUpdateMetric, "b_DensityGrid", densityBuffer);
            feedbackKernel.SetBuffer(kernelUpdateMetric, "b_FluxGrid", fluxBuffer);
            feedbackKernel.SetBuffer(kernelUpdateMetric, "b_MetricTensor", metricTensorBuffer);
            feedbackKernel.SetFloat("u_DeltaTime", deltaTime);
            feedbackKernel.SetFloat("u_DecayRate", 0.05f); // Trajectory dissipation rate

            int threadGroupGrid = GRID_SIZE / 4;
            feedbackKernel.Dispatch(kernelUpdateMetric, threadGroupGrid, threadGroupGrid, threadGroupGrid);

            posBuffer.Release();
            velBuffer.Release();
        }

        private void OnDestroy()
        {
            densityBuffer?.Release();
            fluxBuffer?.Release();
            metricTensorBuffer?.Release();
        }
    }
}
