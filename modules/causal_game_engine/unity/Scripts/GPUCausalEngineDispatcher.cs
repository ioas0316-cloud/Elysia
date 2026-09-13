// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using UnityEngine;

namespace CausalEngine.Unity.GPU
{
    public class GPUCausalEngineDispatcher : MonoBehaviour
    {
        [SerializeField] private ComputeShader causalShader;
        [SerializeField] private RenderTexture tensionFieldTexture;

        private GraphicsBuffer nodesBuffer;
        private GraphicsBuffer linksBuffer;
        private GraphicsBuffer quarantinedBuffer;

        private int evaluateKernel;
        private int renderKernel;
        private const int TOTAL_NODES = 1000000; // 1M CC-Nodes

        private void Start()
        {
            if (causalShader == null) return;

            evaluateKernel = causalShader.FindKernel("CSMain_EvaluateTension");
            renderKernel = causalShader.FindKernel("CSMain_RenderTensionFieldTexture");

            // CCNodeGPUData stride: 2 floats + 3 uints + 2 floats = 28 bytes
            int nodeStride = sizeof(float) * 2 + sizeof(uint) * 3 + sizeof(float) * 2;
            nodesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, TOTAL_NODES, nodeStride);
            quarantinedBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Append, TOTAL_NODES, sizeof(uint));

            quarantinedBuffer.SetCounterValue(0);

            causalShader.SetBuffer(evaluateKernel, "_NodesBuffer", nodesBuffer);
            causalShader.SetBuffer(evaluateKernel, "_QuarantinedQueue", quarantinedBuffer);

            if (tensionFieldTexture != null)
            {
                causalShader.SetTexture(renderKernel, "_TensionFieldTexture", tensionFieldTexture);
            }
        }

        private void Update()
        {
            if (causalShader == null || nodesBuffer == null) return;

            int threadGroups = Mathf.CeilToInt(TOTAL_NODES / 256f);
            causalShader.Dispatch(evaluateKernel, threadGroups, 1, 1);

            if (tensionFieldTexture != null)
            {
                causalShader.Dispatch(renderKernel, tensionFieldTexture.width / 16, tensionFieldTexture.height / 16, 1);
            }
        }

        public GraphicsBuffer GetNodesBuffer() => nodesBuffer;
        public GraphicsBuffer GetQuarantinedBuffer() => quarantinedBuffer;

        private void OnDestroy()
        {
            nodesBuffer?.Release();
            linksBuffer?.Release();
            quarantinedBuffer?.Release();
        }
    }
}
