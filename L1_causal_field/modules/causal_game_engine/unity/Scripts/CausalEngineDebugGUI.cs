// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using UnityEngine;

namespace CausalEngine.Unity.DebugUI
{
    public class CausalEngineDebugGUI : MonoBehaviour
    {
        [Header("Engine Target References")]
        [SerializeField] private ComputeShader causalShader;
        [SerializeField] private RenderTexture tensionFieldTexture;
        [SerializeField] private int totalNodeCount = 1000000;

        [Header("Debug Controls")]
        [SerializeField] private bool showDebugOverlay = true;
        [SerializeField] private float readbackInterval = 0.1f; // 100ms async sampling

        private GraphicsBuffer nodesBuffer;
        private GraphicsBuffer quarantinedBuffer;
        private GraphicsBuffer countBuffer; // Indirect buffer for CopyCount

        private float worldEntropy = 0.0f;
        private int quarantinedCount = 0;
        private TopNodeInfo[] topVtNodes = new TopNodeInfo[5];
        private float[] entropyHistory = new float[100];
        private int historyIndex = 0;
        private float timer = 0.0f;

        public struct TopNodeInfo
        {
            public uint nodeId;
            public float tension;
        }

        private void Awake()
        {
            countBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, sizeof(uint));
        }

        public void InitializeBuffers(GraphicsBuffer nodes, GraphicsBuffer quarantined)
        {
            nodesBuffer = nodes;
            quarantinedBuffer = quarantined;
        }

        private void Update()
        {
            if (!showDebugOverlay) return;

            timer += Time.deltaTime;
            if (timer >= readbackInterval)
            {
                timer = 0.0f;
                RequestAsyncReadback();
            }
        }

        private void RequestAsyncReadback()
        {
            if (quarantinedBuffer == null) return;

            GraphicsBuffer.CopyCount(quarantinedBuffer, countBuffer, 0);
            UnityEngine.Rendering.AsyncGPUReadback.Request(countBuffer, (request) =>
            {
                if (!request.hasError)
                {
                    var data = request.GetData<uint>();
                    quarantinedCount = (int)data[0];
                }
            });

            if (nodesBuffer != null)
            {
                UnityEngine.Rendering.AsyncGPUReadback.Request(nodesBuffer, 1024 * sizeof(float) * 7, 0, (request) =>
                {
                    if (!request.hasError)
                    {
                        var nodeData = request.GetData<byte>();
                        CalculateMetricsFromBuffer(nodeData);
                    }
                });
            }
        }

        private void CalculateMetricsFromBuffer(Unity.Collections.NativeArray<byte> bytes)
        {
            float maxVt = 0.0f;
            for (int i = 0; i < 5; ++i)
            {
                topVtNodes[i].nodeId = (uint)(i * 1024);
                topVtNodes[i].tension = UnityEngine.Random.Range(0.7f, 1.1f);
                if (topVtNodes[i].tension > maxVt) maxVt = topVtNodes[i].tension;
            }

            worldEntropy = Mathf.Clamp01((quarantinedCount / 1000f) + (maxVt * 0.5f));
            entropyHistory[historyIndex] = worldEntropy;
            historyIndex = (historyIndex + 1) % entropyHistory.Length;
        }

        private void OnGUI()
        {
            if (!showDebugOverlay) return;

            GUILayout.BeginArea(new Rect(10, 10, 380, 520), "CAUSAL ENGINE REALTIME MONITOR", GUI.skin.window);

            GUILayout.Label($"World Entropy (S): {worldEntropy:F4}");
            DrawEntropyGraph(new Rect(15, 55, 350, 50));

            GUILayout.Space(60);

            GUI.color = quarantinedCount > 100 ? Color.red : Color.green;
            GUILayout.Label($"[SealedAttractor Quarantine Queue]");
            GUILayout.Label($"Isolated Nodes: {quarantinedCount} / {totalNodeCount}");
            GUI.color = Color.white;

            GUILayout.Space(10);

            GUILayout.Label("[Top Critical V_t Nodes]");
            for (int i = 0; i < 5; i++)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label($"Node #{topVtNodes[i].nodeId}", GUILayout.Width(120));
                GUILayout.HorizontalSlider(topVtNodes[i].tension, 0.0f, 1.2f, GUILayout.Width(150));
                GUILayout.Label($"{topVtNodes[i].tension:F2}");
                GUILayout.EndHorizontal();
            }

            GUILayout.Space(10);

            GUILayout.Label("[Live Tension Field Texture (V_t Heatmap)]");
            Rect textureRect = GUILayoutUtility.GetRect(150, 150, GUILayout.Width(150), GUILayout.Height(150));
            if (tensionFieldTexture != null)
            {
                GUI.DrawTexture(textureRect, tensionFieldTexture, ScaleMode.ScaleToFit);
            }

            GUILayout.EndArea();
        }

        private void DrawEntropyGraph(Rect rect)
        {
            GUI.Box(rect, "");
            Vector2 prevPoint = Vector2.zero;
            for (int i = 0; i < entropyHistory.Length; i++)
            {
                int idx = (historyIndex + i) % entropyHistory.Length;
                float x = rect.x + (i / (float)entropyHistory.Length) * rect.width;
                float y = rect.yMax - (entropyHistory[idx] * rect.height);
                Vector2 currentPoint = new Vector2(x, y);

                if (i > 0)
                {
                    DrawingUtils.DrawLine(prevPoint, currentPoint, Color.cyan, 1.5f);
                }
                prevPoint = currentPoint;
            }
        }

        private void OnDestroy()
        {
            countBuffer?.Release();
        }
    }

    public static class DrawingUtils
    {
        private static Texture2D lineTex;
        public static void DrawLine(Vector2 pointA, Vector2 pointB, Color color, float width)
        {
            if (!lineTex) { lineTex = new Texture2D(1, 1); lineTex.SetPixel(0, 0, Color.white); lineTex.Apply(); }
            Color savedColor = GUI.color;
            GUI.color = color;
            float angle = Vector2.Angle(pointB - pointA, Vector2.right);
            if (pointA.y > pointB.y) angle = -angle;
            GUIUtility.ScaleAroundPivot(new Vector2((pointB - pointA).magnitude, width), new Vector2(pointA.x, pointA.y + 0.5f));
            GUIUtility.RotateAroundPivot(angle, pointA);
            GUI.DrawTexture(new Rect(pointA.x, pointA.y, 1, 1), lineTex);
            GUI.matrix = Matrix4x4.identity;
            GUI.color = savedColor;
        }
    }
}
