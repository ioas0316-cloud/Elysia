// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#if UNITY_EDITOR
using Unity.Entities;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace CausalEngine.Unity.Homeostasis
{
    public class HomeostasisProfilerWindow : EditorWindow
    {
        private const int HistorySize = 200;
        private readonly float[] _uCoreHistory = new float[HistorySize];
        private readonly float[] _sigmaHHistory = new float[HistorySize];
        private int _historyIndex = 0;

        [MenuItem("Window/AI Architecture/Homeostasis Profiler")]
        public static void Open()
        {
            GetWindow<HomeostasisProfilerWindow>("Homeostasis Profiler");
        }

        private void OnEnable() => EditorApplication.update += Repaint;
        private void OnDisable() => EditorApplication.update -= Repaint;

        private void OnGUI()
        {
            var world = World.DefaultGameObjectInjectionWorld;
            if (world == null || !world.IsCreated)
            {
                EditorGUILayout.HelpBox("DOTS World가 활성화되어 있지 않습니다.", MessageType.Info);
                return;
            }

            var em = world.EntityManager;
            var query = em.CreateEntityQuery(typeof(HomeostasisConfigSingleton));
            if (!query.HasSingleton<HomeostasisConfigSingleton>())
            {
                EditorGUILayout.HelpBox("HomeostasisConfigSingleton을 찾을 수 없습니다.", MessageType.Warning);
                return;
            }

            var config = query.GetSingleton<HomeostasisConfigSingleton>();

            // 1. 실시간 히스토리 버퍼 갱신
            if (Application.isPlaying)
            {
                _uCoreHistory[_historyIndex] = config.TotalUCore;

                float normalizedU = math.saturate(config.TotalUCore / math.max(config.UMax, 0.0001f));
                float sigmaH = math.exp(-config.Lambda * math.pow(normalizedU, config.PowerP));
                _sigmaHHistory[_historyIndex] = sigmaH;

                _historyIndex = (_historyIndex + 1) % HistorySize;
            }

            // 2. 그래프 드로잉
            DrawGraph(config);

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Invariable Core 실시간 계수 튜닝", EditorStyles.boldLabel);

            // 3. 파라미터 실시간 트래킹 및 Write-Back
            EditorGUI.BeginChangeCheck();

            float kappa = EditorGUILayout.FloatField("Kappa Core (강성 계수)", config.KappaCore);
            float uMax = EditorGUILayout.FloatField("U_max (최대 허용 임계)", config.UMax);
            float lambda = EditorGUILayout.Slider("Lambda (감쇄 경사도)", config.Lambda, 0.1f, 20.0f);
            float powerP = EditorGUILayout.Slider("Power P (지수 파워)", config.PowerP, 1.0f, 10.0f);
            float gamma = EditorGUILayout.Slider("Gamma (중심 복원력)", config.Gamma, 0.01f, 5.0f);

            if (EditorGUI.EndChangeCheck())
            {
                config.KappaCore = math.max(0.001f, kappa);
                config.UMax = math.max(0.001f, uMax);
                config.Lambda = lambda;
                config.PowerP = powerP;
                config.Gamma = gamma;

                var entity = query.GetSingletonEntity();
                em.SetComponentData(entity, config);
            }
        }

        private void DrawGraph(HomeostasisConfigSingleton config)
        {
            Rect graphRect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(200), GUILayout.ExpandWidth(true));
            EditorGUI.DrawRect(graphRect, new Color(0.08f, 0.08f, 0.1f, 1.0f));

            // 가이드라인 (50%, 80% 임계 라인)
            float line80Y = graphRect.yMax - (0.8f * graphRect.height);
            Handles.color = new Color(1.0f, 0.3f, 0.2f, 0.5f);
            Handles.DrawLine(new Vector3(graphRect.x, line80Y), new Vector3(graphRect.xMax, line80Y));

            // 좌표 바운딩 및 폴리라인 생성
            Vector3[] uCorePoints = new Vector3[HistorySize];
            Vector3[] sigmaHPoints = new Vector3[HistorySize];
            float xStep = graphRect.width / (HistorySize - 1);

            for (int i = 0; i < HistorySize; i++)
            {
                int idx = (_historyIndex + i) % HistorySize;
                float x = graphRect.x + i * xStep;

                // U_core 정규화 좌표
                float normU = math.saturate(_uCoreHistory[idx] / math.max(config.UMax, 0.0001f));
                uCorePoints[i] = new Vector3(x, graphRect.yMax - (normU * graphRect.height), 0);

                // SigmaH 정규화 좌표
                sigmaHPoints[i] = new Vector3(x, graphRect.yMax - (_sigmaHHistory[idx] * graphRect.height), 0);
            }

            // 렌더링
            Handles.color = new Color(0.2f, 0.8f, 1.0f); // Cyan: U_core
            Handles.DrawAAPolyLine(2.5f, uCorePoints);

            Handles.color = new Color(1.0f, 0.8f, 0.2f); // Yellow: Sigma_H
            Handles.DrawAAPolyLine(2.0f, sigmaHPoints);

            // 범례 및 텍스트
            GUI.Label(new Rect(graphRect.x + 10, graphRect.y + 5, 300, 20), $"U_core : {config.TotalUCore:F3} / {config.UMax:F3}", EditorStyles.boldLabel);
            GUI.Label(new Rect(graphRect.x + 10, graphRect.y + 25, 300, 20), "Cyan: U_core (포텐셜) | Yellow: σ_H (V_t 감쇄율)", EditorStyles.miniLabel);
        }
    }
}
#endif
