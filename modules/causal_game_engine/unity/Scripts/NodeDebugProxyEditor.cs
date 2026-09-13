// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

#if UNITY_EDITOR
using Unity.Entities;
using UnityEditor;
using UnityEngine;

namespace CausalEngine.Unity.Homeostasis
{
    [CustomEditor(typeof(NodeDebugProxy))]
    public class NodeDebugProxyEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            var proxy = (NodeDebugProxy)target;
            var world = World.DefaultGameObjectInjectionWorld;

            if (world == null || !world.IsCreated)
            {
                EditorGUILayout.HelpBox("DOTS World가 활성화되어 있지 않습니다.", MessageType.Info);
                return;
            }

            var em = world.EntityManager;
            if (proxy.TargetEntity == Entity.Null || !em.Exists(proxy.TargetEntity))
            {
                EditorGUILayout.HelpBox("유효한 엔티티가 선택되지 않았습니다.", MessageType.Warning);
                return;
            }

            if (!em.HasComponent<NodeDebugHistoryComponent>(proxy.TargetEntity))
            {
                EditorGUILayout.HelpBox("선택한 엔티티에 NodeDebugHistoryComponent가 없습니다.", MessageType.Warning);
                return;
            }

            var history = em.GetComponentData<NodeDebugHistoryComponent>(proxy.TargetEntity);

            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("파열 이력 검사기 (RingBuffer Inspection)", EditorStyles.boldLabel);

            // 동결 상태 알림 표시
            if (history.IsFrozen)
            {
                EditorGUILayout.HelpBox("CRITICAL: 연산 오염 발생으로 RingBuffer가 동결(IsFrozen = true)되었습니다.", MessageType.Error);
            }
            else
            {
                EditorGUILayout.HelpBox("정상 작동 중 (실시간 기록 진행 중)", MessageType.Info);
            }

            EditorGUILayout.Space(5);

            // 10프레임 역추적 이력 테이블 헤더
            Rect tableHeaderRect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(22), GUILayout.ExpandWidth(true));
            EditorGUI.DrawRect(tableHeaderRect, new Color(0.18f, 0.18f, 0.2f, 1.0f));

            GUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Frame", EditorStyles.boldLabel, GUILayout.Width(70));
            EditorGUILayout.LabelField("Offset", EditorStyles.boldLabel, GUILayout.Width(60));
            EditorGUILayout.LabelField("Current Weight", EditorStyles.boldLabel, GUILayout.Width(110));
            EditorGUILayout.LabelField("Raw Vt Gradient", EditorStyles.boldLabel, GUILayout.ExpandWidth(true));
            GUILayout.EndHorizontal();

            EditorGUILayout.Space(2);

            // 링 버퍼 10개 데이터 샘플 복원 (가장 오래된 프레임 -> 오염 직전 최신 프레임)
            for (int i = 0; i < NodeDebugHistoryComponent.HistoryCapacity; i++)
            {
                history.GetSampleAt(i, out float weight, out float rawVt, out uint frame);

                int frameOffset = i - (NodeDebugHistoryComponent.HistoryCapacity - 1); // 0 -> -9 (과거), 9 -> 0 (최신/파열점)

                // 파열점(가장 최근 프레임) 배경 및 글자색 강조
                bool isRupturePoint = (i == NodeDebugHistoryComponent.HistoryCapacity - 1);
                Color rowBgColor = isRupturePoint
                    ? new Color(0.5f, 0.1f, 0.1f, 0.4f)
                    : (i % 2 == 0 ? new Color(0.12f, 0.12f, 0.12f, 0.3f) : new Color(0.16f, 0.16f, 0.16f, 0.3f));

                Rect rowRect = GUILayoutUtility.GetRect(GUIContent.none, GUIStyle.none, GUILayout.Height(20), GUILayout.ExpandWidth(true));
                EditorGUI.DrawRect(rowRect, rowBgColor);

                GUILayout.BeginHorizontal();

                // 프레임 및 오프셋 정보
                EditorGUILayout.LabelField(frame == 0 ? "-" : frame.ToString(), GUILayout.Width(70));
                EditorGUILayout.LabelField(frameOffset == 0 ? "NOW" : $"{frameOffset}f", GUILayout.Width(60));

                // Weight 수치 (유효성 검사 경고 시 붉은색)
                GUIStyle weightStyle = new GUIStyle(EditorStyles.label);
                if (!float.IsFinite(weight)) weightStyle.normal.textColor = Color.red;
                EditorGUILayout.LabelField(weight.ToString("F4"), weightStyle, GUILayout.Width(110));

                // Raw Vt 수치 (유효성 검사 경고 시 붉은색)
                GUIStyle vtStyle = new GUIStyle(EditorStyles.label);
                if (!float.IsFinite(rawVt)) vtStyle.normal.textColor = Color.red;
                EditorGUILayout.LabelField(rawVt.ToString("F4"), vtStyle, GUILayout.ExpandWidth(true));

                GUILayout.EndHorizontal();
            }

            EditorGUILayout.Space(10);

            // 동결 해제(Unfreeze) 및 수동 복구 버튼
            if (history.IsFrozen)
            {
                if (GUILayout.Button("동결 해제 및 링버퍼 리셋 (Unfreeze)"))
                {
                    history.IsFrozen = false;
                    history.HeadIndex = 0;
                    em.SetComponentData(proxy.TargetEntity, history);
                }
            }
        }
    }
}
#endif
