// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine;

namespace CausalEngine.Unity.Homeostasis
{
    [WorldSystemFilter(WorldSystemFilterFlags.Default | WorldSystemFilterFlags.Editor)]
    [UpdateInGroup(typeof(PresentationSystemGroup))]
    public partial class FrozenNodeGizmoSystem : SystemBase
    {
        protected override void OnUpdate()
        {
            // IsFrozen 상태인 노드만 필터링하여 Scene 뷰 및 Game 뷰에 3D Gizmo 라인 시각화
            foreach (var (history, ltw) in SystemAPI.Query<RefRO<NodeDebugHistoryComponent>, RefRO<LocalToWorld>>())
            {
                ref readonly var hist = ref history.ValueRO;
                if (!hist.IsFrozen) continue;

                float3 basePos = ltw.ValueRO.Position;

                // 1. 파열 위치 수직 신호 핀 (Red Alert Line)
                Debug.DrawLine(basePos, basePos + new float3(0, 2.0f, 0), Color.red);

                float3 prevPoint = float3.zero;

                // 2. 직전 10프레임 이력 궤적 시각화 (X: 프레임 오프셋, Y: Weight 변위)
                for (int i = 0; i < NodeDebugHistoryComponent.HistoryCapacity; i++)
                {
                    hist.GetSampleAt(i, out float weight, out float rawVt, out uint frame);

                    // 엔티티 위치 기준 시간 흐름(X축 -0.25f 간격)과 Weight 변화(Y축) 공간 복원
                    float xOffset = (i - (NodeDebugHistoryComponent.HistoryCapacity - 1)) * 0.25f;
                    float3 currentPoint = basePos + new float3(xOffset, weight * 0.5f, 0);

                    if (i > 0)
                    {
                        // 파열 직전으로 갈수록 노란색에서 붉은색으로 그라데이션
                        float t = (float)i / (NodeDebugHistoryComponent.HistoryCapacity - 1);
                        Color lineColor = Color.Lerp(new Color(1.0f, 0.8f, 0.2f), Color.red, t);

                        Debug.DrawLine(prevPoint, currentPoint, lineColor);
                    }

                    // 각 프레임 샘플링 위치 십자 마커
                    Debug.DrawRay(currentPoint, new float3(0, 0.08f, 0), Color.cyan);

                    prevPoint = currentPoint;
                }
            }
        }
    }
}
