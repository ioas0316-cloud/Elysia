// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using UnityEngine;

namespace CausalEngine.Unity.Homeostasis
{
    // 엔티티를 선택했을 때 Inspector 표출을 유도하는 Debug Target 컴포넌트
    public class NodeDebugProxy : MonoBehaviour
    {
        // 런타임에 선택된 Entity 참조 저장
        public Unity.Entities.Entity TargetEntity;
    }
}
