// Copyright (c) 2025 Elysia Causal Intelligence. All Rights Reserved.

using Unity.Collections;
using Unity.Entities;
using UnityEngine;

namespace CausalEngine.Unity.DOTS
{
    public class AttractorGlobalAuthoring : MonoBehaviour
    {
        public class Baker : Baker<AttractorGlobalAuthoring>
        {
            public override void Bake(AttractorGlobalAuthoring authoring)
            {
                var entity = GetEntity(TransformUsageFlags.None);

                using (var builder = new BlobBuilder(Allocator.Temp))
                {
                    ref var root = ref builder.ConstructRoot<AttractorConfigurationBlob>();
                    var arrayBuilder = builder.Allocate(ref root.Basins, 4);

                    arrayBuilder[0] = new AttractorBasinBlobData { Type = AttractorTypeEnum.Equilibrium, CenterVt = 0.15f, DepthWeight = 10.0f, HysteresisThreshold = 0.05f };
                    arrayBuilder[1] = new AttractorBasinBlobData { Type = AttractorTypeEnum.Defensive,   CenterVt = 0.45f, DepthWeight = 8.0f,  HysteresisThreshold = 0.08f };
                    arrayBuilder[2] = new AttractorBasinBlobData { Type = AttractorTypeEnum.Obsessive,   CenterVt = 0.75f, DepthWeight = 12.0f, HysteresisThreshold = 0.04f };
                    arrayBuilder[3] = new AttractorBasinBlobData { Type = AttractorTypeEnum.Panic,       CenterVt = 0.95f, DepthWeight = 15.0f, HysteresisThreshold = 0.02f };

                    var blobRef = builder.CreateBlobAssetReference<AttractorConfigurationBlob>(Allocator.Persistent);

                    AddComponent(entity, new AttractorConfigSingleton { BlobRef = blobRef });
                }
            }
        }
    }
}
