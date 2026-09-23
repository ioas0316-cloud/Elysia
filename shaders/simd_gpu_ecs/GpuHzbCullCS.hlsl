// HLSL Compute Shader: Frustum + HZB Occlusion Culling Engine

struct GPUParticle
{
    float3 Position;
    float  Age;
    float3 Velocity;
    float  MaxLifetime;
    float4 Color;
    float  Scale;
    uint   Active;
};

#define ARG_INDEX_COUNT_PER_INSTANCE 0
#define ARG_INSTANCE_COUNT           1
#define ARG_START_INDEX_LOCATION     2
#define ARG_BASE_VERTEX_LOCATION     3
#define ARG_START_INSTANCE_LOCATION  4

cbuffer CullAndHzbConstants : register(b0)
{
    float4x4 g_ViewMatrix;
    float4x4 g_ProjMatrix;
    float4x4 g_ViewProjMatrix;
    float4   g_FrustumPlanes[6];
    float2   g_HzbScreenDimensions; // (Width, Height) at Mip 0
    uint     g_HzbMaxMipLevel;      // e.g., 10
    uint     g_TotalParticleCount;
    uint     g_IndexCountPerQuad;   // 6
};

StructuredBuffer<GPUParticle> g_ParticleBuffer       : register(t0);
Texture2D<float>              g_HzbTexture          : register(t1);
SamplerState                  g_PointClampSampler    : register(s0);

RWStructuredBuffer<uint>      g_IndirectDrawArgs     : register(u0);
RWStructuredBuffer<uint>      g_CulledInstanceIndices : register(u1);

bool IsOccludedByHZB(float3 worldCenter, float radius)
{
    float4 viewCenter = mul(float4(worldCenter, 1.0f), g_ViewMatrix);
    float viewNearestZ = viewCenter.z - radius;

    float4 nearClipPos = mul(float4(0.0f, 0.0f, viewNearestZ, 1.0f), g_ProjMatrix);
    float particleNearestNDC_Z = nearClipPos.z / nearClipPos.w;

    float4 clipCenter = mul(float4(worldCenter, 1.0f), g_ViewProjMatrix);
    if (clipCenter.w <= 0.0001f) return false;

    float3 ndcCenter = clipCenter.xyz / clipCenter.w;

    float2 projRadius = float2(
        abs(g_ProjMatrix[0][0]) * radius / clipCenter.w,
        abs(g_ProjMatrix[1][1]) * radius / clipCenter.w
    );

    float2 uvMin = saturate((ndcCenter.xy - projRadius) * float2(0.5f, -0.5f) + 0.5f);
    float2 uvMax = saturate((ndcCenter.xy + projRadius) * float2(0.5f, -0.5f) + 0.5f);

    float2 uvBoxMin = min(uvMin, uvMax);
    float2 uvBoxMax = max(uvMin, uvMax);

    float2 boxSizePixels = (uvBoxMax - uvBoxMin) * g_HzbScreenDimensions;
    float maxDimensionPixels = max(boxSizePixels.x, boxSizePixels.y);

    float mipLevel = clamp(ceil(log2(max(maxDimensionPixels, 1.0f))), 0.0f, (float)g_HzbMaxMipLevel);

    float4 hzbDepthSamples;
    hzbDepthSamples.x = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMin.x, uvBoxMin.y), mipLevel).r;
    hzbDepthSamples.y = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMax.x, uvBoxMin.y), mipLevel).r;
    hzbDepthSamples.z = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMin.x, uvBoxMax.y), mipLevel).r;
    hzbDepthSamples.w = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMax.x, uvBoxMax.y), mipLevel).r;

    float maxHzbDepth = max(max(hzbDepthSamples.x, hzbDepthSamples.y),
                            max(hzbDepthSamples.z, hzbDepthSamples.w));

    return particleNearestNDC_Z > maxHzbDepth;
}

[numthreads(256, 1, 1)]
void CS_CullParticlesFrustumAndHZB(uint3 DTid : SV_DispatchThreadID)
{
    uint particleIdx = DTid.x;
    if (particleIdx >= g_TotalParticleCount) return;

    GPUParticle p = g_ParticleBuffer[particleIdx];

    if (p.Active == 0 || p.Scale <= 0.0001f || p.Age >= p.MaxLifetime)
    {
        return;
    }

    float radius = p.Scale * 0.707f;
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        if (dot(g_FrustumPlanes[i].xyz, p.Position) + g_FrustumPlanes[i].w < -radius)
        {
            return;
        }
    }

    if (IsOccludedByHZB(p.Position, radius))
    {
        return;
    }

    uint writeIndex = 0;
    InterlockedAdd(g_IndirectDrawArgs[ARG_INSTANCE_COUNT], 1, writeIndex);
    g_CulledInstanceIndices[writeIndex] = particleIdx;
}
