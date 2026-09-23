// HLSL Compute Shader: Motion-Vector Integrated Dynamic HZB Occlusion Culling

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

cbuffer DynamicCullConstants : register(b0)
{
    float4x4 g_CurrentViewProjMatrix;
    float4x4 g_PrevViewProjMatrix;
    float4x4 g_CurrentProjMatrix;
    float2   g_HzbScreenDimensions;
    uint     g_HzbMaxMipLevel;
    float    g_DeltaTime;
    float    g_BaseDepthBias;
    uint     g_TotalParticleCount;
    uint     g_IndexCountPerQuad;
};

StructuredBuffer<GPUParticle> g_ParticleBuffer        : register(t0);
Texture2D<float>              g_HzbTexture           : register(t1);
Texture2D<float2>             g_MotionVectorTexture  : register(t2);
SamplerState                  g_PointClampSampler     : register(s0);
SamplerState                  g_LinearClampSampler    : register(s1);

RWStructuredBuffer<uint>      g_IndirectDrawArgs     : register(u0);
RWStructuredBuffer<uint>      g_CulledInstanceIndices : register(u1);

bool IsOccludedByDynamicObjects(float3 currWorldPos, float3 particleVel, float scale, float age)
{
    if (age < (g_DeltaTime * 2.0f)) return false;

    float radius = scale * 0.707f;

    float4 clipCurr = mul(float4(currWorldPos, 1.0f), g_CurrentViewProjMatrix);
    if (clipCurr.w <= 0.0001f) return false;

    float3 ndcCurr = clipCurr.xyz / clipCurr.w;
    float2 uvCurr  = saturate(ndcCurr.xy * float2(0.5f, -0.5f) + 0.5f);

    float2 occluderMotionUV = g_MotionVectorTexture.SampleLevel(g_LinearClampSampler, uvCurr, 0).rg;

    float2 uvPrevOccluder = saturate(uvCurr - occluderMotionUV);

    float3 prevWorldPos = currWorldPos - (particleVel * g_DeltaTime);
    float4 clipPrevParticle = mul(float4(prevWorldPos, 1.0f), g_PrevViewProjMatrix);
    float3 ndcPrevParticle  = (clipPrevParticle.w > 0.0001f) ? (clipPrevParticle.xyz / clipPrevParticle.w) : ndcCurr;
    float2 uvPrevParticle   = saturate(ndcPrevParticle.xy * float2(0.5f, -0.5f) + 0.5f);

    float projRadius = abs(g_CurrentProjMatrix[0][0]) * radius / clipCurr.w;

    float2 boxMin = min(min(uvCurr, uvPrevParticle), uvPrevOccluder) - projRadius;
    float2 boxMax = max(max(uvCurr, uvPrevParticle), uvPrevOccluder) + projRadius;
    boxMin = saturate(boxMin);
    boxMax = saturate(boxMax);

    float2 boxSizePixels = (boxMax - boxMin) * g_HzbScreenDimensions;
    float maxDimensionPixels = max(boxSizePixels.x, boxSizePixels.y);
    float mipLevel = clamp(ceil(log2(max(maxDimensionPixels, 1.0f))), 0.0f, (float)g_HzbMaxMipLevel);

    float4 hzbSamples;
    hzbSamples.x = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(boxMin.x, boxMin.y), mipLevel).r;
    hzbSamples.y = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(boxMax.x, boxMin.y), mipLevel).r;
    hzbSamples.z = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(boxMin.x, boxMax.y), mipLevel).r;
    hzbSamples.w = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(boxMax.x, boxMax.y), mipLevel).r;

    float maxHzbDepth = max(max(hzbSamples.x, hzbSamples.y), max(hzbSamples.z, hzbSamples.w));

    float2 particleMotionUV = uvCurr - uvPrevParticle;
    float relativeSpeed = length(particleMotionUV - occluderMotionUV);
    float dynamicBias = g_BaseDepthBias + (relativeSpeed * 0.05f);

    float particleMinNDC_Z = min(ndcCurr.z, ndcPrevParticle.z) - (projRadius * 0.5f);

    return (particleMinNDC_Z - dynamicBias) > maxHzbDepth;
}

[numthreads(256, 1, 1)]
void CS_CullParticlesDynamicHZB(uint3 DTid : SV_DispatchThreadID)
{
    uint particleIdx = DTid.x;
    if (particleIdx >= g_TotalParticleCount) return;

    GPUParticle p = g_ParticleBuffer[particleIdx];

    if (p.Active == 0 || p.Scale <= 0.0001f || p.Age >= p.MaxLifetime)
    {
        return;
    }

    if (IsOccludedByDynamicObjects(p.Position, p.Velocity, p.Scale, p.Age))
    {
        return;
    }

    uint writeIndex = 0;
    InterlockedAdd(g_IndirectDrawArgs[ARG_INSTANCE_COUNT], 1, writeIndex);
    g_CulledInstanceIndices[writeIndex] = particleIdx;
}
