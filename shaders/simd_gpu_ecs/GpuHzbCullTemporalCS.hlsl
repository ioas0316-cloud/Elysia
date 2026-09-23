// HLSL Compute Shader: Temporal Reprojection & Velocity-Aware HZB Occlusion Culling

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

cbuffer TemporalCullConstants : register(b0)
{
    float4x4 g_CurrentViewProjMatrix;
    float4x4 g_PrevViewProjMatrix;
    float4x4 g_PrevProjMatrix;
    float3   g_CameraWorldVelocity;
    float    g_DeltaTime;
    float2   g_HzbScreenDimensions;
    uint     g_HzbMaxMipLevel;
    float    g_BaseDepthBias;
    uint     g_TotalParticleCount;
    uint     g_IndexCountPerQuad;
};

StructuredBuffer<GPUParticle> g_ParticleBuffer       : register(t0);
Texture2D<float>              g_HzbTexture          : register(t1);
SamplerState                  g_PointClampSampler    : register(s0);

RWStructuredBuffer<uint>      g_IndirectDrawArgs     : register(u0);
RWStructuredBuffer<uint>      g_CulledInstanceIndices : register(u1);

bool IsOccludedWithTemporalCompensation(float3 currPos, float3 velocity, float scale, float age)
{
    if (age < (g_DeltaTime * 2.0f))
    {
        return false;
    }

    float radius = scale * 0.707f;

    float3 prevPos = currPos - (velocity * g_DeltaTime);

    float4 clipCurrPrevSpace = mul(float4(currPos, 1.0f), g_PrevViewProjMatrix);
    float4 clipPrevPrevSpace = mul(float4(prevPos, 1.0f), g_PrevViewProjMatrix);

    if (clipCurrPrevSpace.w <= 0.0001f || clipPrevPrevSpace.w <= 0.0001f)
    {
        return false;
    }

    float3 ndcCurr = clipCurrPrevSpace.xyz / clipCurrPrevSpace.w;
    float3 ndcPrev = clipPrevPrevSpace.xyz / clipPrevPrevSpace.w;

    float projRadius = abs(g_PrevProjMatrix[0][0]) * radius / clipCurrPrevSpace.w;

    float2 uvCurr = saturate(ndcCurr.xy * float2(0.5f, -0.5f) + 0.5f);
    float2 uvPrev = saturate(ndcPrev.xy * float2(0.5f, -0.5f) + 0.5f);

    float2 uvBoxMin = min(uvCurr, uvPrev) - projRadius;
    float2 uvBoxMax = max(uvCurr, uvPrev) + projRadius;
    uvBoxMin = saturate(uvBoxMin);
    uvBoxMax = saturate(uvBoxMax);

    float2 boxSizePixels = (uvBoxMax - uvBoxMin) * g_HzbScreenDimensions;
    float maxDimensionPixels = max(boxSizePixels.x, boxSizePixels.y);
    float mipLevel = clamp(ceil(log2(max(maxDimensionPixels, 1.0f))), 0.0f, (float)g_HzbMaxMipLevel);

    float4 hzbSamples;
    hzbSamples.x = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMin.x, uvBoxMin.y), mipLevel).r;
    hzbSamples.y = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMax.x, uvBoxMin.y), mipLevel).r;
    hzbSamples.z = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMin.x, uvBoxMax.y), mipLevel).r;
    hzbSamples.w = g_HzbTexture.SampleLevel(g_PointClampSampler, float2(uvBoxMax.x, uvBoxMax.y), mipLevel).r;

    float maxHzbDepth = max(max(hzbSamples.x, hzbSamples.y), max(hzbSamples.z, hzbSamples.w));

    float cameraSpeed = length(g_CameraWorldVelocity);
    float particleSpeed = length(velocity);
    float adaptiveBias = g_BaseDepthBias + (cameraSpeed + particleSpeed) * g_DeltaTime * 0.002f;

    float particleMinNDC_Z = min(ndcCurr.z, ndcPrev.z) - (projRadius * 0.5f);

    return (particleMinNDC_Z - adaptiveBias) > maxHzbDepth;
}

[numthreads(256, 1, 1)]
void CS_CullParticlesTemporalHZB(uint3 DTid : SV_DispatchThreadID)
{
    uint particleIdx = DTid.x;
    if (particleIdx >= g_TotalParticleCount) return;

    GPUParticle p = g_ParticleBuffer[particleIdx];

    if (p.Active == 0 || p.Scale <= 0.0001f || p.Age >= p.MaxLifetime)
    {
        return;
    }

    if (IsOccludedWithTemporalCompensation(p.Position, p.Velocity, p.Scale, p.Age))
    {
        return;
    }

    uint writeIndex = 0;
    InterlockedAdd(g_IndirectDrawArgs[ARG_INSTANCE_COUNT], 1, writeIndex);
    g_CulledInstanceIndices[writeIndex] = particleIdx;
}
