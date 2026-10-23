// HLSL Compute Shader: GPU-Driven Culling & Indirect Argument Generation

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

cbuffer CullConstants : register(b0)
{
    float4x4 g_ViewProjMatrix;
    float4   g_FrustumPlanes[6];
    uint     g_TotalParticleCount;
    uint     g_IndexCountPerQuad; // 보통 6
};

StructuredBuffer<GPUParticle> g_ParticleBuffer : register(t0);

RWStructuredBuffer<uint> g_IndirectDrawArgs     : register(u0);
RWStructuredBuffer<uint> g_CulledInstanceIndices : register(u1);

[numthreads(1, 1, 1)]
void CS_ClearIndirectArgs(uint3 DTid : SV_DispatchThreadID)
{
    g_IndirectDrawArgs[ARG_INDEX_COUNT_PER_INSTANCE] = g_IndexCountPerQuad;
    g_IndirectDrawArgs[ARG_INSTANCE_COUNT]           = 0;
    g_IndirectDrawArgs[ARG_START_INDEX_LOCATION]     = 0;
    g_IndirectDrawArgs[ARG_BASE_VERTEX_LOCATION]     = 0;
    g_IndirectDrawArgs[ARG_START_INSTANCE_LOCATION]  = 0;
}

bool IsInFrustum(float3 center, float radius)
{
    [unroll]
    for (int i = 0; i < 6; ++i)
    {
        if (dot(g_FrustumPlanes[i].xyz, center) + g_FrustumPlanes[i].w < -radius)
        {
            return false;
        }
    }
    return true;
}

[numthreads(256, 1, 1)]
void CS_CullAndBuildIndirectArgs(uint3 DTid : SV_DispatchThreadID)
{
    uint particleIdx = DTid.x;
    if (particleIdx >= g_TotalParticleCount) return;

    GPUParticle p = g_ParticleBuffer[particleIdx];

    if (p.Active == 0 || p.Scale <= 0.0001f || p.Age >= p.MaxLifetime)
    {
        return;
    }

    if (!IsInFrustum(p.Position, p.Scale * 0.707f))
    {
        return;
    }

    uint writeIndex = 0;
    InterlockedAdd(g_IndirectDrawArgs[ARG_INSTANCE_COUNT], 1, writeIndex);

    g_CulledInstanceIndices[writeIndex] = particleIdx;
}
