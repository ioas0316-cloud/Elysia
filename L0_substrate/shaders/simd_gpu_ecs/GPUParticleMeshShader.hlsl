// HLSL Mesh Shader: Direct Meshlet Primitive Emission for Particles

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

cbuffer CameraConstants : register(b0)
{
    float4x4 g_ViewProjMatrix;
    float3   g_CameraRight;
    float    g_Padding0;
    float3   g_CameraUp;
    float    g_Padding1;
};

StructuredBuffer<GPUParticle> g_ParticleBuffer : register(t0);

struct MS_OUTPUT
{
    float4 PositionCS : SV_Position;
    float4 Color      : COLOR0;
    float2 UV         : TEXCOORD0;
};

[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void MS_RenderParticles(
    uint gtid : SV_GroupThreadID,
    uint gid  : SV_GroupID,
    out vertices MS_OUTPUT verts[128],
    out indices uint3 triangles[64])
{
    uint particleIndex = gid * 32 + gtid;
    GPUParticle p = g_ParticleBuffer[particleIndex];

    bool isActive = (p.Active == 1 && p.Scale > 0.0001f);

    SetMeshOutputCounts(128, 64);

    uint vertBaseIdx = gtid * 4;
    uint primBaseIdx = gtid * 2;

    if (!isActive)
    {
        triangles[primBaseIdx + 0] = uint3(0, 0, 0);
        triangles[primBaseIdx + 1] = uint3(0, 0, 0);
        return;
    }

    float3 posTL = p.Position + (-g_CameraRight + g_CameraUp) * 0.5f * p.Scale;
    float3 posTR = p.Position + ( g_CameraRight + g_CameraUp) * 0.5f * p.Scale;
    float3 posBL = p.Position + (-g_CameraRight - g_CameraUp) * 0.5f * p.Scale;
    float3 posBR = p.Position + ( g_CameraRight - g_CameraUp) * 0.5f * p.Scale;

    verts[vertBaseIdx + 0].PositionCS = mul(float4(posTL, 1.0f), g_ViewProjMatrix);
    verts[vertBaseIdx + 0].Color      = p.Color;
    verts[vertBaseIdx + 0].UV         = float2(0.0f, 0.0f);

    verts[vertBaseIdx + 1].PositionCS = mul(float4(posTR, 1.0f), g_ViewProjMatrix);
    verts[vertBaseIdx + 1].Color      = p.Color;
    verts[vertBaseIdx + 1].UV         = float2(1.0f, 0.0f);

    verts[vertBaseIdx + 2].PositionCS = mul(float4(posBL, 1.0f), g_ViewProjMatrix);
    verts[vertBaseIdx + 2].Color      = p.Color;
    verts[vertBaseIdx + 2].UV         = float2(0.0f, 1.0f);

    verts[vertBaseIdx + 3].PositionCS = mul(float4(posBR, 1.0f), g_ViewProjMatrix);
    verts[vertBaseIdx + 3].Color      = p.Color;
    verts[vertBaseIdx + 3].UV         = float2(1.0f, 1.0f);

    triangles[primBaseIdx + 0] = uint3(vertBaseIdx + 0, vertBaseIdx + 1, vertBaseIdx + 2);
    triangles[primBaseIdx + 1] = uint3(vertBaseIdx + 2, vertBaseIdx + 1, vertBaseIdx + 3);
}
