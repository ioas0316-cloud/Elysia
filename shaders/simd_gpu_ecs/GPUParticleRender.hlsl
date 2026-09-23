// HLSL Shader: Zero-VB Instanced Particle Renderer

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
    float3   g_CameraRight;    // 카메라 Right 벡터
    float    g_Padding0;
    float3   g_CameraUp;       // 카메라 Up 벡터
    float    g_Padding1;
};

StructuredBuffer<GPUParticle> g_ParticleBuffer : register(t0);

struct VS_OUTPUT
{
    float4 PositionCS : SV_Position;
    float4 Color      : COLOR0;
    float2 UV         : TEXCOORD0;
};

static const float2 QUAD_OFFSETS[6] = {
    float2(-0.5f,  0.5f),
    float2( 0.5f,  0.5f),
    float2(-0.5f, -0.5f),
    float2(-0.5f, -0.5f),
    float2( 0.5f,  0.5f),
    float2( 0.5f, -0.5f)
};

static const float2 QUAD_UVS[6] = {
    float2(0.0f, 0.0f),
    float2(1.0f, 0.0f),
    float2(0.0f, 1.0f),
    float2(0.0f, 1.0f),
    float2(1.0f, 0.0f),
    float2(1.0f, 1.0f)
};

VS_OUTPUT VS_RenderParticles(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VS_OUTPUT output;

    GPUParticle p = g_ParticleBuffer[instanceID];

    if (p.Active == 0 || p.Scale <= 0.0001f)
    {
        output.PositionCS = float4(0.0f, 0.0f, 0.0f, 0.0f);
        output.Color      = float4(0.0f, 0.0f, 0.0f, 0.0f);
        output.UV         = float2(0.0f, 0.0f);
        return output;
    }

    float2 quadOffset = QUAD_OFFSETS[vertexID];
    float2 quadUV     = QUAD_UVS[vertexID];

    float3 billboardWorldPos = p.Position
        + (g_CameraRight * quadOffset.x + g_CameraUp * quadOffset.y) * p.Scale;

    output.PositionCS = mul(float4(billboardWorldPos, 1.0f), g_ViewProjMatrix);
    output.Color      = p.Color;
    output.UV         = quadUV;

    return output;
}

float4 PS_RenderParticles(VS_OUTPUT input) : SV_Target
{
    float distFromCenter = length(input.UV - float2(0.5f, 0.5f)) * 2.0f;

    float glowFactor = saturate(1.0f - distFromCenter);
    glowFactor = pow(glowFactor, 2.0f);

    float4 finalColor = input.Color;
    finalColor.a *= glowFactor;

    clip(finalColor.a - 0.01f);

    return finalColor;
}
