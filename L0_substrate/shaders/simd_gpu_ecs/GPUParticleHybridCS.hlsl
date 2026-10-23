// HLSL Compute Shader: Fixed-Gear (Baked) + Variable-Gear (Dynamic Delta)

Texture2D<float4>        g_BakedVatTexture    : register(t0);
StructuredBuffer<float3> g_DynamicDeltaBuffer : register(t1);
StructuredBuffer<float>  g_ActivityStateState : register(t2);

cbuffer PlaybackConstants : register(b0)
{
    float g_NormalizedTime;    // Playback Time (0.0 ~ 1.0)
    uint  g_TotalBakedFrames;  // 사전 계산된 총 프레임 수
    uint  g_TotalParticleCount;
};

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

RWStructuredBuffer<GPUParticle> g_ParticleBuffer : register(u0);

[numthreads(256, 1, 1)]
void CS_EvaluateHybridParticles(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x;
    if (id >= g_TotalParticleCount) return;

    float activityState = g_ActivityStateState[id];

    // [Track A: 고정 기어] VAT Lerp Sampling
    float exactFrame = g_NormalizedTime * (float)(g_TotalBakedFrames - 1);
    uint frame0 = uint(floor(exactFrame)) % g_TotalBakedFrames;
    uint frame1 = (frame0 + 1) % g_TotalBakedFrames;
    float alpha = frac(exactFrame);

    float4 data0 = g_BakedVatTexture.Load(int3(id, frame0, 0));
    float4 data1 = g_BakedVatTexture.Load(int3(id, frame1, 0));
    float4 blendedVAT = lerp(data0, data1, alpha);

    float3 bakedPos   = blendedVAT.xyz;
    float  bakedScale = blendedVAT.w;

    // [Track B: 가변 기어] Dynamic Delta
    float3 dynamicDelta = float3(0.0f, 0.0f, 0.0f);

    if (activityState > 0.001f)
    {
        dynamicDelta = g_DynamicDeltaBuffer[id] * activityState;
    }

    // [Phase-Lock Blending]
    GPUParticle p = g_ParticleBuffer[id];

    p.Position = bakedPos + dynamicDelta;
    p.Scale    = bakedScale * (1.0f + activityState * 0.3f);
    p.Active   = 1;

    g_ParticleBuffer[id] = p;
}
