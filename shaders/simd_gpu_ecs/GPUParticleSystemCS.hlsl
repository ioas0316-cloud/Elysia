// HLSL Compute Shader: Torque-Driven GPU Particle System

struct GPUParticle
{
    float3 Position;
    float  Age;          // 현재 생존 시간
    float3 Velocity;
    float  MaxLifetime;  // 최대 수명
    float4 Color;
    float  Scale;        // 파티클 크기
    uint   Active;       // 활성화 여부 (1: Active, 0: Dead)
};

cbuffer ParticleSystemConstants : register(b0)
{
    float g_DeltaTime;
    uint  g_EntityCount;
    uint  g_MaxParticlesPerEntity; // 엔티티당 할당된 파티클 풀 크기
    float g_Time;                  // 난수 생성용 글로벌 타임
};

StructuredBuffer<float>  g_DerivedOutput   : register(t0); // 순간 출력 토크 (Tau_out)
StructuredBuffer<float>  g_AngularVelocity : register(t1); // 플라이휠 각속도 (Omega)
StructuredBuffer<float3> g_EntityPositions : register(t2); // 엔티티 월드 위치

RWStructuredBuffer<GPUParticle> g_ParticleBuffer : register(u0);

float3 Hash31(float p)
{
    float3 p3 = frac(float3(p, p, p) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.xxy + p3.yzz) * p3.zyx);
}

[numthreads(256, 1, 1)]
void CS_EmitAndSimulateParticles(uint3 DTid : SV_DispatchThreadID)
{
    uint entityIdx = DTid.x;
    if (entityIdx >= g_EntityCount) return;

    float outputTorque = g_DerivedOutput[entityIdx];
    float omega        = g_AngularVelocity[entityIdx];
    float3 entityPos   = g_EntityPositions[entityIdx];

    uint targetSpawnCount = (uint)clamp(outputTorque * 0.05f, 0.0f, (float)g_MaxParticlesPerEntity);

    float particleScale   = 0.1f + (omega * 0.02f);
    float initialSpeed    = 2.0f + (omega * 0.1f);
    float particleLife    = clamp(0.5f + (omega * 0.01f), 0.2f, 3.0f);

    uint baseParticleIdx = entityIdx * g_MaxParticlesPerEntity;

    for (uint i = 0; i < g_MaxParticlesPerEntity; ++i)
    {
        uint pIdx = baseParticleIdx + i;
        GPUParticle p = g_ParticleBuffer[pIdx];

        if (p.Active == 1)
        {
            p.Age += g_DeltaTime;

            if (p.Age >= p.MaxLifetime)
            {
                p.Active = 0;
            }
            else
            {
                p.Velocity.y -= 9.8f * 0.1f * g_DeltaTime;
                p.Position += p.Velocity * g_DeltaTime;

                float alpha = 1.0f - (p.Age / p.MaxLifetime);
                p.Color.a = alpha;
            }
        }
        else if (i < targetSpawnCount && outputTorque > 1.0f)
        {
            float seed = float(pIdx) + g_Time;
            float3 rndDir = Hash31(seed) * 2.0f - 1.0f;

            p.Position    = entityPos;
            p.Velocity    = normalize(rndDir) * initialSpeed;
            p.Age         = 0.0f;
            p.MaxLifetime = particleLife;
            p.Scale       = particleScale;
            p.Active      = 1;

            float intensity = clamp(outputTorque * 0.01f, 0.0f, 1.0f);
            p.Color = float4(1.0f, 0.3f + intensity * 0.7f, 0.1f * (1.0f - intensity), 1.0f);
        }

        g_ParticleBuffer[pIdx] = p;
    }
}
