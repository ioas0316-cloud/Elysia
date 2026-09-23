// HLSL Compute Shader: State Machine & Transition Evaluator

cbuffer GearStateConstants : register(b0)
{
    float g_DeltaTime;
    float g_TorqueThreshold;      // 가변 기어 전환 토크 임계값
    float g_AngularAccThreshold;  // 각가속도 임계값
    float g_CooldownRate;         // 가변 기어 유지 후 복귀 속도
};

struct ParticlePhysicsInput
{
    float3 CurrentTorque;
    float3 PreviousTorque;
    float3 AngularVelocity;
    float3 PreviousAngularVelocity;
    uint   ExternalEventMask;     // 충돌/이벤트 발생 비트마스크
};

StructuredBuffer<ParticlePhysicsInput> g_PhysicsInput : register(t0);
RWStructuredBuffer<float>              g_ActivityState : register(u0);

[numthreads(256, 1, 1)]
void CS_EvaluateGearState(uint3 DTid : SV_DispatchThreadID)
{
    uint id = DTid.x;

    ParticlePhysicsInput input = g_PhysicsInput[id];
    float currentState = g_ActivityState[id];

    float3 deltaTorque = input.CurrentTorque - input.PreviousTorque;
    float dTorqueMag   = length(deltaTorque);

    float3 angularAcc  = (input.AngularVelocity - input.PreviousAngularVelocity) / max(g_DeltaTime, 0.0001f);
    float dAngAccMag   = length(angularAcc);

    bool isTorqueSpike    = dTorqueMag > g_TorqueThreshold;
    bool isAngularSpike   = dAngAccMag > g_AngularAccThreshold;
    bool isExternalImpact = (input.ExternalEventMask & 0x01u) != 0u;

    bool shouldBeVariableGear = isTorqueSpike || isAngularSpike || isExternalImpact;

    float targetState = currentState;

    if (shouldBeVariableGear)
    {
        targetState = 1.0f;
    }
    else
    {
        targetState = max(0.0f, currentState - g_CooldownRate * g_DeltaTime);
    }

    g_ActivityState[id] = targetState;
}
