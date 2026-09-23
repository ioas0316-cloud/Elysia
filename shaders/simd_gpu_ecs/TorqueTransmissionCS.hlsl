// HLSL Compute Shader: Torque Transmission & Flywheel Integration Pipeline

cbuffer DynamicSystemConstants : register(b0)
{
    float g_DeltaTime;       // 프레임 경과 시간
    uint  g_EntityCount;      // 전체 엔티티 수
    float2 g_Padding;        // 16-Byte Alignment 보정
};

StructuredBuffer<float> g_FixedRatio    : register(t0); // R_fixed
StructuredBuffer<float> g_SystemInertia : register(t1); // Base Inertia

RWStructuredBuffer<float> g_CurrentRatio: register(u0); // R_var (State Update)
StructuredBuffer<float>   g_TargetRatio : register(t2); // R_target
StructuredBuffer<float>   g_ShiftSpeed  : register(t3); // Shift Speed
StructuredBuffer<float>   g_Friction    : register(t4); // Debuff Friction

StructuredBuffer<float>   g_InputTorque    : register(t5); // 외부 토크 입력
RWStructuredBuffer<float> g_BrakingTorque  : register(u1); // 피격 충격 토크 (소산용)
RWStructuredBuffer<float> g_AngularVelocity: register(u2); // 각속도 (Vital State)
StructuredBuffer<float>   g_DampingFactor  : register(t6); // 공기/유체 소산
RWStructuredBuffer<float> g_DerivedOutput  : register(u3); // 실시간 유도 데미지

[numthreads(256, 1, 1)]
void CS_TorqueTransmissionKernel(uint3 DTid : SV_DispatchThreadID)
{
    const uint id = DTid.x;

    if (id >= g_EntityCount) return;

    // [Phase 1] Coalesced Memory Fetch
    float fixedRatio   = g_FixedRatio[id];
    float inertia      = g_SystemInertia[id];
    float currentRatio = g_CurrentRatio[id];
    float targetRatio  = g_TargetRatio[id];
    float shiftSpeed   = g_ShiftSpeed[id];
    float friction     = g_Friction[id];
    float inputTorque  = g_InputTorque[id];
    float brakingTorque= g_BrakingTorque[id];
    float omega        = g_AngularVelocity[id];
    float damping      = g_DampingFactor[id];

    // [Phase 2] Variable Gear Transmission Step
    float dR = shiftSpeed * (targetRatio - currentRatio) - (friction * currentRatio);
    currentRatio = max(0.0f, currentRatio + dR * g_DeltaTime);

    g_CurrentRatio[id] = currentRatio;

    // [Phase 3] Torque Derivation & Flywheel Integration
    float outputTorque = inputTorque * fixedRatio * currentRatio;
    g_DerivedOutput[id] = outputTorque;

    float dampingTorque = damping * omega;
    float netTorque = outputTorque - brakingTorque - dampingTorque;

    float totalInertia = inertia * fixedRatio;

    float alpha = netTorque / totalInertia;

    omega = max(0.0f, omega + alpha * g_DeltaTime);

    // [Phase 4] Writeback & Impulse Reset
    g_AngularVelocity[id] = omega;
    g_BrakingTorque[id]   = 0.0f;
}
