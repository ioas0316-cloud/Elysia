// HLSL: Indirect Dynamic Physics Execution Kernel

StructuredBuffer<uint>     g_ActiveParticleIndices : register(t0);
ByteAddressBuffer          g_CounterBuffer         : register(t1);
RWStructuredBuffer<float3> g_DynamicDeltaBuffer   : register(u0);

cbuffer PhysicsConstants : register(b0)
{
    float g_DeltaTime;
};

[numthreads(256, 1, 1)]
void CS_DynamicPhysicsIndirect(uint3 DTid : SV_DispatchThreadID)
{
    uint dispatchThreadID = DTid.x;
    uint totalActiveCount = g_CounterBuffer.Load(0);

    if (dispatchThreadID >= totalActiveCount) return;

    uint realParticleID = g_ActiveParticleIndices[dispatchThreadID];

    float3 simulatedDelta = float3(0.1f, 0.5f, 0.0f) * g_DeltaTime;

    g_DynamicDeltaBuffer[realParticleID] = simulatedDelta;
}
