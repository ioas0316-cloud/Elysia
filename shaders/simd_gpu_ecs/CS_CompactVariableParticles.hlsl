// HLSL: Wave-Optimized Stream Compaction & Dispatch Arguments Generator

StructuredBuffer<float>     g_ActivityState         : register(t0);
RWStructuredBuffer<uint>    g_ActiveParticleIndices : register(u0);
RWByteAddressBuffer         g_DispatchArgsBuffer    : register(u1);
RWByteAddressBuffer         g_CounterBuffer         : register(u2);

cbuffer CompactionConstants : register(b0)
{
    uint g_TotalParticleCount;
};

[numthreads(1, 1, 1)]
void CS_ClearDispatchArgs(uint3 DTid : SV_DispatchThreadID)
{
    g_CounterBuffer.Store(0, 0);
    g_DispatchArgsBuffer.Store3(0, uint3(0, 1, 1));
}

[numthreads(256, 1, 1)]
void CS_CompactVariableParticles(uint3 DTid : SV_DispatchThreadID)
{
    uint particleID = DTid.x;
    bool isActive = false;

    if (particleID < g_TotalParticleCount)
    {
        isActive = (g_ActivityState[particleID] > 0.0f);
    }

    uint waveActiveCount = WaveActiveCountBits(isActive);
    uint wavePrefixIndex = WavePrefixCountBits(isActive);
    uint waveBaseOffset  = 0;

    if (WaveIsFirstLane() && waveActiveCount > 0)
    {
        g_CounterBuffer.InterlockedAdd(0, waveActiveCount, waveBaseOffset);
    }
    waveBaseOffset = WaveReadLaneFirst(waveBaseOffset);

    if (isActive)
    {
        uint globalWriteIndex = waveBaseOffset + wavePrefixIndex;
        g_ActiveParticleIndices[globalWriteIndex] = particleID;
    }

    AllMemoryBarrierWithGroupSync();
    if (particleID == g_TotalParticleCount - 1)
    {
        uint totalActiveCount = g_CounterBuffer.Load(0);
        uint threadGroupCountX = (totalActiveCount + 255) / 256;
        g_DispatchArgsBuffer.Store(0, threadGroupCountX);
    }
}
