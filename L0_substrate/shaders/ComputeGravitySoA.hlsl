// ComputeGravitySoA.hlsl - Project Elysia
// Surface Tension Gravity Particle Update & Zero-Bank-Conflict LDS Compute Shader
#define THREAD_GROUP_SIZE 256
#define PI 3.14159265359f

struct SpatiotemporalSphere
{
    float3 centerPos;       // World Position (x, y, z)
    float  baseRadius;      // Base Radius R_0 (l=0 Monopole)
    float3 velocity;        // Sphere Velocity (vx, vy, vz)
    float  surfaceTension;  // Surface Tension T
    float4 shCoeffs_Low;   // c_1^{-1}, c_1^0, c_1^1 (Dipole) + c_2^0 (Quadrupole)
    float3 shCoeffs_High;  // c_2^2, c_3^0, c_3^3 (High-order ripples)
    uint   flags;           // Bit 0: IsInfiltrated, Bit 1: IsActive, Bit 2: HasChildren
};

struct Particle
{
    float3 position;
    float  mass;
    float3 velocity;
    float  lifetime;
};

cbuffer SimulationConstants : register(b0)
{
    uint   g_ParticleCount;
    uint   g_SphereCount;
    float  g_DeltaTime;
    float  g_PhaseDensity;    // rho_Phi (Effective phase mass density)
    float  g_DampingFactor;   // Field viscosity damping
    float  g_Epsilon;         // Singularity prevention threshold
};

// ------------------------------------------------------------------
// SoA (Structure of Arrays) LDS Allocation - Zero Bank Conflicts
// Total LDS Footprint: 256 * 16 * 4 bytes = 16,384 Bytes (16 KB)
// ------------------------------------------------------------------
groupshared float  s_centerPosX[THREAD_GROUP_SIZE];
groupshared float  s_centerPosY[THREAD_GROUP_SIZE];
groupshared float  s_centerPosZ[THREAD_GROUP_SIZE];
groupshared float  s_baseRadius[THREAD_GROUP_SIZE];
groupshared float  s_surfaceTension[THREAD_GROUP_SIZE];
groupshared float4 s_shCoeffs_Low[THREAD_GROUP_SIZE];
groupshared float3 s_shCoeffs_High[THREAD_GROUP_SIZE];
groupshared uint   s_flags[THREAD_GROUP_SIZE];

StructuredBuffer<SpatiotemporalSphere> g_Spheres   : register(t0);
RWStructuredBuffer<Particle>           g_Particles : register(u0);

[wave_size(32)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void CSMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint  groupIndex       : SV_GroupIndex)
{
    uint particleIndex = dispatchThreadID.x;
    bool isParticleValid = (particleIndex < g_ParticleCount);

    Particle p;
    if (isParticleValid)
    {
        p = g_Particles[particleIndex];
        if (p.lifetime <= 0.0f) isParticleValid = false;
    }

    float3 totalAccel = float3(0.0f, 0.0f, 0.0f);
    uint numTiles = (g_SphereCount + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

    float phaseDensity = WaveReadLaneFirst(g_PhaseDensity);
    float eps          = WaveReadLaneFirst(g_Epsilon);

    [loop]
    for (uint tile = 0; tile < numTiles; ++tile)
    {
        // 1. BANK-CONFLICT-FREE COOPERATIVE LOAD
        // Thread k writes to slot [k] -> Bank k = k % 32 (100% Conflict-Free)
        uint globalIdx = tile * THREAD_GROUP_SIZE + groupIndex;
        if (globalIdx < g_SphereCount)
        {
            SpatiotemporalSphere sphere = g_Spheres[globalIdx];
            s_centerPosX[groupIndex]     = sphere.centerPos.x;
            s_centerPosY[groupIndex]     = sphere.centerPos.y;
            s_centerPosZ[groupIndex]     = sphere.centerPos.z;
            s_baseRadius[groupIndex]     = sphere.baseRadius;
            s_surfaceTension[groupIndex] = sphere.surfaceTension;
            s_shCoeffs_Low[groupIndex]   = sphere.shCoeffs_Low;
            s_shCoeffs_High[groupIndex]  = sphere.shCoeffs_High;
            s_flags[groupIndex]          = sphere.flags;
        }

        GroupMemoryBarrierWithGroupSync();

        // 2. INNER LOOP WITH BROADCAST READS & LIVE-RANGE COMPRESSION
        if (isParticleValid)
        {
            uint currentTileCount = min(THREAD_GROUP_SIZE, g_SphereCount - tile * THREAD_GROUP_SIZE);

            [loop]
            for (uint i = 0; i < currentTileCount; ++i)
            {
                if ((s_flags[i] & 0x02) == 0) continue; // Skip inactive spheres

                float rx = p.position.x - s_centerPosX[i];
                float ry = p.position.y - s_centerPosY[i];
                float rz = p.position.z - s_centerPosZ[i];

                float distSq = mad(rx, rx, mad(ry, ry, rz * rz));
                if (distSq < eps) continue;

                float invDist = rsqrt(distSq);

                // Direction vector computation
                float nx = rx * invDist;
                float ny = ry * invDist;
                float nz = rz * invDist;

                // Trig-Free Spherical Harmonics Evaluation (l=0, l=1, l=2)
                const float k_Y1 = 0.5f * sqrt(3.0f / PI);
                float R_dyn = mad(s_shCoeffs_Low[i].x, k_Y1 * ny, s_baseRadius[i]);
                R_dyn       = mad(s_shCoeffs_Low[i].y, k_Y1 * nz, R_dyn);
                R_dyn       = mad(s_shCoeffs_Low[i].z, k_Y1 * nx, R_dyn);

                const float k_Y2 = 0.25f * sqrt(5.0f / PI);
                R_dyn       = mad(s_shCoeffs_Low[i].w, k_Y2 * (3.0f * nz * nz - 1.0f), R_dyn);
                R_dyn       = max(R_dyn, eps);

                // Surface Tension Gravity Calculation: g_mag = (T / rho_Phi) * H where H = 2 / R_dyn
                float H = 2.0f / R_dyn;
                float g_mag = (s_surfaceTension[i] / max(phaseDensity, eps)) * H;
                float dist = 1.0f / invDist;
                float disp = dist - R_dyn;
                float atten = exp(-abs(disp) * 0.5f);

                float accelMag = -g_mag * atten * sign(disp);

                totalAccel.x = mad(nx, accelMag, totalAccel.x);
                totalAccel.y = mad(ny, accelMag, totalAccel.y);
                totalAccel.z = mad(nz, accelMag, totalAccel.z);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (isParticleValid)
    {
        p.velocity += totalAccel * g_DeltaTime;
        p.velocity *= g_DampingFactor;
        p.position += p.velocity * g_DeltaTime;
        g_Particles[particleIndex] = p;
    }
}
