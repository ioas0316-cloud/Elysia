using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

namespace Elysia.GameMechanics
{
    // Agent Parameters Component
    public struct MetricAgentComponent : IComponentData
    {
        public float3 Velocity;
        public float  Mass;
        public float  Energy; // Energy reserved for Phase Jump / Separatrix Tunneling
    }

    // Metric Field Physics & Riemannian Navigation System
    [UpdateInGroup(typeof(SimulationSystemGroup))]
    public partial struct ElysiaMetricPhysicsSystem : ISystem
    {
        public void OnUpdate(ref SystemState state)
        {
            float deltaTime = SystemAPI.Time.DeltaTime;
            float time = (float)SystemAPI.Time.ElapsedTime;

            foreach (var (transform, agent) in SystemAPI.Query<RefRW<LocalTransform>, RefRW<MetricAgentComponent>>())
            {
                float3 pos = transform.ValueRO.Position;

                // 1. Evaluate implicit Separatrix field f(p) & normal gradient
                float fieldVal = EvaluateSeparatrixField(pos, time);
                float3 normal = EvaluateFieldGradient(pos, time);

                // 2. Evaluate Geodesic Divergence Indicator (GDI)
                float gdi = EvaluateGDI(pos);

                // 3. Accumulate physical & Riemannian field forces
                float3 force = float3.zero;

                // [Separatrix Wall Repulsion]
                if (math.abs(fieldVal) < 0.3f)
                {
                    float repelMagnitude = 15.0f / (math.abs(fieldVal) + 0.1f);
                    force += math.sign(fieldVal) * normal * repelMagnitude;
                }

                // [GDI Dynamics: Convergence vs Divergence Drift]
                if (gdi < 0.0f)
                {
                    // Consensus Well: Acceleration inward / Reduced friction
                    agent.ValueRW.Velocity *= 1.02f;
                }
                else if (gdi > 0.0f)
                {
                    // Divergence Saddle: Lateral shear drift along Christoffel curvature
                    float3 lateralShear = math.cross(agent.ValueRO.Velocity, new float3(0, 1, 0));
                    force += lateralShear * gdi * 2.0f;
                }

                // 4. Euler-Cromer Integration step
                agent.ValueRW.Velocity += (force / agent.ValueRO.Mass) * deltaTime;

                // Max velocity clamping
                if (math.length(agent.ValueRO.Velocity) > 20.0f)
                {
                    agent.ValueRW.Velocity = math.normalize(agent.ValueRO.Velocity) * 20.0f;
                }

                // Update position
                transform.ValueRW.Position += agent.ValueRO.Velocity * deltaTime;
            }
        }

        public static float EvaluateSeparatrixField(float3 p, float time)
        {
            float3 d1 = p - new float3(-2.0f, 0.0f, 0.0f);
            float3 d2 = p - new float3(2.0f, 0.0f, 0.0f);
            float pot1 = 1.0f / (math.lengthsq(d1) + 0.1f);
            float pot2 = 1.0f / (math.lengthsq(d2) + 0.1f);
            float wave = 0.1f * math.sin(3.0f * p.x + time) * math.cos(3.0f * p.z);
            return (pot1 - pot2 + wave);
        }

        public static float3 EvaluateFieldGradient(float3 p, float time)
        {
            float h = 0.01f;
            float dx = EvaluateSeparatrixField(p + new float3(h, 0, 0), time) - EvaluateSeparatrixField(p - new float3(h, 0, 0), time);
            float dy = EvaluateSeparatrixField(p + new float3(0, h, 0), time) - EvaluateSeparatrixField(p - new float3(0, h, 0), time);
            float dz = EvaluateSeparatrixField(p + new float3(0, 0, h), time) - EvaluateSeparatrixField(p - new float3(0, 0, h), time);
            return math.normalize(new float3(dx, dy, dz));
        }

        public static float EvaluateGDI(float3 p)
        {
            return math.sin(p.x * 0.5f) * math.cos(p.z * 0.5f) * 3.0f;
        }
    }
}
