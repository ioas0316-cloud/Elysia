using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;

namespace Elysia.GameMechanics
{
    public struct PhaseJumpComponent : IComponentData
    {
        public float CurrentEnergy;    // Current phase jump energy
        public float MaxEnergy;        // Maximum capacity
        public float JumpCostBase;     // Base tunneling cost
        public float CooldownTimer;    // Cooldown duration (seconds)
        public bool  IsTunneling;      // Active tunneling status
    }

    [UpdateInGroup(typeof(SimulationSystemGroup))]
    public partial struct ElysiaPhaseJumpSystem : ISystem
    {
        public void OnUpdate(ref SystemState state)
        {
            float deltaTime = SystemAPI.Time.DeltaTime;
            float time = (float)SystemAPI.Time.ElapsedTime;

            foreach (var (transform, agent, jumpComp) in
                     SystemAPI.Query<RefRW<LocalTransform>, RefRW<MetricAgentComponent>, RefRW<PhaseJumpComponent>>())
            {
                // Decrement cooldown
                if (jumpComp.ValueRO.CooldownTimer > 0.0f)
                {
                    jumpComp.ValueRW.CooldownTimer -= deltaTime;
                    continue;
                }

                float3 pos = transform.ValueRO.Position;
                float fieldVal = ElysiaMetricPhysicsSystem.EvaluateSeparatrixField(pos, time);
                float3 normal = ElysiaMetricPhysicsSystem.EvaluateFieldGradient(pos, time);

                // Near Separatrix barrier threshold check (|f(p)| < 0.25)
                if (math.abs(fieldVal) < 0.25f)
                {
                    float barrierEnergy = math.length(normal) * 12.0f; // Local barrier threshold energy

                    // Insufficient Energy -> Elastic bounce away from wall
                    if (jumpComp.ValueRO.CurrentEnergy < barrierEnergy)
                    {
                        float3 repelForce = math.sign(fieldVal) * normal * 30.0f;
                        agent.ValueRW.Velocity += repelForce * deltaTime;
                    }
                    // Energy Sufficient -> Quantum Phase Tunneling (Phase Jump)
                    else
                    {
                        // 1. Calculate target position across the Separatrix membrane
                        float tunnelDistance = 0.8f;
                        float3 targetPos = pos - math.sign(fieldVal) * normal * tunnelDistance;

                        // 2. Consume energy & displace agent
                        jumpComp.ValueRW.CurrentEnergy -= barrierEnergy;
                        transform.ValueRW.Position = targetPos;

                        // 3. Impart directional momentum across wall
                        agent.ValueRW.Velocity = -math.sign(fieldVal) * normal * 15.0f;

                        // 4. Set tunneling state & cooldown
                        jumpComp.ValueRW.IsTunneling = true;
                        jumpComp.ValueRW.CooldownTimer = 1.5f;
                    }
                }
                else
                {
                    jumpComp.ValueRW.IsTunneling = false;
                }

                // Natural energy recovery (2.0 per sec)
                jumpComp.ValueRW.CurrentEnergy = math.min(
                    jumpComp.ValueRO.MaxEnergy,
                    jumpComp.ValueRO.CurrentEnergy + 2.0f * deltaTime
                );
            }
        }
    }
}
