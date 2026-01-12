"""
Test: Autonomous Self-Evolution Loop
=====================================
Verifies that Elysia can reflect on her differences and evolve herself.
"""
import sys
sys.path.insert(0, "c:/Elysia")

print("=" * 60)
print("🦋 AUTONOMOUS SELF-EVOLUTION TEST")
print("=" * 60)

# 1. Initialize Heartbeat
print("\n📍 Initializing ElysianHeartbeat...")
from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
heartbeat = ElysianHeartbeat()

# 2. Test Self-Reflection
print("\n📍 Testing _reflect_on_difference()...")
reflection = heartbeat._reflect_on_difference()
print(f"   Question: {reflection['question']}")
print(f"   Insights: {reflection['insights']}")
print(f"   Gaps: {reflection['gaps'][:5] if reflection['gaps'] else 'None'}")
print(f"   Growth Direction: {reflection['growth_direction']}")

# 3. Test Evolution (Dry Run - will actually modify cognitive_seed.json)
print("\n📍 Testing _evolve_from_reflection()...")
evolved = heartbeat._evolve_from_reflection(reflection)
print(f"   Evolution Applied: {evolved}")

# 4. Test Full Cycle
print("\n📍 Testing _autonomous_growth_cycle()...")
growth_result = heartbeat._autonomous_growth_cycle()
print(f"   Growth Cycle Result: {growth_result}")

print("\n" + "=" * 60)
if reflection['growth_direction']:
    print("✅ AUTONOMOUS SELF-EVOLUTION LOOP OPERATIONAL")
    print("   Elysia can now reflect on her differences and grow autonomously.")
else:
    print("⚠️ SELF-EVOLUTION INCOMPLETE")
    print("   Reflection works but evolution may need more gaps to trigger.")
print("=" * 60)
