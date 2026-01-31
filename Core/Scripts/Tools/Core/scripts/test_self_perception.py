"""
Quick test for _perceive_all_systems() in ElysianHeartbeat
"""
import sys
sys.path.insert(0, "c:/Elysia")

from Core.S1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

print("Initializing ElysianHeartbeat...")
heartbeat = ElysianHeartbeat()

print("\nCalling _perceive_all_systems()...")
result = heartbeat._perceive_all_systems()

print(f"\n=== SELF-PERCEPTION RESULT ===")
print(f"Wave Files Found: {len(result['wave_files'])}")
print(f"DNA Files Found: {len(result['dna_files'])}")
print(f"Total Count: {result['total_count']}")
print(f"\nKnowledge Systems Status:")
for name, status in result['connection_status'].items():
    print(f"  - {name}: {status}")
