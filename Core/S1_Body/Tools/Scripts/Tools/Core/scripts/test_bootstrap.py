"""Test the bootstrap understanding mechanism."""
import sys
sys.path.insert(0, "c:/Elysia")

from Core.S1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

print("🔄 Creating ElysianHeartbeat and running bootstrap...")
h = ElysianHeartbeat()

principles = h._bootstrap_understanding()

print(f"\n✨ Discovered {len(principles)} principles\n")
for p in principles[:8]:
    text = p['text'][:50]
    related = len(p['related_to'])
    print(f"  📖 {text}...")
    if related > 0:
        print(f"     ↳ Related to {related} other principles")
