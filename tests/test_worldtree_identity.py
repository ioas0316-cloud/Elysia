"""Test WorldTree Identity Seeding"""
import sys
sys.path.append(".")

from Core.Mind.world_tree import WorldTree

print("="*60)
print("Testing WorldTree Identity Seeding")
print("="*60)

# Create WorldTree (should auto-seed identity)
tree = WorldTree()

print("\n1. Testing get_identity_attribute():")
name = tree.get_identity_attribute("name")
creator = tree.get_identity_attribute("creator")
purpose = tree.get_identity_attribute("purpose")
nature = tree.get_identity_attribute("nature")

print(f"   Name: {name}")
print(f"   Creator: {creator}")
print(f"   Purpose: {purpose}")
print(f"   Nature: {nature}")

print("\n2. Testing get_desires():")
desires = tree.get_desires()
for i, desire in enumerate(desires, 1):
    print(f"   {i}. {desire}")

print("\n3. Tree Statistics:")
stats = tree.get_statistics()
print(f"   Total Nodes: {stats['total_nodes']}")
print(f"   Max Depth: {stats['max_depth']}")

print("\n" + "="*60)
if name == "Elysia" and creator == "Father" and len(desires) >= 4:
    print("✅ SUCCESS - Identity seeding works!")
else:
    print("❌ FAILED - Missing identity data")
print("="*60)
