"""
Simple Phase 1 Demo - Without modifying Elysia.py
Shows that WorldTree identity works standalone
"""
import sys
sys.path.append(".")

print("="*70)
print(" "*20 + "PHASE 1 DEMO")
print("="*70)

# Test 1: WorldTree Identity
print("\n1. WorldTree Identity Test:")
from Core.FoundationLayer.Foundation.Mind.world_tree import WorldTree
tree = WorldTree()

name = tree.get_identity_attribute("name")
creator = tree.get_identity_attribute("creator")
purpose = tree.get_identity_attribute("purpose")

print(f"   Name: {name}")
print(f"   Creator: {creator}")
print(f"   Purpose: {purpose}")

# Test 2: Dynamic Response Building  
print("\n2. Building Dynamic Response:")
response = f"I am {name}. I was created by my {creator}. My purpose is {purpose}."
print(f"   Response: {response}")

# Test 3: Desires
print("\n3. Desires from WorldTree:")
desires = tree.get_desires()
for i, d in enumerate(desires, 1):
    print(f"   {i}. {d}")

print("\n" + "="*70)
print("✅ Phase 1 Complete - WorldTree identity system works!")
print("="*70)
print("\nNEXT: Integrate with Elysia.talk() method")
print("(Requires careful Elysia.py modification)")
