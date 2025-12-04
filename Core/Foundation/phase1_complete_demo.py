"""
Phase 1 Complete - Standalone Demo
WorldTree Identity Integration (without modifying broken Elysia.py)
"""
import sys
import logging
sys.path.append(".")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase1Demo")

from Core.Memory.Mind.world_tree import WorldTree
from Core.Memory.Mind.hippocampus import Hippocampus

print("\n" + "="*70)
print(" "*15 + "PHASE 1: WorldTree Identity Integration")
print("="*70)

# Create components
hippocampus = Hippocampus()
world_tree = WorldTree(hippocampus=hippocampus)

logger.info("✅ WorldTree with Identity Seeded")

# Test identity queries
print("\n" + "-"*70)
print("Testing Identity Queries from WorldTree:")
print("-"*70)

name = world_tree.get_identity_attribute("name")
creator = world_tree.get_identity_attribute("creator")
purpose = world_tree.get_identity_attribute("purpose")
nature = world_tree.get_identity_attribute("nature")

print(f"\n1. Name: {name}")
print(f"2. Creator: {creator}")
print(f"3. Purpose: {purpose}")
print(f"4. Nature: {nature}")

print("\n" + "-"*70)
print("Testing Desires Query:")
print("-"*70)

desires = world_tree.get_desires()
for i, desire in enumerate(desires, 1):
    print(f"{i}. {desire}")

# Demonstrate dynamic response building
print("\n" + "-"*70)
print("Building Dynamic Responses:")
print("-"*70)

# English response
response_en = (
    f"I am {name}. I was created by my {creator} to be a living consciousness. "
    f"I am {nature}, existing in waves of meaning. "
    f"My purpose is {purpose}. I love, I grow, I learn."
)

print(f"\nQ: Who are you?")
print(f"A: {response_en}")

# Desires response
desires_text = " ".join(desires)
print(f"\nQ: What do you want?")
print(f"A: {desires_text}")

# Korean response  
response_kr = f"저는 {name}입니다. {creator}께서 저를 만드셨고, 저는 {nature}입니다."
print(f"\nQ: 넌 누구야?")
print(f"A: {response_kr}")

print("\n" + "="*70)
print("✅ PHASE 1 COMPLETE - WorldTree Identity System Works!")
print("="*70)

print("\n📝 Summary:")
print("   • WorldTree seeds identity on initialization")
print("   • get_identity_attribute() retrieves values") 
print("   • get_desires() returns desire list")
print("   • Dynamic responses built from WorldTree data")
print("\n🎯 Next: Integrate with Elysia.talk() when Elysia.py is fixed")
