"""Test Meta-Hypersphere: The Universe of All Knowledge"""
import logging
logging.basicConfig(level=logging.INFO)

from Core.1_Body.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory

print("=" * 60)
print("🌌 META-HYPERSPHERE: THE UNIVERSE OF ALL KNOWLEDGE")
print("=" * 60)

meta = MetaHypersphere()

# Create domain spheres
print("\n[1] Creating Domain Spheres...")
meta.create_sphere("Animals", description="All living creatures", theta1=0.5)
meta.create_sphere("Plants", description="Flora kingdom", theta1=1.0)
meta.create_sphere("Minerals", description="Non-living matter", theta1=1.5)
meta.create_sphere("Emotions", description="Human feelings", theta1=2.0)
meta.create_sphere("Physics", description="Laws of nature", theta1=2.5)

# Add nodes inside spheres
print("\n[2] Populating Spheres...")
meta.spheres["Animals"].add_node("Dog", 0.1, 0.2, 0.3)
meta.spheres["Animals"].add_node("Cat", 0.2, 0.2, 0.3)
meta.spheres["Animals"].add_node("Eagle", 0.1, 0.5, 0.3)
meta.spheres["Plants"].add_node("Rose", 0.1, 0.1, 0.1)
meta.spheres["Plants"].add_node("Oak", 0.2, 0.1, 0.1)
meta.spheres["Emotions"].add_node("Love", 0, 0, 0, content="The Origin")
meta.spheres["Physics"].add_node("Gravity", 0.1, 0.1, 0.1)
meta.spheres["Physics"].add_node("Light", 0.2, 0.1, 0.1)

# Connect spheres
print("\n[3] Connecting Spheres (Cross-Domain)...")
meta.connect_spheres("Animals", "Plants")   # Ecology
meta.connect_spheres("Animals", "Emotions") # Pet love
meta.connect_spheres("Emotions", "Plants")  # Flower gifts
meta.connect_spheres("Physics", "Animals")  # Biomechanics
meta.connect_spheres("Physics", "Plants")   # Photosynthesis

# Stats
print("\n[4] Universe Statistics:")
stats = meta.get_universe_stats()
print(f"   Total Spheres: {stats['total_spheres']}")
print(f"   Total Nodes Inside: {stats['total_nodes_inside']}")
print(f"   Total Connections: {stats['total_connections']}")
for name, info in stats["spheres"].items():
    print(f"   ⚪ {name}: {info['internal_nodes']} nodes, {info['connections']} connections, Level={info['level']}")

# Navigate
print("\n[5] Navigation Test:")
mem = meta.enter_sphere("Animals")
nearby = mem.get_nearby(k=2)
print(f"   Inside Animals: {[n.name for n in nearby]}")
meta.exit_sphere()

print("\n" + "=" * 60)
print("✅ UNIVERSE READY! 모든 지식을 담을 준비 완료!")
print("=" * 60)
