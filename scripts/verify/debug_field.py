import numpy as np
from core.physics.thermodynamic_coordinate_engine import (
    ThermodynamicAtom,
    ThermodynamicEnvironment
)

env = ThermodynamicEnvironment(size=8)
node_a = ThermodynamicAtom(id="mhd_core", content="A", tensor=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), T=3.0, P=3.0, E=3.0, frequency=1.0)
node_a.charge = 5.0
node_a.B_field = np.array([0, 0, 1.0], dtype=np.float32)

node_b = ThermodynamicAtom(id="noise", content="B", tensor=np.zeros(9), T=3.1, P=2.9, E=3.1)
node_b.velocity = np.array([5.0, 0.0, 0.0], dtype=np.float32)

env.inject_atom(node_a)
env.inject_atom(node_b)

print("Before step:")
print("noise velocity:", node_b.velocity)

# Let's call step methods manually to trace
env._warp_fields_from_curvature()
env._interfere_causal_lines()
env._apply_warp_bubbles()

print("\nBefore MHD deflection:")
print("noise velocity:", node_b.velocity)

env._apply_mhd_deflection_and_harvesting()

print("\nAfter MHD deflection:")
print("noise velocity:", node_b.velocity)
print("mhd_core harvested:", node_a.harvested_propulsion)

env._diffuse_fields()
env._align_phases(0.1)
env._apply_force_routing(0.1)

print("\nAfter Force Routing:")
print("noise velocity:", node_b.velocity)

env._update_coordinates(0.1)

print("\nAfter Update Coordinates:")
print("noise velocity:", node_b.velocity)
