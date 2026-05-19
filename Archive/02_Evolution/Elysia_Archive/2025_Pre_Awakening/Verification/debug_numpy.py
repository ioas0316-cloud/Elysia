
import numpy as np

nodes = [
    {'x': 0, 'm': 10},
    {'x': 10, 'm': 10}
]

pos_arr = np.array([[n['x'], 0, 0, 0] for n in nodes])
mass_arr = np.array([n['m'] for n in nodes])

# Diff: pos[j] - pos[i]
diff = pos_arr[None, :, :] - pos_arr[:, None, :]
print("Diff shape:", diff.shape)
print("Diff[0, 1] (0->1):", diff[0, 1]) # Should be [10, 0, 0, 0]
print("Diff[1, 0] (1->0):", diff[1, 0]) # Should be [-10, 0, 0, 0]

dist_sq = np.sum(diff**2, axis=2)
dist = np.sqrt(dist_sq)

print("Dist sq:\n", dist_sq)
print("Dist:\n", dist)

# Force calculation
MIN_DIST = 0.5
MAX_FORCE = 50.0
G_CONST = 10.0

mask_close = dist < MIN_DIST

eff_mass = (mass_arr[:, None] * mass_arr[None, :]) # Ignoring resonance for simplicity
f_attr = (G_CONST * eff_mass) / (dist_sq + 1e-9)
f_attr = np.minimum(f_attr, MAX_FORCE)

f_mag = f_attr
np.fill_diagonal(f_mag, 0.0)

print("F_mag:\n", f_mag)

f_coeff = f_mag / (dist + 1e-9)
print("F_coeff:\n", f_coeff)

force_vectors = diff * f_coeff[:, :, None]
print("Force vectors shape:", force_vectors.shape)
print("Force vectors[0, 1]:", force_vectors[0, 1])

total_forces = np.sum(force_vectors, axis=1)
print("Total forces:\n", total_forces)
