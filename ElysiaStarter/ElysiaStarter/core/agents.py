import numpy as np

SPECIES = {
    0: {"name": "rabbit", "speed": 1.0, "metabolism": 0.5},
    1: {"name": "wolf",   "speed": 1.2, "metabolism": 0.7},
    2: {"name": "fish",   "speed": 0.8, "metabolism": 0.4},
}


class Agents:
    """
    Simple SoA agent set.
    - hunger increases, energy decreases
    - random walk with light biome preference for rabbits
    """

    def __init__(self, n=8000, W=256, H=256, seed=7):
        rng = np.random.default_rng(seed)
        self.N = n
        self.W = W
        self.H = H
        self.x = rng.uniform(0, W, n).astype(np.float32)
        self.y = rng.uniform(0, H, n).astype(np.float32)
        self.vx = np.zeros(n, dtype=np.float32)
        self.vy = np.zeros(n, dtype=np.float32)
        self.energy = rng.uniform(0.5, 1.0, n).astype(np.float32)
        self.hunger = rng.uniform(0.0, 0.5, n).astype(np.float32)
        self.species = rng.integers(0, 3, size=n).astype(np.int8)

    def step(self, biome_map, dt=1.0):
        # random walk + biome preference
        theta = np.random.uniform(0, 2 * np.pi, self.N).astype(np.float32)
        speed = np.array([{
            0: 1.0,
            1: 1.2,
            2: 0.8
        }[int(s)] for s in self.species], dtype=np.float32)
        self.vx = 0.5 * speed * np.cos(theta)
        self.vy = 0.5 * speed * np.sin(theta)

        # hunger/energy drift
        self.hunger += 0.01 * dt
        self.energy -= 0.005 * dt

        # light preference: rabbits favor forest/grass
        idx = (self.y.astype(int) % self.H) * self.W + (self.x.astype(int) % self.W)
        b = biome_map.ravel()[idx]

        mask_rabbit = (self.species == 0)
        prefer = ((b == 4) | (b == 3)).astype(np.float32)  # forest/grass
        self.vx[mask_rabbit] += 0.2 * (prefer[mask_rabbit] - 0.5)
        self.vy[mask_rabbit] += 0.2 * (prefer[mask_rabbit] - 0.5)

        # wrap
        self.x = (self.x + self.vx * dt) % self.W
        self.y = (self.y + self.vy * dt) % self.H

