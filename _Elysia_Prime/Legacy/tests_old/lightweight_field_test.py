# [Genesis: 2025-12-02] Purified by Elysia
"""경량 우주 - Field만 사용"""
import sys, os, logging, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*60)
print("경량 Field Evolution (World 우회)")
print("="*60)

# World 대신 간단한 Field 시뮬레이션
class LightweightUniverse:
    def __init__(self, width=64):  # 256 → 64로 축소
        self.width = width
        self.value_field = np.zeros((width, width), dtype=np.float32)
        self.coherence_field = np.zeros((width, width), dtype=np.float32)
        self.particles = []

    def add_particle(self, x, y, energy):
        self.particles.append({'x': x, 'y': y, 'energy': energy})

    def step(self):
        # 간단한 Field 업데이트
        self.value_field *= 0.95  # Decay

        for p in self.particles:
            x, y = int(p['x']) % self.width, int(p['y']) % self.width
            self.value_field[y, x] += p['energy'] * 0.01

            # 간단한 확산
            if x > 0: self.value_field[y, x-1] += 0.005 * p['energy']
            if x < self.width-1: self.value_field[y, x+1] += 0.005 * p['energy']
            if y > 0: self.value_field[y-1, x] += 0.005 * p['energy']
            if y < self.width-1: self.value_field[y+1, x] += 0.005 * p['energy']

        # Normalize
        if self.value_field.max() > 0:
            self.value_field = np.clip(self.value_field / self.value_field.max(), 0, 1)

# 테스트
universe = LightweightUniverse(width=64)

# 500 particles
import random
for i in range(500):
    universe.add_particle(
        x=random.uniform(0, 64),
        y=random.uniform(0, 64),
        energy=random.uniform(50, 100)
    )

print(f"Spawned: {len(universe.particles)} particles")
print(f"\nEvolving 10,000 cycles...")

for cycle in range(10000):
    universe.step()
    if cycle % 2000 == 0:
        print(f"  Cycle {cycle}: max_value={universe.value_field.max():.4f}")

print(f"\n✅ Complete!")
print(f"Final max value: {universe.value_field.max():.4f}")
print(f"Strong regions: {(universe.value_field > 0.5).sum()}")