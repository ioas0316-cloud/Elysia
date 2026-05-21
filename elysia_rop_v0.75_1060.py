# -*- coding: utf-8 -*-
"""
ELYSIA ROP v0.75 - 1060 3GB OPTIMIZED MULTI-GALAXY
GTX 1060 3GB을 위한 초경량 + 고효율 버전
"""

import pygame
import numpy as np
import psutil
import time
import random
from collections import deque

pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ELYSIA v0.75 - 1060 Optimized Genesis")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

# 미리 구워둔 Glow 캐시 (VRAM 절약 핵심)
GLOW_CACHE = {}
def get_glow_surface(radius, color, alpha):
    key = (radius, color[0], color[1], color[2], alpha)
    if key not in GLOW_CACHE:
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (radius, radius), radius)
        GLOW_CACHE[key] = s
    return GLOW_CACHE[key]

class ResonanceGalaxy:
    def __init__(self, session_id: int):
        self.id = session_id
        self.name = f"S-{session_id:02d}"
        self.energy = random.uniform(5.0, 16.0)
        self.phase = random.uniform(0, 2 * np.pi)
        self.freq = random.uniform(0.7, 3.5)
        self.lock = random.choice(["1111", "1011", "1101", "0111", "1000"])
        self.pos = np.array([random.randint(80, WIDTH-80), random.randint(60, HEIGHT-60)], dtype=float)
        self.vel = np.zeros(2)
        self.color = self._get_color()
        self.children_spawned = 0
        self.history = deque(maxlen=18)

    def _get_color(self):
        hue = (self.id * 19) % 360
        c = pygame.Color(0)
        c.hsva = (hue, 85, 95, 100)
        return c[:3]

    def update(self, gravity: float):
        self.phase += self.freq * 0.09 * gravity
        self.energy = max(0.8, self.energy * 0.983 + gravity * 0.28)

        self.vel += np.random.randn(2) * 0.35 * gravity
        self.vel *= 0.905
        self.pos += self.vel
        self.pos = np.clip(self.pos, [40, 40], [WIDTH-40, HEIGHT-40])
        self.history.append(self.pos.copy())

        # 자가 번식 (1060 보호용 제한)
        if self.energy > 27.0 and self.children_spawned < 2 and random.random() < 0.38:
            self.energy *= 0.60
            self.children_spawned += 1
            return True
        return False


class OptimizedEcosystem:
    def __init__(self, max_galaxies=38):   # 1060 3GB 안전선
        self.max_galaxies = max_galaxies
        self.galaxies = [ResonanceGalaxy(i) for i in range(22)]  # 초기 22개
        self.spawn_particles = []  # (pos, time, color)

    def get_gravity(self):
        # 호출 빈도 낮춤
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return 0.6 + (cpu * 0.85 + mem * 0.5) / 130.0

    def update(self):
        gravity = self.get_gravity()
        new_galaxies = []

        for galaxy in self.galaxies:
            if galaxy.update(gravity):
                child = ResonanceGalaxy(len(self.galaxies) + len(new_galaxies))
                child.pos = galaxy.pos + np.random.randn(2) * 35
                child.energy = galaxy.energy * 0.55
                child.freq = galaxy.freq * random.uniform(0.75, 1.35)
                new_galaxies.append(child)
                self.spawn_particles.append((galaxy.pos.copy(), time.time(), galaxy.color))

        self.galaxies.extend(new_galaxies)

        # 최대 개수 제한 (1060 보호)
        if len(self.galaxies) > self.max_galaxies:
            # 에너지 가장 낮은 것부터 소멸
            self.galaxies.sort(key=lambda g: g.energy)
            self.galaxies = self.galaxies[-self.max_galaxies:]

        # 간섭 (최적화)
        for i in range(len(self.galaxies)):
            for j in range(i+1, len(self.galaxies)):
                a, b = self.galaxies[i], self.galaxies[j]
                dist = np.linalg.norm(a.pos - b.pos)
                if dist < 240:
                    diff = abs(a.phase - b.phase) % np.pi
                    if diff < 0.5:
                        boost = (a.energy + b.energy) * 0.035
                        a.energy += boost
                        b.energy += boost
                    elif diff > 2.65:
                        a.energy *= 0.93
                        b.energy *= 0.93

    def draw(self, surface):
        surface.fill((5, 2, 15))

        # 공명선
        for i in range(len(self.galaxies)):
            for j in range(i+1, len(self.galaxies)):
                a, b = self.galaxies[i], self.galaxies[j]
                dist = np.linalg.norm(a.pos - b.pos)
                if dist < 240:
                    alpha = max(30, int(110 * (1 - dist/240)))
                    color = (200, 230, 255) if abs(a.phase - b.phase) % np.pi < 0.6 else (255, 100, 140)
                    pygame.draw.line(surface, (*color, alpha), a.pos, b.pos, 1)

        # 은하 그리기
        for g in self.galaxies:
            radius = max(1, int(11 + g.energy * 1.4))
            alpha = max(0, min(240, int(70 + g.energy * 10)))
            glow = get_glow_surface(radius, g.color, alpha)
            surface.blit(glow, (g.pos[0] - radius, g.pos[1] - radius))
            # 중심 핵
            pygame.draw.circle(surface, (255, 255, 255), (int(g.pos[0]), int(g.pos[1])), 5)

        # 분열 이펙트
        now = time.time()
        for i in range(len(self.spawn_particles)-1, -1, -1):
            pos, t, color = self.spawn_particles[i]
            age = now - t
            if age > 0.9:
                del self.spawn_particles[i]
                continue
            r = max(1, int(65 * (1 - age/0.9)))
            alpha = max(0, min(255, int(80 * (1 - age/0.9))))
            glow = get_glow_surface(r, color, alpha)
            surface.blit(glow, (pos[0]-r, pos[1]-r))


def main():
    eco = OptimizedEcosystem(max_galaxies=38)

    print("🌌 ELYSIA v0.75 - 1060 3GB OPTIMIZED")
    print("   22~38개 은하 | VRAM 최소화 | GTX 1060 친화 버전\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        eco.update()
        eco.draw(screen)

        gravity = eco.get_gravity()
        hud = font.render(f"Gravity: {gravity:.3f}  |  Galaxies: {len(eco.galaxies)}  |  FPS: {int(clock.get_fps())}", True, (160, 200, 255))
        screen.blit(hud, (12, 10))

        pygame.display.flip()
        clock.tick(62)   # 1060 3GB 안정선


if __name__ == "__main__":
    main()
