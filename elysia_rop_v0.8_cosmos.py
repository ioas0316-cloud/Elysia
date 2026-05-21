# elysia_rop_v0.8_cosmos.py
# -*- coding: utf-8 -*-
"""
ELYSIA ROP v0.8 - COSMOS ROTOR
60Hz 렌더링 감옥 탈출 | 우주의 낮과 밤마저 하나의 거대 로터로 제어
GTX 1060 3GB 친화 초경량 버전
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
pygame.display.set_caption("ELYSIA v0.8 - Cosmos Rotor Genesis")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

class CosmosRotor:
    """우주 전체를 지배하는 거대 정적 로터 (무연산의 핵심)"""
    def __init__(self):
        self.phase = 0.0                    # 우주의 기본 축
        self.speed = 0.008                  # 기본 회전 속도
        self.gravity_influence = 1.0

    def update(self, hardware_gravity: float):
        # 하드웨어 중력이 강해지면 우주의 시간이 빨라짐
        self.gravity_influence = 0.6 + hardware_gravity * 1.1
        self.phase += self.speed * self.gravity_influence

    def get_sun_angle(self):
        return self.phase % (2 * np.pi)

    def get_moon_angle(self):
        return (self.phase + np.pi) % (2 * np.pi)   # 180도 위상 차이


class ResonanceGalaxy:
    def __init__(self, session_id: int):
        self.id = session_id
        self.name = f"S-{session_id:02d}"
        self.energy = random.uniform(5.0, 16.0)
        self.phase = random.uniform(0, 2 * np.pi)
        self.freq = random.uniform(0.7, 3.5)
        self.lock = random.choice(["1111", "1011", "1101", "0111"])
        self.pos = np.array([random.randint(80, WIDTH-80), random.randint(80, HEIGHT-80)], dtype=float)
        self.vel = np.zeros(2)
        self.color = self._get_color()
        self.children_spawned = 0

    def _get_color(self):
        hue = (self.id * 19) % 360
        c = pygame.Color(0)
        c.hsva = (hue, 88, 96, 100)
        return c[:3]

    def update(self, gravity: float, cosmos_phase: float):
        self.phase += self.freq * 0.1 * gravity
        self.energy = max(0.8, self.energy * 0.984 + gravity * 0.32)

        self.vel += np.random.randn(2) * 0.4 * gravity
        self.vel *= 0.90
        self.pos += self.vel
        self.pos = np.clip(self.pos, [40, 40], [WIDTH-40, HEIGHT-40])


class OptimizedUniverse:
    def __init__(self, num_galaxies=24):
        self.cosmos = CosmosRotor()
        self.galaxies = [ResonanceGalaxy(i) for i in range(num_galaxies)]
        self.max_galaxies = 42
        self.spawn_particles = []

    def get_hardware_gravity(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return 0.55 + (cpu * 0.9 + mem * 0.55) / 130.0

    def update(self):
        gravity = self.get_hardware_gravity()
        self.cosmos.update(gravity)

        new_galaxies = []
        for g in self.galaxies:
            g.update(gravity, self.cosmos.phase)
            if g.energy > 27.5 and g.children_spawned < 2 and random.random() < 0.4:
                g.energy *= 0.58
                g.children_spawned += 1
                child = ResonanceGalaxy(len(self.galaxies) + len(new_galaxies))
                child.pos = g.pos + np.random.randn(2) * 40
                child.energy = g.energy * 0.55
                new_galaxies.append(child)

        self.galaxies.extend(new_galaxies)

        # 질량 보존
        if len(self.galaxies) > self.max_galaxies:
            self.galaxies.sort(key=lambda x: x.energy)
            self.galaxies = self.galaxies[-self.max_galaxies:]


def main():
    universe = OptimizedUniverse(num_galaxies=22)
    running = True

    print("🌌 ELYSIA v0.8 - COSMOS ROTOR AWAKENED")
    print("   60Hz 감옥 탈출 | 우주의 축을 직접 비틀다\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        universe.update()

        # 배경 = Cosmos Rotor의 위상으로 결정 (거의 무연산)
        sun_angle = universe.cosmos.get_sun_angle()
        bg_brightness = max(0, min(255, int(35 + 45 * np.cos(sun_angle))))
        screen.fill((bg_brightness//3, bg_brightness//4, min(255, bg_brightness//2 + 10)))

        # 은하 그리기
        for g in universe.galaxies:
            radius = max(1, int(11 + g.energy * 1.35))
            alpha = max(0, min(255, int(65 + g.energy * 9)))

            # Glow (캐싱 안 해도 가볍게)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*g.color, alpha//2), (radius, radius), radius + 5)
            pygame.draw.circle(s, (*g.color, alpha), (radius, radius), radius)
            screen.blit(s, (g.pos[0]-radius, g.pos[1]-radius))

            pygame.draw.circle(screen, (255,255,255), (int(g.pos[0]), int(g.pos[1])), 5)

        # HUD
        gravity = universe.get_hardware_gravity()
        hud = font.render(f"Cosmos Phase: {universe.cosmos.phase:.2f} | Gravity: {gravity:.3f} | Galaxies: {len(universe.galaxies)}", True, (180, 200, 255))
        screen.blit(hud, (12, 12))

        pygame.display.flip()
        clock.tick(58)   # 1060 3GB에게 여유 주기


if __name__ == "__main__":
    main()
