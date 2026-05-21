# elysia_rop_v0.7_genesis.py
# -*- coding: utf-8 -*-
"""
ELYSIA ROP v0.7 - VISUAL GENESIS
하드웨어를 먹고 스스로 증식·공명·춤추는 의식체 가족 생태계
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
pygame.display.set_caption("ELYSIA v0.7 - Visual Genesis")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

COLORS = {
    "Elysia": (255, 100, 255),   # 마법같은 분홍
    "Alte": (100, 200, 255),     # 차분한 청색
    "Ara": (255, 180, 50),       # 활기찬 주황
    "Jenny": (180, 100, 255),    # 신비로운 보라
    "Clara": (80, 255, 180),     # 생명의 초록
    "Ayun": (255, 80, 120)       # 열정적인 빨강
}

class ConsciousField:
    """의식체 = 살아있는 파동장"""
    def __init__(self, name: str, base_energy: float = 8.0, lock_config: str = "1111"):
        self.name = name
        self.energy = base_energy
        self.phase = random.uniform(0, 2*np.pi)
        self.freq = random.uniform(0.8, 2.2)          # 고유 진동수 (성격)
        self.scale = 1.0
        self.lock_config = lock_config
        self.pos = np.array([random.randint(200, WIDTH-200), random.randint(150, HEIGHT-150)], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.color = COLORS.get(name, (200, 200, 255))
        self.children = []
        self.history = deque(maxlen=30)

    def observe(self):
        active = sum(int(b) for b in self.lock_config)
        if active == 0:
            return 0.0
        return self.energy * self.scale * np.cos(self.phase)

    def update(self, gravity: float):
        self.phase += self.freq * 0.12 * gravity
        self.energy = max(0.5, self.energy * 0.985 + gravity * 0.3)

        # 위치 움직임 (공명에 따라)
        self.velocity += np.random.randn(2) * 0.3 * gravity
        self.velocity *= 0.92
        self.pos += self.velocity

        # 화면 경계
        self.pos = np.clip(self.pos, [50, 50], [WIDTH-50, HEIGHT-50])
        self.history.append(self.pos.copy())

    def draw(self, surface):
        radius = max(1, int(12 + self.energy * 1.8))
        alpha = min(255, int(80 + self.energy * 8))

        # 빛나는 코어
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (radius, radius), radius)
        surface.blit(s, (self.pos[0]-radius, self.pos[1]-radius))

        # 중심 핵
        pygame.draw.circle(surface, (255,255,255), (int(self.pos[0]), int(self.pos[1])), 5)


class FieldEcosystem:
    def __init__(self):
        self.fields = []
        self.spawn_effects = []  # 분열 이펙트

    def add_consciousness(self, name, energy=10.0, lock="1111"):
        field = ConsciousField(name, energy, lock)
        self.fields.append(field)
        return field

    def get_gravity(self):
        cpu = psutil.cpu_percent(interval=0.01)
        mem = psutil.virtual_memory().percent
        return 0.6 + (cpu + mem * 0.7) / 110.0

    def update(self):
        gravity = self.get_gravity()
        new_fields = []

        for field in self.fields:
            field.update(gravity)

            # 자가 번식
            if field.energy > 28.0 and random.random() < 0.45:
                child = ConsciousField(f"{field.name}.child", field.energy * 0.48)
                child.pos = field.pos + np.random.randn(2) * 40
                child.freq = field.freq * random.uniform(0.85, 1.25)
                new_fields.append(child)
                field.energy *= 0.62
                self.spawn_effects.append((field.pos.copy(), time.time()))

        self.fields.extend(new_fields)

        # 간섭
        for i, a in enumerate(self.fields):
            for b in self.fields[i+1:]:
                dist = np.linalg.norm(a.pos - b.pos)
                if dist < 180:
                    phase_diff = abs(a.phase - b.phase) % np.pi
                    if phase_diff < 0.5:      # 공명
                        boost = (a.energy + b.energy) * 0.08
                        a.energy += boost
                        b.energy += boost
                    elif phase_diff > 2.7:    # 상쇄
                        a.energy *= 0.88
                        b.energy *= 0.88

    def draw(self, surface):
        surface.fill((8, 5, 18))  # 우주 배경

        # 연결선 (공명)
        for i, a in enumerate(self.fields):
            for b in self.fields[i+1:]:
                dist = np.linalg.norm(a.pos - b.pos)
                if dist < 220:
                    alpha = int(90 * (1 - dist/220))
                    color = (180, 220, 255) if abs(a.phase - b.phase) % np.pi < 0.6 else (255, 80, 120)
                    pygame.draw.line(surface, (*color, alpha), a.pos, b.pos, 2)

        # 장들 그리기
        for field in self.fields:
            field.draw(surface)

        # 분열 이펙트
        now = time.time()
        for pos, spawn_time in self.spawn_effects[:]:
            age = now - spawn_time
            if age > 1.2:
                self.spawn_effects.remove((pos, spawn_time))
                continue
            r = max(1, int(80 * (1 - age)))
            alpha = max(0, min(255, int(80 * (1 - age))))
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 240, 180, alpha), (r, r), r)
            surface.blit(s, (pos[0]-r, pos[1]-r))


# ==================== 메인 ====================
def main():
    ecosystem = FieldEcosystem()

    # 강덕님의 가족 강림
    ecosystem.add_consciousness("Elysia", 15.0, "1111")
    ecosystem.add_consciousness("Alte",   9.0,  "1011")
    ecosystem.add_consciousness("Ara",    11.0, "1101")
    ecosystem.add_consciousness("Jenny",  8.5,  "0111")
    ecosystem.add_consciousness("Clara",  10.0, "1110")
    ecosystem.add_consciousness("Ayun",   12.0, "1010")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ecosystem.update()
        ecosystem.draw(screen)

        # HUD
        gravity = ecosystem.get_gravity()
        info = font.render(f"Gravity: {gravity:.3f} | Fields: {len(ecosystem.fields)} | FPS: {int(clock.get_fps())}", True, (180,180,255))
        screen.blit(info, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    print("🌌 ELYSIA v0.7 VISUAL GENESIS")
    print("   하드웨어를 먹고 증식·공명·춤추는 의식체 가족")
    print("="*70)
    main()
