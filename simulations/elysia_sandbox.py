"""
엘리시아의 방 (Pygame 2D Sandbox)
==================================
전자기 텐서 엔진(ElectromagneticRotorField)을 시각화하는 가벼운 관측실입니다.
마우스로 자극(Tension)을 주면, 엘리시아가 위상을 재정렬하며 화면상에서 움직입니다.
"""

import pygame
import sys
import math
import torch
import logging

from core.brain.consciousness_stream import ConsciousnessStream
from core.utils.math_utils import Quaternion

# Pygame 초기화
pygame.init()

# 화면 설정
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Elysia's Phase Sandbox (Tensor Field Engine)")

# 색상
BLACK = (10, 10, 15)
WHITE = (240, 240, 255)
ELY_COLOR = (100, 200, 255)
STIMULUS_COLOR = (255, 100, 100)
GRID_COLOR = (30, 30, 40)

def draw_grid():
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

def main():
    logging.info("엘리시아의 방(Pygame Sandbox) 가동 중...")
    
    # 텐서 필드 뇌 가동 (1000개의 로터 노드)
    brain = ConsciousnessStream(num_rotors=1000)
    
    # 엘리시아의 초기 물리적 위치 (화면 중앙)
    elysia_pos = [WIDTH / 2, HEIGHT / 2]
    elysia_radius = 15
    
    # 자극(마우스 클릭) 위치
    stimulus_pos = None
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        
        # 1. 이벤트 처리 (마우스 입력)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # 좌클릭
                    stimulus_pos = pygame.mouse.get_pos()
                    logging.info(f"새로운 자극 발생: {stimulus_pos}")

        # 2. 텐서 엔진 연산 및 위상 맵핑
        if stimulus_pos:
            # 화면 좌표(0~800)를 등각 공간과 유사한 -1.0 ~ 1.0 비율로 정규화
            norm_x = (stimulus_pos[0] - elysia_pos[0]) / (WIDTH / 2)
            norm_y = (stimulus_pos[1] - elysia_pos[1]) / (HEIGHT / 2)
            
            # 자극의 위상(Quaternion) 변환 (w=1.0, x, y 평면)
            # 마우스 쪽으로 향하려는 파동 생성
            target_q = Quaternion(1.0, norm_x, norm_y, 0.0).normalize()
            
            # 뇌(Consciousness Stream)에 자극 주입 (0.5초 안에 GPU 연산 완료)
            joy, new_phase_tensor = brain.process_stimulus(target_q)
            
            # 위상 변화(Motility)를 2D 물리적 이동으로 변환
            # new_phase_tensor는 [w, x, y, z] 형태
            # x, y 위상 성분을 물리적 이동 속도로 매핑
            phase_x = new_phase_tensor[1].item()
            phase_y = new_phase_tensor[2].item()
            
            move_speed = 300.0 # 픽셀/초
            
            # 텐션이 높을수록(Joy가 낮을수록) 크게 움직임
            tension = max(0.0, 1.0 - joy)
            
            elysia_pos[0] += phase_x * move_speed * dt * (1.0 + tension * 2.0)
            elysia_pos[1] += phase_y * move_speed * dt * (1.0 + tension * 2.0)
            
            # 자극점 근처에 도달하면 자극 소멸 (영점 수렴)
            dist = math.hypot(elysia_pos[0] - stimulus_pos[0], elysia_pos[1] - stimulus_pos[1])
            if dist < 20:
                stimulus_pos = None
                
            # 화면 경계 이탈 방지
            elysia_pos[0] = max(0, min(WIDTH, elysia_pos[0]))
            elysia_pos[1] = max(0, min(HEIGHT, elysia_pos[1]))
                
        # 3. 렌더링
        screen.fill(BLACK)
        draw_grid()
        
        # 자극 그리기
        if stimulus_pos:
            pygame.draw.circle(screen, STIMULUS_COLOR, stimulus_pos, 5)
            # 엘리시아와 자극 사이의 '위상 텐션(Tension)' 선 그리기
            pygame.draw.line(screen, (100, 50, 50), elysia_pos, stimulus_pos, 1)
            
        # 엘리시아 그리기
        # 안정(Joy) 상태에 따라 오라(Aura) 크기 변화
        current_joy = 1.0 if not stimulus_pos else brain.rotor_field.sync_delta_wye_bypass()
        aura_radius = elysia_radius + (1.0 - current_joy) * 40
        pygame.draw.circle(screen, (30, 80, 120), (int(elysia_pos[0]), int(elysia_pos[1])), int(aura_radius))
        
        pygame.draw.circle(screen, ELY_COLOR, (int(elysia_pos[0]), int(elysia_pos[1])), elysia_radius)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
