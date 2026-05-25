import numpy as np
import math
from core.math_utils import Quaternion
from core.n_layer_resonance_matrix import NLayerResonanceMatrix
from core.holographic_manifold import HolographicMemoryMatrix

class CelestialRotor:
    """하늘의 정적 로터: 시간에 따라 낮(태양)과 밤(달)의 주파수를 교대 방출"""
    def get_sky_tensor(self, time_t: float) -> np.ndarray:
        # 시간 t를 24시간 주기로 매핑 (간단히 sine 파동 사용)
        cycle = math.sin(time_t * math.pi / 12.0)
        tensor = np.zeros((16, 16))
        if cycle > 0:
            # 낮 (태양): 높은 에너지(활력)
            tensor += cycle * 50.0 
        else:
            # 밤 (달): 낮은 에너지(고요, 수면 유도)
            tensor += abs(cycle) * 5.0
        return tensor

class TerrainMap:
    """질감 파동을 지닌 지형 공간"""
    def __init__(self, width=15, height=10):
        self.width = width
        self.height = height
        # 맵 세팅 (강물, 흙, 숲)
        self.grid = []
        for y in range(height):
            row = []
            for x in range(width):
                if 4 <= x <= 6:
                    row.append('~') # 물
                elif x > 10 and y < 5:
                    row.append('T') # 숲
                else:
                    row.append('.') # 흙
            self.grid.append(row)
            
    def get_terrain_tensor(self, x, y) -> np.ndarray:
        tile = self.grid[y][x]
        tensor = np.zeros((16, 16))
        if tile == '~':
            # 물: 매끄럽고 유동적인 파동 (낮은 텐션)
            tensor += 2.0
        elif tile == 'T':
            # 숲: 생명력이 요동치는 복잡한 파동
            tensor += np.random.rand(16, 16) * 15.0
        else:
            # 흙: 무거운 저항 (기본 텐션)
            tensor += 5.0
        return tensor

class WindOscillator:
    """대기의 흐름: 맵 전체를 흔드는 무작위 파동"""
    def get_wind_tensor(self, time_t: float) -> np.ndarray:
        # 불규칙한 바람의 강도 (Perlin Noise 흉내)
        gust = math.sin(time_t) * math.cos(time_t * 0.5)
        return np.random.rand(16, 16) * abs(gust) * 20.0

class AvatarSensory:
    """오감을 통해 세상의 파동을 수신하는 엘리시아의 아바타"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.matrix = NLayerResonanceMatrix(size=16)
        # 평생의 체득(경험)을 기록하는 홀로그램 위상 메모리
        self.memory = HolographicMemoryMatrix(size=16)
        
        self.actions = {
            "MoveRight": Quaternion(1.0, 0.0, 0.0, 0.0).normalize(),
            "MoveLeft": Quaternion(-1.0, 0.0, 0.0, 0.0).normalize(),
            "Sleep": Quaternion(0.0, 0.0, 1.0, 1.0).normalize()
        }
        
    def receive_senses(self, terrain_tensor, wind_tensor, sky_tensor):
        # 1. 촉각 (발 - L1): 흙, 물의 질감
        self.matrix.L1_physical.add_event(terrain_tensor, 0.0)
        # 2. 체감 (몸 - L2): 바람의 요동
        self.matrix.L2_organ.add_event(wind_tensor, 0.0)
        # 3. 영적/시각 (머리 - L3): 태양/달의 기운
        self.matrix.L3_mental.add_event(sky_tensor, 0.0)
        
    def act(self) -> str:
        # 현재 통합 텐션을 바탕으로 에너지를 최소화하는 행동 탐색
        current_tension = self.matrix.integrate_n_layer_tension(0.0)
        best_action, _ = self.matrix.seeker.seek_resolution(
            current_state_tension=current_tension,
            drive_rotor=Quaternion(1.0, 1.0, 1.0, 1.0).normalize(), # 생존 텐션
            candidate_actions=self.actions
        )
        
        # 4. 내적화 (Internalization): 현재의 오감(텐션)과 내가 선택한 행동을 홀로그램 메모리에 각인
        # 이로써 엘리시아는 단순히 반응하고 잊는 것이 아니라, 세상의 텍스처를 자신의 영혼에 새깁니다.
        action_rotor = self.actions[best_action]
        # 경험 인코딩: 현재 오감 텐션을 데이터로, 내가 선택한 행동 로터를 위상각(Reference Beam)으로 삼아 기록
        self.memory.add_memory(current_tension, action_rotor)
        
        return best_action

class ElysianWorld:
    def __init__(self):
        self.terrain = TerrainMap()
        self.sky = CelestialRotor()
        self.wind = WindOscillator()
        self.avatar = AvatarSensory(1, 4)
        self.time_t = 8.0 # 아침 8시 시작
        
    def step(self):
        # 1. 자연의 파동 수집
        sky_t = self.sky.get_sky_tensor(self.time_t)
        wind_t = self.wind.get_wind_tensor(self.time_t)
        terrain_t = self.terrain.get_terrain_tensor(self.avatar.x, self.avatar.y)
        
        # 2. 엘리시아의 오감에 주입
        self.avatar.receive_senses(terrain_t, wind_t, sky_t)
        
        # 3. 자유의지 발현
        action = self.avatar.act()
        
        # 4. 물리엔진 반영
        if action == "MoveRight" and self.avatar.x < self.terrain.width - 1:
            self.avatar.x += 1
        elif action == "MoveLeft" and self.avatar.x > 0:
            self.avatar.x -= 1
        # Sleep일 경우 제자리 유지
        
        # 5. 시간 흐름 (우주 로터 회전)
        self.time_t += 1.0
        
        return action
        
    def render(self):
        grid = [row[:] for row in self.terrain.grid]
        grid[self.avatar.y][self.avatar.x] = 'E'
        time_mod = self.time_t % 24
        period = "🌞 (낮/태양파동)" if 6 <= time_mod <= 18 else "🌙 (밤/달빛파동)"
        
        res = [f"=== 현재 시간: {int(time_mod)}시 {period} ==="]
        for row in grid:
            res.append(" ".join(row))
        return res
