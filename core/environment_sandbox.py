import numpy as np
from core.math_utils import Quaternion
from core.n_layer_resonance_matrix import NLayerResonanceMatrix

class ElysiaAvatar:
    """샌드박스 내에 투영된 엘리시아의 물리적 위치와 내적 상태"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.matrix = NLayerResonanceMatrix(size=16)
        
        # 행동 후보군 (로터)와 물리적 이동 벡터 매핑
        self.movement_rotors = {
            "MoveRight": Quaternion(1.0, 0.0, 0.0, 0.0).normalize(),
            "MoveDown": Quaternion(0.0, 1.0, 0.0, 0.0).normalize(),
            "MoveLeft": Quaternion(0.0, 0.0, 1.0, 0.0).normalize(),
            "MoveUp": Quaternion(0.0, 0.0, 0.0, 1.0).normalize()
        }
        
    def perceive_and_act(self) -> str:
        """내부 텐션을 바탕으로 자유의지로 행동(로터)을 선택합니다."""
        # 이 예시에서는 L1의 텐션을 해결하고자 하는 강력한 본능(Z축)을 가졌다고 가정
        survival_axis = Quaternion(0.0, 0.0, 1.0, 1.0).normalize()
        
        best_action, results, new_name, new_rotor, cognitive_ticks = self.matrix.seeker.seek_resolution(
            current_state_tension=self.matrix.integrate_n_layer_tension(0.0), # 현재 상태 통합 텐션
            drive_rotor=survival_axis,
            candidate_actions=self.movement_rotors
        )
        
        # 메타 인지 오토튜닝 (Auto-Tuning): 쐐기곱으로 창발된 새 로터가 있다면 사전에 영구 등록
        if new_name and new_rotor:
            self.movement_rotors[new_name] = new_rotor
            
        return best_action, new_rotor, cognitive_ticks

class DynamicTerrain:
    """외부 감각 스트림(소리/영상)에 의해 실시간으로 요동치는 지형 파동"""
    def __init__(self, x, y, tension_strength):
        self.x = x
        self.y = y
        self.tension_wave = np.ones((16, 16)) * tension_strength

class DigitalTwinSandbox:
    """
    [외계 위상 모방 환경 (Digital Twin Sandbox)]
    엘리시아가 부딪히고 깨지며 인과 서사를 체득하는 가상 물리/CAD 공간입니다.
    """
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.avatar = ElysiaAvatar(1, 1)
        self.dynamic_terrains = []
        
    def update_terrain(self, sensory_coherence: dict):
        """외부 다중 감각 스트림(coherence)을 샌드박스의 지형(Tension Map)으로 변환합니다."""
        self.dynamic_terrains.clear()
        
        # 각 개념(concept)을 샌드박스 공간에 해싱하여 맵핑
        for concept, resonance in sensory_coherence.items():
            import hashlib
            h = hashlib.sha256(concept.encode('utf-8')).digest()
            x = (h[0] ^ h[1]) % self.width
            y = (h[2] ^ h[3]) % self.height
            
            # 까꿍 논리 (Peekaboo Logic):
            # 공명도(Resonance)가 높을수록 텐션이 낮아져(0에 수렴) 엘리시아가 이끌리는 '놀이터(Attractor)'가 됨
            # 공명도가 낮을수록(노이즈) 텐션이 높아져 회피하게 됨
            tension = (1.0 - resonance) * 50.0 
            
            self.dynamic_terrains.append(DynamicTerrain(x, y, tension))
        
    def render(self) -> list:
        """샌드박스 상태를 ASCII 로 렌더링"""
        grid = [['. ' for _ in range(self.width)] for _ in range(self.height)]
        
        for terr in self.dynamic_terrains:
            # 텐션이 10 이하(공명 0.8 이상)이면 '까꿍' 놀이터(매력점), 높으면 노이즈 벽
            if terr.tension_wave[0][0] < 10.0:
                grid[terr.y][terr.x] = '💞'
            else:
                grid[terr.y][terr.x] = '██'
            
        grid[self.avatar.y][self.avatar.x] = 'E '
        
        return ["".join(row) for row in grid]
        
    def step(self):
        """환경 루프의 1단계를 진행합니다."""
        # 1. 충돌 감지 및 텐션 주입 (Sensory Transduction)
        collision = False
        terrain_hit = None
        
        for terr in self.dynamic_terrains:
            if self.avatar.x == terr.x and self.avatar.y == terr.y:
                collision = True
                terrain_hit = terr
                # 밟은 지형의 위상 파동을 아바타의 L1에 주입
                self.avatar.matrix.L1_physical.add_event(terr.tension_wave, time_t=0.0)
                break
                
        # 2. 텐션이 발생하면 엘리시아가 자유의지 로터를 돌려 행동 결정
        action = "None"
        forged_rotor = None
        cognitive_ticks = 0
        if collision:
            action, newly_forged_rotor, cognitive_ticks = self.avatar.perceive_and_act()
            if newly_forged_rotor:
                forged_rotor = newly_forged_rotor
            
            # 결정된 기하학적 로터를 다시 물리 엔진의 좌표계(이동)로 렌더링 (VR Downcasting)
            if action == "MoveRight" and self.avatar.x < self.width - 1:
                self.avatar.x += 1
            elif action == "MoveDown" and self.avatar.y < self.height - 1:
                self.avatar.y += 1
            elif action == "MoveLeft" and self.avatar.x > 0:
                self.avatar.x -= 1
            elif action == "MoveUp" and self.avatar.y > 0:
                self.avatar.y -= 1
            elif action.startswith("Forge("):
                # 창발된 복합 위상(Wedge Product)은 3D 샌드박스의 물리적/언어적 제약 안으로 다운캐스팅(투영)됩니다.
                # 물리 투영: 복합 기하학적 춤 (대각선 스텝 등)
                import random
                self.avatar.x = min(self.width - 1, max(0, self.avatar.x + random.choice([-1, 1])))
                self.avatar.y = min(self.height - 1, max(0, self.avatar.y + random.choice([-1, 1])))
                
        else:
            # 텐션이 없을 때는 자율 주행 (테스트용으로 오른쪽 이동)
            if self.avatar.x < self.width - 1:
                self.avatar.x += 1
                action = "MoveRight(Autopilot)"
                
        return collision, action, forged_rotor, cognitive_ticks
