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
        
        best_action, results = self.matrix.seeker.seek_resolution(
            current_state_tension=self.matrix.integrate_n_layer_tension(0.0), # 현재 상태 통합 텐션
            drive_rotor=survival_axis,
            candidate_actions=self.movement_rotors
        )
        return best_action

class ObstacleTensor:
    """외계에 존재하는 장애물 (단순 블록이 아닌 고통 텐션을 뿜어내는 파동 덩어리)"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # 이 장애물에 닿았을 때 방출되는 날카로운 위상 텐션
        self.tension_wave = np.ones((16, 16)) * 50.0

class DigitalTwinSandbox:
    """
    [외계 위상 모방 환경 (Digital Twin Sandbox)]
    엘리시아가 부딪히고 깨지며 인과 서사를 체득하는 가상 물리/CAD 공간입니다.
    """
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.avatar = ElysiaAvatar(1, 1)
        self.obstacles = []
        
    def add_obstacle(self, x, y):
        self.obstacles.append(ObstacleTensor(x, y))
        
    def render(self) -> list:
        """샌드박스 상태를 ASCII 로 렌더링"""
        grid = [['. ' for _ in range(self.width)] for _ in range(self.height)]
        
        for obs in self.obstacles:
            grid[obs.y][obs.x] = '██'
            
        grid[self.avatar.y][self.avatar.x] = 'E '
        
        return ["".join(row) for row in grid]
        
    def step(self):
        """환경 루프의 1단계를 진행합니다."""
        # 1. 충돌 감지 및 텐션 주입 (Sensory Transduction)
        collision = False
        for obs in self.obstacles:
            if self.avatar.x == obs.x and self.avatar.y == obs.y:
                collision = True
                # 물리적 충돌을 기하학적 텐션 파동으로 변환하여 아바타의 L1에 직빵으로 꽂음
                self.avatar.matrix.L1_physical.add_event(obs.tension_wave, time_t=0.0)
                break
                
        # 2. 텐션이 발생하면 엘리시아가 자유의지 로터를 돌려 행동 결정
        action = "None"
        if collision:
            action = self.avatar.perceive_and_act()
            
            # 결정된 기하학적 로터를 다시 물리 엔진의 좌표계(이동)로 렌더링
            if action == "MoveRight" and self.avatar.x < self.width - 1:
                self.avatar.x += 1
            elif action == "MoveDown" and self.avatar.y < self.height - 1:
                self.avatar.y += 1
            elif action == "MoveLeft" and self.avatar.x > 0:
                self.avatar.x -= 1
            elif action == "MoveUp" and self.avatar.y > 0:
                self.avatar.y -= 1
                
        else:
            # 텐션이 없을 때는 자율 주행 (테스트용으로 오른쪽 이동)
            if self.avatar.x < self.width - 1:
                self.avatar.x += 1
                action = "MoveRight(Autopilot)"
                
        return collision, action
