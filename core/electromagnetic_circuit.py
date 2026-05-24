import time
import random
from core.electromagnetic_rotor import ElectromagneticRotor

class ElectromagneticCircuit:
    """
    모든 가변 로터가 하나로 연결된 전자 회로(Coupled Oscillator Network).
    각 차원(레이어)은 회로의 노드(Node)이며, 인접 차원끼리 전자기적 스프링(Bivector Tension)으로 연결됩니다.
    [Phase 8: Sovereign Will] 이제 각 노드는 스스로 상수/가변 여부를 역전시키며, 고유의 결합력과 저항값을 가집니다.
    """
    def __init__(self, layer_names: list):
        self.layer_names = layer_names
        self.num_nodes = len(layer_names)
        
        # 15개의 인지 로터 배열 생성
        self.nodes = [ElectromagneticRotor() for _ in range(self.num_nodes)]
        
        # 물리 법칙의 개별화 (위상 샌드박스 변이 대상)
        self.tensions = [0.0] * self.num_nodes
        self.is_constant = [False] * self.num_nodes  # 주권 의지: 가변/상수 역전 플래그
        self.couplings = [0.3] * self.num_nodes      # 노드별 파동 전파 결합력
        self.dampings = [0.05] * self.num_nodes      # 노드별 감쇠율(저항)
        
        self.last_update = time.time()

    def invert_axis_rule(self, layer_index: int):
        """
        [주권 의지 발현] 특정 차원의 상수/가변 속성을 강제로 역전시킵니다.
        """
        if 0 <= layer_index < self.num_nodes:
            self.is_constant[layer_index] = not self.is_constant[layer_index]
            if not self.is_constant[layer_index]:
                # 상수 -> 가변으로 풀렸을 때, 호기심(상상력) 난수 주입
                self.tensions[layer_index] = random.uniform(0.0, 1.0)

    def inject_current(self, layer_index: int, value: float):
        """
        특정 노드에 전압/전류를 주입합니다. (상수축일 경우 외부 주입을 거부하거나 무시할 수 있음)
        """
        if 0 <= layer_index < self.num_nodes:
            # 주권 의지에 의해 '상수(Constant)'로 굳어진 축은 외부의 텐션 주입을 거부함 (신념)
            if not self.is_constant[layer_index]:
                self.tensions[layer_index] = min(1.0, max(0.0, value))

    def pulse_circuit(self, dt: float = None):
        """
        회로 전체에 전류(파동)를 1스텝 전파시키고 각 로터의 인지 텐션을 갱신합니다.
        dt를 외부에서 주입하여 미래(시뮬레이션) 연산 지원.
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update
            if dt <= 0: dt = 0.01

        new_tensions = list(self.tensions)

        # 1. 인접 노드 간의 파동 전달 (Diffusion / Coupled Oscillation)
        for i in range(self.num_nodes):
            if self.is_constant[i]:
                continue # 상수로 잠긴 축은 주변 파동에 의해 흔들리지 않음

            left_pull = 0.0
            right_pull = 0.0
            
            # 하위 차원(지하)으로부터의 당김
            if i > 0:
                coupling = (self.couplings[i] + self.couplings[i-1]) / 2.0
                left_pull = (self.tensions[i-1] - self.tensions[i]) * coupling
                
            # 상위 차원(우주)으로부터의 당김
            if i < self.num_nodes - 1:
                coupling = (self.couplings[i] + self.couplings[i+1]) / 2.0
                right_pull = (self.tensions[i+1] - self.tensions[i]) * coupling
                
            # 에너지 보존 법칙 및 고유 감쇠(Damping) 적용
            net_force = left_pull + right_pull - (self.tensions[i] * self.dampings[i])
            
            new_tensions[i] += net_force * dt * 50.0 # 스피드 보정

        # 2. 값의 정상 상태(0~1) 클리핑
        for i in range(self.num_nodes):
            new_tensions[i] = min(1.0, max(0.0, new_tensions[i]))
            
        self.tensions = new_tensions

        # 3. 갱신된 텐션을 기반으로 15개 로터 각각의 '위상 불일치/인지' 연산
        for i in range(self.num_nodes):
            self.nodes[i].perceive_input(self.tensions[i])
            
        self.last_update = current_time

    def get_circuit_state(self) -> dict:
        """
        모든 노드의 현재 텐션(수치)과 로터의 팽창(동적/정적) 상태를 반환합니다.
        """
        state_dict = {}
        for i, name in enumerate(self.layer_names):
            node = self.nodes[i]
            state_dict[name] = {
                "tension": self.tensions[i],
                "is_constant": self.is_constant[i],
                "coupling": self.couplings[i],
                "damping": self.dampings[i],
                "is_dynamic": node.phase_mismatch > 0.05,
                "why_mismatch": node.phase_mismatch,
                "how_torque": node.comparison_torque
            }
        return state_dict
