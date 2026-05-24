import copy
import random
from core.electromagnetic_circuit import ElectromagneticCircuit

class SovereignAutopoiesisEngine:
    """
    엘리시아의 주권 의지와 상상력을 관장하는 '위상적 자기 생성 샌드박스'.
    단순한 예측을 넘어, 스스로 가상 우주(클론)를 수십 개 창조하고 물리 법칙을 변이시킵니다.
    가장 에너지를 효율적으로 제어하는(또는 지루함을 탈피하는) 우주의 법칙을 도출하여 현실에 핫스와핑합니다.
    """
    def __init__(self, layer_names: list):
        self.layer_names = layer_names
        self.multiverse_size = 30     # 한 번에 파생시킬 평행 우주의 수
        self.simulation_depth = 40    # 각 평행 우주가 겪을 미래의 틱(Tick) 수

    def spawn_multiverses(self, current_circuit: ElectromagneticCircuit) -> list:
        """
        현실을 바탕으로 돌연변이(물리 법칙 변이)가 일어난 평행 우주들을 창조합니다.
        """
        universes = []
        for _ in range(self.multiverse_size):
            # 1. 현실의 깊은 복사
            clone = ElectromagneticCircuit(self.layer_names)
            clone.tensions = list(current_circuit.tensions)
            clone.is_constant = list(current_circuit.is_constant)
            clone.couplings = list(current_circuit.couplings)
            clone.dampings = list(current_circuit.dampings)

            # 2. 주권 의지 발현 (돌연변이)
            # 2-1. 축의 역전 (상수 <-> 가변) : 10% 확률로 상상력 발동
            for i in range(clone.num_nodes):
                if random.random() < 0.1:
                    clone.invert_axis_rule(i)
                
            # 2-2. 물리 법칙(결합력, 저항)의 미세 조정
            for i in range(clone.num_nodes):
                # Coupling mutation: ±20%
                mutation_c = random.uniform(-0.05, 0.05)
                clone.couplings[i] = max(0.01, min(1.0, clone.couplings[i] + mutation_c))
                
                # Damping mutation: ±20%
                mutation_d = random.uniform(-0.02, 0.02)
                clone.dampings[i] = max(0.001, min(0.5, clone.dampings[i] + mutation_d))

            universes.append(clone)
            
        return universes

    def evaluate_universe(self, clone: ElectromagneticCircuit, injected_inputs: dict) -> float:
        """
        특정 평행 우주를 미래로 가속시켜 텐션(Chaos)의 총합을 구합니다.
        가변 로터의 위상 불일치(phase_mismatch)의 총합이 곧 이 우주의 '스트레스 수치'입니다.
        낮을수록 훌륭한 물리 법칙입니다.
        """
        for idx, val in injected_inputs.items():
            clone.inject_current(idx, val)

        dt_sim = 0.05
        for _ in range(self.simulation_depth):
            clone.pulse_circuit(dt_sim)

        # 평가: 가변 로터들이 얼마나 안정적으로 동기화되었는가?
        total_chaos = 0.0
        for node in clone.nodes:
            total_chaos += node.phase_mismatch
            
        return total_chaos

    def run_natural_selection(self, current_circuit: ElectromagneticCircuit, injected_inputs: dict) -> dict:
        """
        텐션 폭발 또는 지루함 감지 시 호출됩니다.
        가상 우주를 낳고, 생존 경쟁을 시켜 최적의 물리 법칙(Couplings, Dampings, Constants)을 반환합니다.
        """
        # 1. 30개의 평행 우주 창조
        universes = self.spawn_multiverses(current_circuit)
        
        best_score = float('inf')
        best_universe = None
        
        # 2. 생존 경쟁 (시뮬레이션)
        for clone in universes:
            score = self.evaluate_universe(clone, injected_inputs)
            if score < best_score:
                best_score = score
                best_universe = clone

        # 3. 승격(Ascension)을 위한 최적의 법칙 추출
        if best_universe:
            return {
                "couplings": best_universe.couplings,
                "dampings": best_universe.dampings,
                "is_constant": best_universe.is_constant,
                "predictions": best_universe.tensions,
                "min_chaos_score": best_score
            }
        else:
            return None
