from core.brain.spacetime_rotor import SpacetimeRotor
from core.brain.tri_phase_transistor import TriPhaseTransistor
from core.utils.math_utils import Quaternion
import math

class ActiveFractalRotor(SpacetimeRotor):
    """
    [Phase 144: Tri-Phase Globe (지구본 프랙탈 스케일)]
    단순한 '특이점(블랙홀)'이 아니라, 내부에 삼상 인지 트랜지스터를 품고 있는 거대한 가변축입니다.
    마스터의 "가변축 x 가변축 x 가변축" 철학에 따라, 하위 로터들은 이 상위 로터에 기하학적으로 엮입니다.
    """
    def __init__(self, principle_name: str, base_frequency: float = 1.0):
        super().__init__(layer_name=f"[Operator] {principle_name}")
        self.principle_name = principle_name
        self.base_frequency = base_frequency
        
        # 지구본의 '시공간 상수축 (Spacetime Constant Axis)'
        self.globe_axis = Quaternion(1.0, 0.0, 0.0, 0.0)
        
        # 원인-과정-결과의 삼상 기어박스
        self.transistor = TriPhaseTransistor(process_axis=self.globe_axis)
        
        # 하위 종속 로터들
        self.child_rotors = []

    def bind_child(self, child_rotor):
        """하위 가변축을 이 상위 지구본에 물리적으로 엮습니다(종속시킵니다)."""
        self.child_rotors.append(child_rotor)

    def exert_4d_gravity(self, memory_map: dict):
        """
        [지구본 스핀 (Globe Spin)]
        과거처럼 수만 개의 데이터를 일일이 당기거나 밀어내지 않습니다.
        상위 가변축(트랜지스터)을 한 바퀴 살짝 돌려주면, 
        여기에 엮인 모든 하위 파동들이 스핀 샌드위치 연쇄(Chain of Spin)를 타고 일제히 회전(관측)됩니다.
        """
        warped_count = 0
        logs = []
        
        # dict인지 list인지에 따라 안전하게 순회
        items = memory_map.items() if isinstance(memory_map, dict) else enumerate(memory_map)
        
        for name, node in items:
            name_str = str(name)
            if "Operator" in name_str or "Archetype" in name_str:
                continue
                
            current_q = getattr(node, 'lens_offset', getattr(node, 'get', lambda k, d: None)('lens_offset', None)) if not isinstance(node, dict) else node.get('lens_offset')
            if current_q is None:
                current_q = Quaternion(1, 0, 0, 0)
            
            # 1. 삼상 트랜지스터 연산: [원인(current_q)] -> [과정(Globe Axis)] -> [결과(Result)]
            result_wave = self.transistor.process_wave(current_q)
            
            # 2. 인지적 불일치 검사 및 해소 (Cognitive Dissonance Resolution)
            from core.brain.cognitive_dissonance_resolver import CognitiveDissonanceResolver
            new_logs = CognitiveDissonanceResolver.resolve(self)
            logs.extend(new_logs)
            
            # 3. 프랙탈 스핀 연쇄 (Fractal Spin Chain)
            if self.child_rotors:
                for child in self.child_rotors:
                    if hasattr(child, 'transistor'):
                        result_wave = child.transistor.process_wave(result_wave)
            
            # 최종 연산 결과를 노드에 갱신
            if isinstance(node, dict):
                node['lens_offset'] = result_wave
            else:
                node.lens_offset = result_wave
            warped_count += 1
            
        return warped_count, logs
