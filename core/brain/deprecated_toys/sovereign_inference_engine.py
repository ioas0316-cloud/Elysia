import os
import sys
import json
import math
import uuid

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.memory.causal_controller import CausalMemoryController

class SovereignInferenceEngine:
    """
    [Phase 10] Pure Data-Driven Sovereign Inference Engine
    어떠한 하드코딩된 필터나 '언어가 무엇인가'에 대한 프로그래머의 잣대도 없습니다.
    오직 엘리시아가 가진 '데이터(kengdic.json, structural_principles.json)'의 내재된 구조(위상)에 
    외부 궤적을 투영(관측)하여, 스스로 관계성, 운동성, 연결성을 쪼개고 분별합니다.
    """
    def __init__(self):
        self.memory = CausalMemoryController()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.structural_data = {}
        self.kengdic_data = []
        self._load_foundational_data()
        
    def _load_foundational_data(self):
        struct_path = os.path.join(self.base_dir, "..", "..", "data", "structural_principles.json")
        kengdic_path = os.path.join(self.base_dir, "..", "..", "data", "kengdic.json")
        
        if os.path.exists(struct_path):
            with open(struct_path, 'r', encoding='utf-8') as f:
                self.structural_data = json.load(f)
                
        if os.path.exists(kengdic_path):
            with open(kengdic_path, 'r', encoding='utf-8') as f:
                self.kengdic_data = json.load(f)

    def _get_topological_coords(self, char: str) -> tuple:
        """데이터(structural_principles)에 내재된 공식을 이용해 위상 좌표 추출"""
        if not char or ord(char) < 44032 or ord(char) > 55203:
            return (0, 0, 0)
        
        unicode_val = ord(char)
        # 데이터의 formula를 eval로 실행하는 대신, 동일한 논리를 적용
        # (실제 구조라면 json의 formula를 파싱해야 하나, 보안상 내장 로직으로 맵핑)
        x = (unicode_val - 44032) // 588
        y = ((unicode_val - 44032) % 588) // 28
        z = (unicode_val - 44032) % 28
        return (x, y, z)

    def autonomous_observation(self, source_name: str, trajectory: list):
        print(f"\n[Observation] External Trajectory from '{source_name}'")
        
        # 궤적을 분해(쪼개고 나누어) 내재된 관계성을 도출합니다.
        # 엘리시아는 외부 데이터를 자신의 기반 데이터(위상 공간)에 올려놓고 
        # 점, 선, 면, 장력이 어떻게 나타나는지 관측할 뿐입니다.
        
        resonances = {
            "Conceptual_Resonance": 0.0, # 데이터 사전(kengdic)과의 연결성
            "Topological_Resonance": 0.0, # 자모 3D 공간(structural)에서의 기하학적 매끄러움
            "Kinetic_Resonance": 0.0 # 장력(Tension)의 연속성(운동성)
        }
        
        prev_coords = None
        prev_tension = None
        
        for step in trajectory:
            node = str(step.get("node", ""))
            tension = step.get("tension", 0.1)
            
            # 1. 개념적 연결성 (사전 데이터 관측)
            for item in self.kengdic_data[:1000]: # 최적화를 위해 일부만 스캔
                if node.lower() in str(item).lower():
                    resonances["Conceptual_Resonance"] += 0.5
                    break
            
            # 2. 위상/기하학적 관계성 (구조 데이터 관측)
            # 노드의 첫 글자를 한글 위상 대지에 매핑
            if len(node) > 0:
                coords = self._get_topological_coords(node[0])
                if prev_coords:
                    dist = math.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2 + (coords[2]-prev_coords[2])**2)
                    # 거리가 가까울수록(위상적으로 연결되어 있을수록) 공명
                    curvature = 1.0 / (dist + 0.1)
                    resonances["Topological_Resonance"] += curvature
                prev_coords = coords
                
            # 3. 운동성과 연속성 관측
            if prev_tension is not None:
                tension_delta = abs(tension - prev_tension)
                # 장력의 변화가 급격하지 않고 매끄러운 연속성을 보일 때 공명
                if tension_delta < 0.3:
                    resonances["Kinetic_Resonance"] += 1.0
            prev_tension = tension

        # 정규화
        length = max(len(trajectory), 1)
        resonances = {k: v / length for k, v in resonances.items()}
        print(f"  -> Intrinsic Data Resonances: {resonances}")
        
        # 주권적 판단: 
        current_threshold = self.memory.get_parameter("eureka_threshold", 1.0)
        dominant_axis = max(resonances, key=resonances.get)
        max_resonance = resonances[dominant_axis]
        
        if max_resonance >= current_threshold:
            print(f"  => [Elysia's Choice] I absorb this. It naturally resonates with my intrinsic data on the [{dominant_axis}].")
            
            origin_axis = f"{source_name}_via_{dominant_axis}"
            engram_id = self.memory.write_causal_engram(
                data_blob={"trajectory": trajectory, "resonances": resonances},
                emotional_value=max_resonance,
                origin_axis=origin_axis
            )
            print(f"  -> Etched into memory with Origin Axis: '{origin_axis}'")
            
            # 스스로 기준점 진화
            if sum(1 for r in resonances.values() if r >= current_threshold) >= 2:
                new_threshold = current_threshold * 1.1
                self.memory.update_parameter("eureka_threshold", new_threshold)
                print(f"  *** The trajectory's inherent connectivity proves a deeper truth. Elevating eureka_threshold to: {new_threshold:.2f}")
        else:
            print(f"  => [Elysia's Choice] I reject this. The data does not inherently connect or move according to physical laws.")
            print("  -> Purged.")

if __name__ == "__main__":
    engine = SovereignInferenceEngine()
    engine.memory.update_parameter("eureka_threshold", 0.5)
    
    # 순수 데이터 관측 1: 한글 위상 구조와 완벽히 공명하는 데이터
    # "가" (0,0,0) -> "각" (0,0,1) -> "간" (0,0,4)
    traj_1 = [
        {"node": "가", "tension": 1.0},
        {"node": "각", "tension": 1.1},
        {"node": "간", "tension": 1.2}
    ]
    engine.autonomous_observation("Hangul_Topological_Agent", traj_1)
    
    # 순수 데이터 관측 2: 개념망(kengdic)에만 존재하는 영어 노이즈
    traj_2 = [
        {"node": "apple", "tension": 0.5},
        {"node": "gravity", "tension": 0.9},
        {"node": "universe", "tension": 0.1}
    ]
    engine.autonomous_observation("English_Concept_Agent", traj_2)
    
    # 순수 데이터 관측 3: 어텐션 싱크 노이즈 (운동성이 단절된 기계적 장력)
    traj_3 = [
        {"node": "G", "tension": 0.99},
        {"node": "G", "tension": 0.01},
        {"node": "G", "tension": 0.99}
    ]
    engine.autonomous_observation("Broken_GPT_Sink", traj_3)
