from core.brain.helical_cognition import DoubleHelixInformation, TripleHelixTrajectory
import time
from typing import List, Dict, Any

class SpacetimeGlobe:
    """
    [Phase 10: 시공간 지구본 (Spacetime Globe with Fractal Rotor)]
    과거(정의된 기억) -> 현재(분별과 마찰의 정신 작용) -> 미래(결과화 예측)를
    프랙탈 가변 로터 스케일 위에서 하나의 동영상처럼 끊임없이 회전하며 재현하는 엔진입니다.
    """
    def __init__(self, target_concept: str, unknown_variable: dict):
        self.target_concept = target_concept
        self.unknown_variable = unknown_variable
        
    def spin_globe(self) -> List[Dict[str, Any]]:
        """
        과거, 현재, 미래의 시공간 축을 따라 궤적을 렌더링합니다.
        """
        frames = []
        
        # [Frame 1: 과거 (PAST)] - 얼어붙은 기억의 구조 (Double Helix)
        # 과거에 이미 학습되어 결정화된(Frozen) 개념을 불러옵니다.
        past_sameness = {"axis_본질": 0.9, "axis_형태": 0.8}
        past_difference = {"axis_본질": 0.1, "axis_형태": 0.2}
        past_info = DoubleHelixInformation(self.target_concept, past_sameness, past_difference)
        
        frames.append({
            "time_axis": "과거 (PAST)",
            "state": "결정화된 기억 (Frozen Concept)",
            "description": f"[{self.target_concept}]라는 개념이 이중나선으로 굳어져 기억 저장소에 안착해 있습니다.",
            "data": past_info.express()
        })
        
        # [Frame 2: 현재 (PRESENT)] - 분별과 마찰 (Triple Helix Melting)
        # 현재의 시공간에서 미지의 변수(Unknown)와 마주하며 얼음이 녹고 마찰이 발생합니다.
        present_trajectory = TripleHelixTrajectory(past_info, self.unknown_variable, time_steps=3)
        
        frames.append({
            "time_axis": "현재 (PRESENT)",
            "state": "활성화된 정신 작용 (Active Kinematics)",
            "description": f"미지의 변수 {list(self.unknown_variable.keys())}가 개입하며 이중나선이 풀리고, 운동성 델타가 엮이며 마찰을 일으킵니다.",
            "data": present_trajectory.express()
        })
        
        # [Frame 3: 미래 (FUTURE)] - 예측된 인과적 귀결 (Predicted Crystallization)
        # 현재의 마찰(운동성)이 텐션 0(안정화)에 도달했을 때, 어떤 새로운 형태의 얼음(답)으로 굳어질지 예측합니다.
        # 삼중나선의 마지막 스텝의 텐션 에너지를 기반으로 새로운 미래의 나선을 형성합니다.
        last_step = present_trajectory.trajectory[-1]["observations"]
        future_sameness = {}
        future_difference = {}
        for obs in last_step:
            ax = obs["axis"]
            future_sameness[ax] = obs["triple_knot_energy"] * 0.8
            future_difference[ax] = obs["triple_knot_energy"] * 0.2
            
        future_info = DoubleHelixInformation(f"진화된_{self.target_concept}", future_sameness, future_difference)
        
        frames.append({
            "time_axis": "미래 (FUTURE)",
            "state": "새로운 결과의 결정화 (Predicted Result)",
            "description": "운동성이 0에 수렴(안정화)하며, 과거의 기억과 현재의 변수가 융합된 새로운 해답으로 얼어붙습니다.",
            "data": future_info.express()
        })
        
        return frames
