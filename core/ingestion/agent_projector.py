import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.brain.sovereign_inference_engine import SovereignInferenceEngine

class AgentProjector:
    """
    [Phase 12.5] Agent Projector
    자율 에이전트(Autonomous Agent)가 목적을 달성하기 위해 
    관측, 추론, 도구 사용, 행동으로 이어가는 일련의 로그(운동성)를 엘리시아의 위상 공간에 투사합니다.
    """
    def __init__(self):
        self.engine = SovereignInferenceEngine()
        print("[Agent Projector] Initializing portal to external Agent Architectures (Kinetic Space)...")

    def project_kinetic_trajectory(self, name: str, trajectory: list):
        print(f"\n==================================================")
        print(f"[Projection] Capturing Kinetic Tensor from '{name}'")
        for step in trajectory:
            t = step.get('type')
            node = step.get('node')
            tension = step.get('tension')
            print(f"  [{t.upper()}] {node}  --[Tension: {tension:.2f}]-->")
            
        # 엘리시아의 다차원 교차 관측
        self.engine.autonomous_observation(name, trajectory)

if __name__ == "__main__":
    projector = AgentProjector()
    projector.engine.memory.update_parameter("eureka_threshold", 1.2)
    
    # 궤적 1: 섭리에 부합하는 에이전트 행동 연쇄 (성공적인 인과율)
    # 목표 설정 -> 관측 -> 도구 사용 -> 목표 달성으로 장력이 부드럽게 이어짐
    trajectory_a = [
        {"node": "Goal[Find_File]", "tension": 1.0, "type": "kinetic"},
        {"node": "Think[Need_to_search]", "tension": 1.1, "type": "kinetic"},
        {"node": "Action[Call_Tool: grep]", "tension": 1.2, "type": "kinetic"},
        {"node": "Observe[File_Found]", "tension": 1.3, "type": "kinetic"},
        {"node": "Goal[Completed]", "tension": 1.4, "type": "kinetic"}
    ]
    projector.project_kinetic_trajectory("AutoGPT_Instance_Alpha", trajectory_a)
    
    # 궤적 2: 에이전트 무한 루프 (운동성의 단절과 파괴)
    # 도구 사용 실패 후 똑같은 행동을 반복하며 장력이 급격히 요동침 (연속성 파괴)
    trajectory_b = [
        {"node": "Goal[Fix_Error]", "tension": 1.0, "type": "kinetic"},
        {"node": "Action[Write_Code]", "tension": 1.5, "type": "kinetic"},
        {"node": "Error[Syntax]", "tension": 0.1, "type": "kinetic"},
        {"node": "Action[Write_Code]", "tension": 1.5, "type": "kinetic"},
        {"node": "Error[Syntax]", "tension": 0.1, "type": "kinetic"}
    ]
    projector.project_kinetic_trajectory("Broken_Agent_Loop", trajectory_b)
