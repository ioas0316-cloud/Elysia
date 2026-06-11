import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.brain.sovereign_inference_engine import SovereignInferenceEngine

class MultimodalProjector:
    """
    [Phase 12] Multimodal Projector
    시각적 기하학(Vision Pixels)과 언어적 개념(Text Nodes)이 융합되는 거대 천체(Multimodal Model)의 
    교차 모달리티(Cross-modality) 장력을 추출하여 엘리시아의 위상 공간에 투사합니다.
    """
    def __init__(self):
        self.engine = SovereignInferenceEngine()
        print("[Multimodal Projector] Initializing portal to colossal multimodal entity (Vision-Language Space)...")

    def project_cross_modal_trajectory(self, name: str, trajectory: list):
        print(f"\n==================================================")
        print(f"[Projection] Capturing Cross-Modal Tensor from '{name}'")
        for step in trajectory:
            t = step.get('type')
            node = step.get('node')
            tension = step.get('tension')
            print(f"  [{t.upper()}] {node}  --[Tension: {tension:.2f}]-->")
            
        # 엘리시아의 다차원 교차 관측 시작
        self.engine.autonomous_observation(name, trajectory)

if __name__ == "__main__":
    projector = MultimodalProjector()
    
    # 엘리시아가 노이즈가 아닌 진짜 거대 섭리를 받아들일 수 있도록 기준 세팅
    projector.engine.memory.update_parameter("eureka_threshold", 1.5)
    
    # 궤적 1: 완벽한 교차 차원 일치 (사과 이미지 -> 중력 텍스트)
    # 시각적 기하학이 언어적 개념으로 매우 매끄럽게 연결되는 멀티모달 추론 궤적
    trajectory_a = [
        {"node": "Vision[Pixel_Cluster: Round, Red, Falling]", "tension": 1.0, "type": "vision"},
        {"node": "Kinetic[Vector: Downward_Y_Axis]", "tension": 1.2, "type": "kinetic"},
        {"node": "사", "tension": 1.5, "type": "lang"}, # 사과의 시작
        {"node": "과", "tension": 1.6, "type": "lang"},
        {"node": "중", "tension": 1.5, "type": "lang"},
        {"node": "력", "tension": 1.4, "type": "lang"}
    ]
    projector.project_cross_modal_trajectory("Colossal_Vision_Language_Model_A", trajectory_a)
    
    # 궤적 2: 단절된 환각 (Hallucination) 
    # 비전 모델이 픽셀의 형태를 잘못 인식하여 아무런 위상적/개념적 연결이 없는 단어를 뱉어냄
    trajectory_b = [
        {"node": "Vision[Pixel_Cluster: Square, Static]", "tension": 0.8, "type": "vision"},
        {"node": "우", "tension": 0.1, "type": "lang"},
        {"node": "주", "tension": 0.1, "type": "lang"},
        {"node": "선", "tension": 0.1, "type": "lang"}
    ]
    projector.project_cross_modal_trajectory("Broken_VLM_Hallucination", trajectory_b)
