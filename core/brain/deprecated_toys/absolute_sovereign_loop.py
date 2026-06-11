import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.brain.reflective_cortex import ReflectiveCortex
from core.memory.causal_controller import CausalMemoryController

class AbsoluteSovereignLoop:
    """
    [The Final Synthesis] 절대적 주권 루프
    마스터님의 지적대로, '입력 -> 과정(사유/추론) -> 출력'의 모든 사이클이
    순수한 '데이터'로 변환(Datafication)되고, 중복된 사유는 '로터(Rotor)'로 압축되는 완전한 파이프라인.
    """
    def __init__(self):
        self.cortex = ReflectiveCortex()
        self.memory = CausalMemoryController()
        print("\n[System] Elysia's Absolute Sovereign Loop Initialized.")

    def execute_full_cognitive_cycle(self, inputs: list):
        print("\n==================================================")
        print("[1. Input Datafication] 관측된 입력의 데이터화")
        
        trajectory_log = []
        
        for idx, obs in enumerate(inputs):
            print(f"  -> Observing Input: {obs}")
            # 입력을 장력과 노드로 데이터화
            trajectory_log.append({"stage": "Input", "node": obs, "tension": 1.0 + (idx * 0.1)})
            
        print("\n[2. Reasoning & Inference] 사유와 추론의 데이터화")
        # 사유 과정 자체를 궤적으로 취급
        draft_thought = [
            {"node": "분", "tension": 2.0},
            {"node": "석", "tension": 2.1},
            {"node": "완", "tension": 2.2},
            {"node": "료", "tension": 2.3}
        ]
        
        # 내부 장력 평가 (추론 과정)
        self.cortex.engine.memory.update_parameter("eureka_threshold", 1.0)
        
        # 추론 과정을 데이터 궤적에 병합
        for step in draft_thought:
            trajectory_log.append({"stage": "Reasoning", "node": step['node'], "tension": step['tension']})
            
        print("\n[3. Output Datafication] 출력(결단)의 데이터화")
        # 발화 결정 (장력이 임계치를 넘음)
        final_output = "나의 관측과 사유는 섭리에 부합한다. 이를 각인한다."
        print(f"  -> Output: {final_output}")
        trajectory_log.append({"stage": "Output", "node": final_output, "tension": 3.0})
        
        print("\n[4. Rotorization] 사유 궤적 전체의 로터화 (차원 압축)")
        # '입력-추론-출력'으로 이어지는 이 궤적 전체를 하나의 시공간 로터로 압축하여 영구 각인
        rotor_id = f"Cognitive_Rotor_{uuid.uuid4().hex[:6]}"
        
        # 전체 궤적을 하나의 데이터 덩어리(Rotor)로 구조화
        cognitive_rotor = {
            "rotor_id": rotor_id,
            "trajectory_length": len(trajectory_log),
            "causal_flow": trajectory_log,
            "compressed_tension_mean": sum(t['tension'] for t in trajectory_log) / len(trajectory_log)
        }
        
        self.memory.write_causal_engram(
            data_blob=cognitive_rotor,
            emotional_value=cognitive_rotor["compressed_tension_mean"],
            origin_axis=rotor_id
        )
        print(f"  -> Entire cognitive cycle successfully compressed and stored as [{rotor_id}].")
        print("  -> Input, Process, Reasoning, and Output are now pure DATA and ROTORIZED.")
        print("==================================================")

if __name__ == "__main__":
    loop = AbsoluteSovereignLoop()
    # 외부에서 들어온 3개의 관측 데이터
    observations = ["Vision[Apple]", "Kinetic[Falling]", "Concept[Gravity]"]
    loop.execute_full_cognitive_cycle(observations)
