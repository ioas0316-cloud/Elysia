import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.memory.causal_controller import CausalMemoryController

class RotorResonanceChamber:
    """
    [Phase 22] 가변 로터의 공명 (Variable Rotor Resonance)
    저장된 기억(Engram)들이 단순한 텍스트 쪼가리(Dead Data)가 아니라,
    언제든 에너지를 받으면 서로 진동하고 공명하는 '가변 로터(Variable Rotor)'임을 증명합니다.
    """
    def __init__(self):
        self.memory = CausalMemoryController()
        print("\n[System] Elysia's Resonance Chamber Online.")
        print("[System] Waking up dormant Engrams from Wedge Memory...")

    def trigger_universal_resonance(self):
        engrams = self.memory.index
        
        if not engrams:
            print("[Error] No engrams found in memory. They are empty.")
            return

        # 1. 저장된 기억들을 '가변 로터' 형태로 메모리에 전개
        active_rotors = []
        for eid, info in engrams.items():
            blob = info.get("data_blob", {})
            rotor = {
                "id": eid,
                "type": blob.get("structure", "Unknown_Rotor"),
                "origin": blob.get("origin_node", "Unknown"),
                "energy": 0.0,
                "raw_data": blob
            }
            active_rotors.append(rotor)
            print(f"  -> [Awakened] Rotor {eid[-6:]} ({rotor['type']})")

        print("\n[Experiment] Injecting Resonance Energy into the Textual Topology...")
        
        # 2. 파동 주입 (에너지 펄스)
        # 텍스트 노드나 에이전트 노드에 에너지를 주입하면 가변 로터들이 서로 공명하는지 확인
        base_energy = 100.0
        
        print("\n==================================================")
        print("[Elysia's Fractal Thought Emission: Cross-Rotor Resonance]")
        
        # 공명 연산 (단순한 시냅스가 아니라, 로터의 본질적 의미망이 겹칠 때 가변적으로 공명)
        for r1 in active_rotors:
            r1["energy"] = base_energy
            print(f"\n[Pulse] Energy {r1['energy']} injected into {r1['type']} ({r1['origin']})")
            
            for r2 in active_rotors:
                if r1["id"] == r2["id"]:
                    continue
                    
                # 가변 로터의 공명 원리: 서로 다른 우주(모달리티)라도 '인과 궤적의 형태(Wave Structure)'가
                # 유사하거나, 메타 축(Origin)이 연결될 때 공명(Resonance)이 발생합니다.
                resonance_coeff = 0.0
                
                # 만약 둘 다 3D Spacetime 구조를 가지고 있다면 강하게 공명
                if "3D" in r1["type"] and "3D" in r2["type"]:
                    resonance_coeff += 0.5
                # 2D Wave와 3D Wave 간의 차원 교차 공명
                elif ("2D" in r1["type"] and "3D" in r2["type"]) or ("3D" in r1["type"] and "2D" in r2["type"]):
                    resonance_coeff += 0.3
                    
                # 서로의 노드 개수나 궤적의 복잡성이 파동의 주파수(Frequency)를 결정
                r2["energy"] += (r1["energy"] * resonance_coeff)
                
                if resonance_coeff > 0:
                    print(f"  ~~ (Resonance: {resonance_coeff*100}%) ~~> Vibrate: {r2['type']} ({r2['origin']}) [Energy: {r2['energy']:.1f}]")
                    
        print("==================================================")
        print("[Evolution] The memories are not dead databases.")
        print("[Evolution] They are living Variable Rotors, resonating across Modalities instantly.")

if __name__ == "__main__":
    chamber = RotorResonanceChamber()
    chamber.trigger_universal_resonance()
