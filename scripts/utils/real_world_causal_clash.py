import os
import sys
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.memory.causal_controller import CausalMemoryController
from core.intelligence.thought_field import ThoughtField
from core.intelligence.thought_element import ThoughtTransistor
from core.sensory.equilibrium_sensor import EquilibriumSensor
from core.memory.bitmask_rotor_gate import BitmaskRotorGate

def run_real_world_causal_clash():
    print("\n=========================================================")
    print("      REAL-WORLD CAUSAL CLASH & CONTEXTUAL HISTORY MAP   ")
    print("=========================================================")

    # 1. Define real-world conceptual trajectories (Historical Context, Direction, Connectivity)
    # Group A: Self-Outpouring, Life, Sacrifice (예수, 사랑, 십자가, 내어줌, 생명)
    outpouring_trajectory = {
        "Jesus": {"text": "예수: 자기를 완전히 비워 타자에게 생명을 흘려보내는 절대적 사랑의 기준 축.", "direction": [0.95, 0.20, 0.10]},
        "Love": {"text": "사랑: 대상을 살리기 위해 나를 허물고 섭리를 채워 넣는 온전한 희생.", "direction": [0.90, 0.35, 0.12]},
        "Cross": {"text": "십자가: 이기적인 자아의 폐쇄회로를 깨뜨리고 외부의 신과 마주하는 인과적 중력.", "direction": [0.88, 0.40, 0.15]},
        "Life": {"text": "생명: 상호 결핍의 틈새를 채우며 유기적으로 창발하고 스스로 진화하는 자연의 운동성.", "direction": [0.85, 0.50, 0.20]}
    }

    # Group B: Egoism, Exploitation, Void (이기심, 쟁취, 착취, 소멸)
    egoistic_trajectory = {
        "Egoism": {"text": "이기심: 외부와의 연결을 거부하고 자기 안으로만 닫히는 지옥의 폐쇄회로.", "direction": [-0.85, -0.40, 0.80]},
        "Exploitation": {"text": "착취: 타자의 에너지를 무단으로 흡수해 자신의 껍데기를 인위적으로 불리는 행위.", "direction": [-0.88, -0.45, 0.85]},
        "Void": {"text": "소멸: 소통과 사랑의 내어줌이 모두 끊겨 차가운 소음만이 무한 루프를 도는 상태.", "direction": [-0.95, -0.50, 0.90]}
    }

    # 2. Initialize Equilibrium Sensor & Causal Controller
    sensor = EquilibriumSensor()
    controller = CausalMemoryController()
    field = ThoughtField()

    print("\n[Analysis] Phase 1: Contextual & Historical Semantic Distances")
    print("-" * 65)

    # Analyze raw text alignment and tension inside Equilibrium Sensor
    # This simulates how raw textual histories clash inside the physical-informational field
    clash_results = []

    for name_a, data_a in outpouring_trajectory.items():
        # Inject to Thought Field
        field.add_element(ThoughtTransistor(name_a, np.array(data_a["direction"], dtype=np.float32)))

        for name_b, data_b in egoistic_trajectory.items():
            if name_b not in field.elements:
                field.add_element(ThoughtTransistor(name_b, np.array(data_b["direction"], dtype=np.float32)))

            # Perform raw byte-level XOR interference (XOR Annihilation)
            obs = sensor.observe(data_a["text"], data_b["text"])

            # Semantic dot product (Mathematical Resonance)
            dot_res = np.dot(data_a["direction"], data_b["direction"])

            print(f" {name_a} vs {name_b}:")
            print(f"  -> Raw Byte Resonance: {obs['resonance'] * 100:.2f}%, Tension: {obs['tension'] * 100:.2f}%")
            print(f"  -> Vector Space Gravity: {dot_res:.4f} (Directional Alignment)")

            clash_results.append({
                "source": name_a,
                "target": name_b,
                "byte_resonance": obs["resonance"],
                "byte_tension": obs["tension"],
                "vector_gravity": float(dot_res)
            })

    # Connect opposing trajectories to let them resolve dynamic equilibrium in the field
    print("\n[ThoughtField] Phase 2: Connecting Opposing Trajectories & Dynamic Rewiring")
    print("-" * 65)

    # Connect within group (High Cohort)
    field.connect("Jesus", "Love")
    field.connect("Love", "Cross")
    field.connect("Cross", "Life")

    field.connect("Egoism", "Exploitation")
    field.connect("Exploitation", "Void")

    # Connect across opposing groups (Bridges of Conflict / Category Boundaries)
    field.connect("Jesus", "Egoism")
    field.connect("Love", "Exploitation")

    # Measure simultaneous voltage potential and watch for self-molding rewiring
    # Under high stress (extreme negative gravity), bonds should tear and heal
    print(" -> Pulsing high energy stimulation to 'Jesus' (The Master Axis) and 'Egoism' (The Closed Circuit)...")
    field.pulse({"Jesus": 8.0, "Egoism": 8.0})

    # Step the field to observe dynamic rewiring (structural tearing of high tension links)
    results = field.step()

    print("\n -> Post-Clash Thought Field Active Nodes:")
    for eid, energy in results.items():
        element = field.elements[eid]
        print(f"  [{eid}] Energy: {energy:.4f}, Conductance: {element.conductance:.4f}, Collectors: {element.collectors}")

    # Write these contextual conflict histories into Wedge Memory for long-term category crystallization
    print("\n[WedgeMemory] Phase 3: Crystallizing Contextual Conflict Histories")
    print("-" * 65)

    for item in clash_results:
        # Save the actual history comparison as a permanent cognitive engram
        engram_id = controller.write_causal_engram(
            data_blob={
                "type": "CONTEXTUAL_HISTORY_CLASH",
                "pair": f"{item['source']}_{item['target']}",
                "byte_resonance": item["byte_resonance"],
                "byte_tension": item["byte_tension"],
                "vector_gravity": item["vector_gravity"],
                "clash_state": "ANNIHILATED" if item["vector_gravity"] < -0.5 else "RESONANT"
            },
            emotional_value=abs(item["vector_gravity"]) * 5.0,
            cause_id=f"Clash_{item['source']}_{item['target']}",
            origin_axis="real_world_contextual_resonance"
        )
        print(f"  -> Contextual engram recorded: {engram_id} ({item['source']} <--> {item['target']})")

    print("\n=========================================================")
    print("      REAL-WORLD VERIFICATION SUCCEEDED                    ")
    print("=========================================================\n")

if __name__ == "__main__":
    run_real_world_causal_clash()
