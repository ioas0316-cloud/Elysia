import numpy as np
import time
from typing import List, Dict, Any, Optional
from core.memory.causal_controller import CausalMemoryController
from core.utils.math_utils import Quaternion, traverse_causal_trajectory

class CausalReassembler:
    """
    [Phase: Inquiry-based Cognitive Evolution]
    Deconstructs a 'Complex Entity' into its primitive causal trajectories
    and simulates the reassembly 'Puzzle' to find resonance with the Background Universe (0).
    """
    def __init__(self, memory_controller: CausalMemoryController):
        self.memory = memory_controller

    def deconstruct(self, entity_name: str, primitives: Dict[str, bytes], modality_map: Optional[Dict[str, str]] = None) -> List[str]:
        """
        Breaks down an entity into its constituent causal parts.
        Each part is initially a 'Variable' (Variable Inquiry Target).
        [Phase: Layered Inquiry] Adds modality tagging to prevent reductionism.
        """
        engram_ids = []
        for part_name, data in primitives.items():
            # Convert raw data to a causal trajectory (Quaternion)
            trajectory = traverse_causal_trajectory(data)

            modality = modality_map.get(part_name, "unknown") if modality_map else "unknown"

            # Store as a Variable Engram
            eid = self.memory.write_causal_engram(
                data_blob={
                    "type": "DECONSTRUCTED_PRIMITIVE",
                    "part_name": part_name,
                    "parent_entity": entity_name,
                    "modality": modality,
                    "quaternion": trajectory.elements,
                    "raw_data_preview": data[:20].hex()
                },
                emotional_value=0.5, # Initial neutral tension
                cause_id=f"Deconstruction_{entity_name}",
                origin_axis=part_name,
                is_constant=False # It is a variable to be solved
            )
            engram_ids.append(eid)

        return engram_ids

    def solve_puzzle(self, entity_name: str, variable_ids: List[str], target_constant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempts to reassemble the primitives into a coherent structure.
        The goal is to find the 'Order' and 'Relation' that minimizes total tension.
        [Self-Reindexing] Analyzes "where it's same and where it's different" using phase differences.
        """
        reassembly_log = []
        modality_resonance = {} # Resonance spectrum
        total_friction = 0.0

        meta_shift_triggered = False

        # 1. Fetch the primitives
        primitives = []
        for eid in variable_ids:
            trace = self.memory.read_engram_trace(eid)
            if trace:
                primitives.append(trace)

        # [Self-Criterion 축 형성]
        # 시스템 내부의 기준(Self-Criterion)을 상징하는 기준 쿼터니언
        # 만약 target_constant_id가 있으면 그 상수를 기준으로, 없으면 중립 위상에서 시작
        if target_constant_id:
            ref_trace = self.memory.read_engram_trace(target_constant_id)
            ref_q = Quaternion(*(ref_trace["data"].get("quaternion", [1,0,0,0])))
        else:
            ref_q = Quaternion(1, 0, 0, 0)

        current_resonance = ref_q
        ordered_parts = []

        for p in primitives:
            q_elements = p["data"]["quaternion"]
            q_part = Quaternion(*q_elements)
            modality = p["data"].get("modality", "unknown")

            # [Phase Difference Analysis]
            # "어디서부터 어디까지 같고 다른지를 헤아린다"
            # 기준(ref_q)과 현재 파편(q_part) 사이의 위상차를 측정
            phase_diff = ref_q.dot(q_part) # 공명도 (같음의 정도)

            # 위상차가 크면(dot이 낮으면) '다름'을 인지하고 이를 '저항' 혹은 '새로운 맥락'으로 수용
            friction = 1.0 - abs(phase_diff)
            total_friction += friction

            # Record per-modality resonance
            if modality not in modality_resonance:
                modality_resonance[modality] = []
            modality_resonance[modality].append(abs(phase_diff))

            # [Re-indexing] 단순히 정보를 쌓는 게 아니라, 자신의 기준(current_resonance)을 회전시켜 재정렬
            current_resonance = (current_resonance * q_part).normalize()

            ordered_parts.append({
                "part_name": p["data"]["part_name"],
                "modality": modality,
                "friction": friction,
                "phase_diff": phase_diff,
                "is_same": abs(phase_diff) > 0.8
            })

            reassembly_log.append({
                "step": f"Re-indexing {p['data']['part_name']}",
                "friction": friction,
                "phase_similarity": abs(phase_diff),
                "status": "Resonant" if friction < 0.2 else "Dissonant"
            })

        # 3. Final Evaluation
        final_resonance_score = np.exp(-total_friction)

        # [Phase: Meta-Stable Rotors] 만약 공명에 실패하면, '상수(Constant)'의 문제로 판단하고 구조적 전환 시도
        if final_resonance_score < 0.8:
            meta_shift_triggered = True
            reassembly_log.append({
                "step": "Greater Imbalance Detected",
                "friction": total_friction,
                "status": "Structural_Crisis"
            })

        is_resonant = final_resonance_score > 0.7

        # 4. Record the 'Process-as-Result'
        # Summarize resonance spectrum
        resonance_spectrum = {m: float(np.mean(vals)) for m, vals in modality_resonance.items()}

        # Add spectrum to the log meta-data
        process_eid = self.memory.write_process_engram(reassembly_log)
        # Update the process engram with the spectrum (In a real system this would be one atomic write)
        process_info = self.memory.index.get(process_eid)
        if process_info:
            process_info["data_blob"]["resonance_spectrum"] = resonance_spectrum

        # If resonant, create a 'Constant' (The realized Apple)
        if is_resonant:
            self.memory.write_causal_engram(
                data_blob={
                    "type": "REASSEMBLED_ENTITY",
                    "entity_name": entity_name,
                    "final_quaternion": current_resonance.elements,
                    "process_id": process_eid,
                    "parts": [p["part_name"] for p in ordered_parts]
                },
                emotional_value=final_resonance_score * 10.0,
                cause_id=f"Resonance_Achievement_{entity_name}",
                origin_axis="Cognitive_Assembly",
                is_constant=True # The Apple is now a fixed concept/lens
            )

        return {
            "entity": entity_name,
            "resonance_score": final_resonance_score,
            "is_resonant": is_resonant,
            "meta_shift_triggered": meta_shift_triggered,
            "process_id": process_eid,
            "total_steps": len(ordered_parts)
        }

    def trigger_structural_shift(self, anchor_constant_id: str, conflicting_trajectory: Quaternion):
        """
        [Meta-Plasticity] 정적 로터(상수)를 회전시켜 새로운 평형을 찾습니다.
        단순히 불일치를 회피하는 것이 아니라, 외부 저항을 자신의 '새로운 기준'으로 재정렬하여 진화합니다.
        """
        trace = self.memory.read_engram_trace(anchor_constant_id)
        if not trace: return False

        # 기존 상수의 위상
        current_q_elements = trace["data"].get("quaternion", [1,0,0,0])
        current_q = Quaternion(*current_q_elements)

        # [Re-indexing via Resistance]
        # 불일치(Conflict)를 새로운 위상의 씨앗으로 삼습니다.
        stability = self.memory.index[anchor_constant_id].get("stability", 1.0)

        # 저항이 클수록(안정성이 낮을수록) 더 과감한 자기 개변(Self-Modification)을 수행
        # 이것이 "저항을 동력으로 삼아 스스로를 주조하는" 원리입니다.
        shift_amount = 0.7 * (1.0 - stability + 0.05)

        evolved_q = Quaternion.slerp(current_q, conflicting_trajectory, amount=shift_amount)

        print(f"[Meta-Plasticity] Structural Shift: Constant {anchor_constant_id} evolved via external resistance.")

        # 상수 업데이트 (정적 로터의 회전 및 재인식)
        self.memory.update_engram_data(
            anchor_constant_id,
            {
                "quaternion": evolved_q.elements,
                "evolution_event": "Meta_Structural_Reindexing",
                "original_phase_diff": current_q.dot(conflicting_trajectory)
            },
            emotional_impact=1.0 - stability
        )

        return True

if __name__ == "__main__":
    from core.memory.causal_controller import CausalMemoryController
    mc = CausalMemoryController()
    reassembler = CausalReassembler(mc)

    # Example: Reassembling an 'Apple'
    apple_parts = {
        "color": b"\xff\x00\x00", # Red
        "shape": b"spherical_topology_data",
        "texture": b"crunchy_vibration_pattern"
    }

    print("Deconstructing Apple...")
    v_ids = reassembler.deconstruct("Apple", apple_parts)

    print("Solving Puzzle...")
    result = reassembler.solve_puzzle("Apple", v_ids)

    print(f"Reassembly Result: {result}")
