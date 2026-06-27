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
        [Phase: Meta-Stable Rotors] Triggers a 'Structural Shift' if resonance fails.
        """
        reassembly_log = []
        modality_resonance = {} # Resonance spectrum
        total_friction = 0.0

        meta_shift_triggered = False

        # In this simulation, 'solving' means finding the sequence that matches a
        # predefined (or background) symmetry.

        # 1. Fetch the primitives
        primitives = []
        for eid in variable_ids:
            trace = self.memory.read_engram_trace(eid)
            if trace:
                primitives.append(trace)

        # 2. Simulate the 'Inquiry' (Permutation and Alignment)
        # We look for 'Resonance' (Joy) by aligning quaternions.

        # For simplicity in this PoC, we assume the 'correct' reassembly is the
        # one that maximizes the cumulative dot product of the quaternions.

        current_resonance = Quaternion(1, 0, 0, 0)
        ordered_parts = []

        for p in primitives:
            q_elements = p["data"]["quaternion"]
            q_part = Quaternion(*q_elements)
            modality = p["data"].get("modality", "unknown")

            # Measure 'Friction' against the current state
            dot_prod = abs(current_resonance.dot(q_part))
            friction = 1.0 - dot_prod
            total_friction += friction

            # Record per-modality resonance
            if modality not in modality_resonance:
                modality_resonance[modality] = []
            modality_resonance[modality].append(dot_prod)

            # Re-orient current resonance (Simulating the 'learning' or 'fitting' process)
            current_resonance = (current_resonance * q_part).normalize()

            ordered_parts.append({
                "part_name": p["data"]["part_name"],
                "modality": modality,
                "friction": friction,
                "resonance_delta": dot_prod
            })

            reassembly_log.append({
                "step": f"Aligning {p['data']['part_name']}",
                "friction": friction,
                "resonance_delta": dot_prod,
                "status": "Joy" if friction < 0.2 else "Pain"
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
        [Phase: Meta-Stable Rotors] 정적 로터(상수)를 회전시켜 새로운 평형을 찾습니다.
        거대한 불일치가 발생했을 때, 기존의 '기준(Lens)' 자체를 가변화하여 진화합니다.
        """
        trace = self.memory.read_engram_trace(anchor_constant_id)
        if not trace: return False

        # 기존 상수의 위상
        current_q_elements = trace["data"].get("quaternion", [1,0,0,0])
        current_q = Quaternion(*current_q_elements)

        # 새로운 정보(conflicting_trajectory)와의 SLERP (진화적 회전)
        # 안정성(Stability)이 낮을수록 더 크게 회전합니다.
        stability = self.memory.index[anchor_constant_id].get("stability", 1.0)
        shift_amount = 0.5 * (1.0 - stability + 0.1)

        evolved_q = Quaternion.slerp(current_q, conflicting_trajectory, amount=shift_amount)

        # 상수 업데이트 (정적 로터의 회전)
        self.memory.update_engram_data(
            anchor_constant_id,
            {"quaternion": evolved_q.elements, "evolution_event": "Structural_Shift"},
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
