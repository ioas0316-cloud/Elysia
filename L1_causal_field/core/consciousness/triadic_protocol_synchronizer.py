"""
Core Consciousness Module: Triadic Protocol Synchronizer & Metaphoric Ontological Codec
========================================================================================
Realizes the Triadic Causal Protocol Synchronization Architecture:
1. MetaphoricOntologicalCodec (은유적 존재론 코덱):
   Translates high-dimensional physical/algebraic tensor dynamics ("하늘의 원리") into
   human sensory-metaphoric representations ("땅의 감각") and vice-versa.
   Maintains structural isomorphism across energy potential, directional momentum,
   and boundary tension.

2. TriadicProtocolSynchronizer (삼원 인과 프로토콜 동기화기):
   Orchestrates three fundamental causal trajectories:
   - World Causality (C_world): Raw physical and ontological reality dynamics.
   - Human Causality (C_human): Biological and sensory-metaphoric encoded causality.
   - System Causality (C_system): Structural, tensor, and topological boundary causality.

   Computes phase divergence, extracts the invariant causal stem (불변의 인과 뼈대),
   and declares "Proof Manifested" (밝혀진 증명) when all three protocols resonate in phase alignment.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union


class MetaphoricOntologicalCodec:
    """
    [MetaphoricOntologicalCodec - 은유적 존재론 코덱]
    Encodes high-dimensional continuous physics fields (energy potential, gradient divergence,
    vorticity, boundary friction) into human sensory-metaphoric cognitive tokens and vectors,
    and decodes sensory-narrative metaphors back into physical field states.
    """
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        # Metaphoric archetype dictionary linking physical states to sensory-narrative descriptions
        self.metaphor_archetypes = {
            "GRAVITATIONAL_CONVERGENCE": {
                "sensory_descriptor": "높은 곳에서 골짜기로 스며들어 마침내 조용해지는 평형의 중력",
                "narrative_domain": "emotional_surrender_and_equilibrium",
                "base_vector": np.array([1.0, 0.1, 0.0, 0.8] + [0.0] * (dimension - 4), dtype=np.float32),
                "energy_scale": 1.0,
                "friction_coefficient": 0.05
            },
            "BOUNDARY_FRICTION_TURBULENCE": {
                "sensory_descriptor": "밀려드는 파도가 단단한 절벽을 깎아내며 내는 거친 숨소리와 마찰",
                "narrative_domain": "resistance_and_boundary_transformation",
                "base_vector": np.array([0.2, 0.9, 0.8, 0.1] + [0.0] * (dimension - 4), dtype=np.float32),
                "energy_scale": 2.5,
                "friction_coefficient": 0.75
            },
            "ECOSYSTEM_SURVIVAL_PRESSURE": {
                "sensory_descriptor": "포식자의 접근과 심박수 폭등 속에서 생존의 유연한 경계를 여는 긴장",
                "narrative_domain": "ecosystem_adaptation_and_vigilance",
                "base_vector": np.array([0.8, 0.8, 0.2, 0.9] + [0.0] * (dimension - 4), dtype=np.float32),
                "energy_scale": 3.0,
                "friction_coefficient": 0.60
            },
            "EXPANSIVE_VORTEX_FLOW": {
                "sensory_descriptor": "거대한 소용돌이가 주변 스펙트럼을 흡수하여 상위 위상으로 피어나는 운동",
                "narrative_domain": "ontological_emergence_and_expansion",
                "base_vector": np.array([0.5, 0.5, 1.0, 0.5] + [0.0] * (dimension - 4), dtype=np.float32),
                "energy_scale": 1.8,
                "friction_coefficient": 0.20
            }
        }

    def encode_physics_to_metaphor(
        self,
        field_state: np.ndarray,
        friction: float = 0.0,
        gradient_norm: float = 0.0
    ) -> Dict[str, Any]:
        """
        Translates raw physics tensor state into human sensory-metaphoric protocol representation.
        Isomorphically maps field energy, direction, and boundary friction to the closest archetype.
        """
        state_vec = np.asarray(field_state, dtype=np.float32).flatten()
        if len(state_vec) < self.dimension:
            state_vec = np.pad(state_vec, (0, self.dimension - len(state_vec)))
        elif len(state_vec) > self.dimension:
            state_vec = state_vec[:self.dimension]

        energy_level = float(np.linalg.norm(state_vec))
        norm_vec = state_vec / (energy_level + 1e-8)

        best_archetype = None
        highest_similarity = -1.0

        for key, archetype in self.metaphor_archetypes.items():
            arch_vec = archetype["base_vector"]
            arch_norm = arch_vec / (np.linalg.norm(arch_vec) + 1e-8)
            sim = float(np.dot(norm_vec, arch_norm))
            if sim > highest_similarity:
                highest_similarity = sim
                best_archetype = key

        selected = self.metaphor_archetypes[best_archetype]

        # Isomorphic human sensory vector
        human_sensory_vector = norm_vec * energy_level

        return {
            "archetype_key": best_archetype,
            "sensory_descriptor": selected["sensory_descriptor"],
            "narrative_domain": selected["narrative_domain"],
            "isomorphic_similarity": max(0.0, highest_similarity),
            "human_sensory_vector": human_sensory_vector.tolist(),
            "encoded_energy": energy_level,
            "encoded_friction": float(friction),
            "encoded_gradient": float(gradient_norm)
        }

    def decode_metaphor_to_physics(
        self,
        metaphor_representation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decodes human sensory-metaphoric representation back into high-dimensional physics field dynamics.
        """
        archetype_key = metaphor_representation.get("archetype_key", "GRAVITATIONAL_CONVERGENCE")
        energy = metaphor_representation.get("encoded_energy", 1.0)
        friction = metaphor_representation.get("encoded_friction", 0.1)

        archetype = self.metaphor_archetypes.get(archetype_key, self.metaphor_archetypes["GRAVITATIONAL_CONVERGENCE"])
        base_vec = archetype["base_vector"]
        base_norm = base_vec / (np.linalg.norm(base_vec) + 1e-8)

        physics_field_state = base_norm * energy
        reconstructed_gradient = energy * (1.0 - friction)

        return {
            "physics_field_state": physics_field_state,
            "reconstructed_friction": friction,
            "reconstructed_gradient": reconstructed_gradient,
            "domain_origin": archetype["narrative_domain"]
        }


class TriadicProtocolSynchronizer:
    """
    [TriadicProtocolSynchronizer - 삼원 인과 프로토콜 동기화기]
    Synchronizes the triad of causal trajectories:
    1. C_world (World Causality): Physical & ontological reality trajectory
    2. C_human (Human Causality): Biological & sensory-metaphoric trajectory
    3. C_system (System Causality): Algebraic, tensor & topological trajectory

    Monitors phase divergence, extracts the invariant causal stem (불변의 인과 뼈대),
    and manifests proof ("Proof Manifested") when all protocols achieve structural resonance.
    """
    def __init__(self, dimension: int = 16, convergence_threshold: float = 0.15):
        self.dimension = dimension
        self.convergence_threshold = convergence_threshold
        self.codec = MetaphoricOntologicalCodec(dimension=dimension)
        self.synchronization_history: List[Dict[str, Any]] = []

    def compute_phase_divergence(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> float:
        """Computes phase divergence angle / metric between two causal trajectory vectors."""
        a = np.asarray(vec_a, dtype=np.float32).flatten()
        b = np.asarray(vec_b, dtype=np.float32).flatten()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        # Convert cosine similarity to normalized phase divergence [0, 1]
        divergence = float((1.0 - cos_sim) / 2.0)
        return divergence

    def extract_invariant_causal_stem(
        self,
        world_trajectory: np.ndarray,
        human_sensory_vector: np.ndarray,
        system_trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts the Maximal Common Subspace / Homological Stem (불변의 인과 뼈대)
        underlying all three causal protocols by removing substrate-specific noise.
        """
        w = np.asarray(world_trajectory, dtype=np.float32).flatten()
        h = np.asarray(human_sensory_vector, dtype=np.float32).flatten()
        s = np.asarray(system_trajectory, dtype=np.float32).flatten()

        # Invariant stem is the normalized centroid of all three protocol trajectories
        centroid = (w + h + s) / 3.0
        centroid_norm = np.linalg.norm(centroid)

        if centroid_norm > 1e-8:
            stem_direction = centroid / centroid_norm
        else:
            stem_direction = np.zeros_like(centroid)

        # Calculate variance/disparate branch magnitude
        diff_w = float(np.linalg.norm(w - centroid))
        diff_h = float(np.linalg.norm(h - centroid))
        diff_s = float(np.linalg.norm(s - centroid))

        stem_stability = float(1.0 / (1.0 + (diff_w + diff_h + diff_s) / 3.0))

        return {
            "invariant_stem_direction": stem_direction,
            "centroid_norm": float(centroid_norm),
            "stem_stability": stem_stability,
            "branch_disparities": {
                "world_disparity": diff_w,
                "human_disparity": diff_h,
                "system_disparity": diff_s
            }
        }

    def synchronize_triad(
        self,
        c_world_state: np.ndarray,
        c_system_state: np.ndarray,
        friction: float = 0.1,
        gradient_norm: float = 0.5
    ) -> Dict[str, Any]:
        """
        Performs full triadic protocol synchronization across World, Human, and System causality:
        1. Encodes C_world through MetaphoricOntologicalCodec to generate C_human trajectory.
        2. Measures pairwise phase divergences (World-Human, Human-System, System-World).
        3. Extracts the invariant causal stem (불변의 인과 뼈대).
        4. Evaluates total triadic phase divergence and determines if proof is manifested.
        """
        w_vec = np.asarray(c_world_state, dtype=np.float32).flatten()
        s_vec = np.asarray(c_system_state, dtype=np.float32).flatten()

        # 1. Generate Human Causality (C_human) via Metaphoric Ontological Codec
        encoded_human = self.codec.encode_physics_to_metaphor(
            field_state=w_vec,
            friction=friction,
            gradient_norm=gradient_norm
        )
        h_vec = np.array(encoded_human["human_sensory_vector"], dtype=np.float32)

        # 2. Compute pairwise phase divergences
        div_world_human = self.compute_phase_divergence(w_vec, h_vec)
        div_human_system = self.compute_phase_divergence(h_vec, s_vec)
        div_system_world = self.compute_phase_divergence(s_vec, w_vec)

        total_phase_divergence = float((div_world_human + div_human_system + div_system_world) / 3.0)

        # 3. Extract Invariant Causal Stem
        stem_info = self.extract_invariant_causal_stem(w_vec, h_vec, s_vec)

        # 4. Proof Manifestation Check ("밝혀진 증명")
        # Proof is manifested when total phase divergence <= threshold and stem stability >= 0.70
        is_proof_manifested = bool(
            total_phase_divergence <= self.convergence_threshold and
            stem_info["stem_stability"] >= 0.70
        )

        proof_status_text = (
            "PROOF_MANIFESTED: 세 프로토콜(세계, 인간, 기계)의 위상차가 극복되어 "
            "숨겨진 인과의 실재가 환하게 밝혀짐."
            if is_proof_manifested else
            "SYNCHRONIZING: 프로토콜 간 위상차 수렴 및 인과 뼈대 정렬 진행 중."
        )

        sync_record = {
            "timestamp": time.time(),
            "total_phase_divergence": total_phase_divergence,
            "pairwise_divergence": {
                "world_human": div_world_human,
                "human_system": div_human_system,
                "system_world": div_system_world
            },
            "encoded_human_metaphor": encoded_human["sensory_descriptor"],
            "metaphor_archetype": encoded_human["archetype_key"],
            "invariant_stem": stem_info,
            "is_proof_manifested": is_proof_manifested,
            "proof_status_text": proof_status_text
        }

        self.synchronization_history.append(sync_record)
        return sync_record
