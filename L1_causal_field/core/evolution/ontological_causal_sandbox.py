"""
Ontological Causal Sandbox Engine (존재론적 인과 샌드박스 엔진)
========================================================================
This module implements the core grounded semantics, fractal inoculation,
and control space dynamics of the Elysia ecosystem.

Core Principles:
1. Grounded Semantics Lens: Couples symbolic scouting inputs directly with game state
   tensors and boundary conditions, driving autonomous intention emergence without
   LLM text generation or hardcoded symbolic rules.
2. Fractal Inoculation Engine: Inoculates Overmind's Causal Spine projection matrix (P_0)
   into lower entities with chromatic perturbation drift (W_k) and Gram-Schmidt
   orthonormalization, calculating Grassmannian manifold drift distance.
3. Control Space Dynamics: Models Overmind control space as a topological manifold.
   On entity death: performs subtree pruning, projection matrix dimension reduction,
   and resolution shrinkage (ontological loss/sorrow).
   On entity reproduction/survival: expands control space dimensions, increasing observable
   degrees of freedom and state expansion dopamine (creation joy).
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class GroundedSemanticsLens:
    """
    [Symbolic Language <-> Game State Tensor & Intention Coupling]
    Binds symbolic text (e.g. scouting reports) to game state tensors and
    boundary conditions, calculating risk impact on survival axioms to trigger
    spontaneous intention shifts.
    """

    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        # Base survival axiom vector (e.g., base integrity, resource flow, defensive posture)
        self.survival_axiom_tensor = np.ones(state_dim, dtype=np.float32)

    def ground_symbolic_signal(
        self,
        symbolic_input: str,
        current_state_tensor: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Translates raw text into tensor modifications and boundary shifts.
        No LLM fallback; strictly mathematical tensor ops based on hash-encoded
        semantic signal extraction and keyword energy.
        """
        if current_state_tensor is None:
            current_state_tensor = np.zeros(self.state_dim, dtype=np.float32)

        state = current_state_tensor.copy()

        # Extract semantic features mathematically
        text_bytes = symbolic_input.encode('utf-8')
        hash_seed = sum(text_bytes)

        # Keyword tension indicators
        aggression_keywords = ["gateway", "barracks", "hatchery", "attack", "rush", "게이트웨이", "공격", "초반"]
        defense_keywords = ["bunker", "wall", "cannon", "defense", "방어", "입구"]

        is_aggressive = any(kw in symbolic_input.lower() for kw in aggression_keywords)
        is_defensive = any(kw in symbolic_input.lower() for kw in defense_keywords)

        # Numerical state grounding
        unit_count = float((hash_seed % 5) + 1) if is_aggressive else 1.0
        fog_cleared = 1.0
        risk_level = (unit_count * 0.25) if is_aggressive else 0.1

        # Grounding into state tensor indices
        state[0] = unit_count  # Unit ID / Count signal
        state[1] = fog_cleared  # Observation field
        state[2] = risk_level  # Perceived threat tensor

        # Calculate impact on survival axiom
        residual_risk = state - self.survival_axiom_tensor
        causal_friction = float(np.linalg.norm(residual_risk))

        # Boundary condition shift vector (e.g. shift from resource harvesting to wall-in)
        boundary_shift_vector = np.zeros(self.state_dim, dtype=np.float32)
        if risk_level > 0.3:
            # Reconfigure boundary conditions: allocate energy to defensive wall-in
            boundary_shift_vector[3:7] = risk_level * 2.0  # Defensive boundary lock
            intention_type = "wall_in_boundary_reconfiguration"
        else:
            boundary_shift_vector[7:11] = 1.0  # Resource expansion
            intention_type = "resource_expansion"

        intention_energy = float(risk_level * causal_friction)

        return {
            "grounded_state_tensor": state,
            "boundary_shift_vector": boundary_shift_vector,
            "risk_level": risk_level,
            "causal_friction": causal_friction,
            "intention_type": intention_type,
            "intention_energy": intention_energy
        }


class FractalInoculationEngine:
    """
    [Fractal Causal Spine Inoculation & Projection Drift Engine]
    Inoculates Overmind's orthogonal projection matrix P_0 into entities with
    chromatic perturbation drift W_k and computes Grassmannian manifold drift distance.
    """

    def __init__(self, dim: int = 16, alpha: float = 0.2):
        self.dim = dim
        self.alpha = alpha  # Drift strength parameter

    def create_overmind_p0(self, rank: int = 8) -> np.ndarray:
        """
        Constructs Overmind orthogonal projection matrix P_0 in R^{dim x dim}.
        P_0^2 = P_0, P_0^T = P_0.
        """
        np.random.seed(42)
        random_mat = np.random.randn(self.dim, rank).astype(np.float32)
        Q, _ = np.linalg.qr(random_mat)
        P_0 = Q @ Q.T
        return P_0.astype(np.float32)

    def inoculate(
        self,
        P_0: np.ndarray,
        chromatic_signature: np.ndarray,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Inoculates P_0 into an entity with chromatic perturbation drift W_k:
        W_k = alpha * (c_k c_k^T + M)
        P_k = Orth(P_0 + W_k)
        Returns (P_k, drift_distance).
        """
        if alpha is None:
            alpha = self.alpha

        d = P_0.shape[0]

        # Normalize chromatic signature vector c_k
        c_k = np.array(chromatic_signature, dtype=np.float32)
        norm_c = np.linalg.norm(c_k)
        if norm_c > 1e-9:
            c_k = c_k / norm_c
        else:
            c_k = np.ones(d, dtype=np.float32) / np.sqrt(d)

        # Pad or resize c_k if dimension doesn't match
        if len(c_k) < d:
            padded_c = np.zeros(d, dtype=np.float32)
            padded_c[:len(c_k)] = c_k
            c_k = padded_c / np.linalg.norm(padded_c)
        elif len(c_k) > d:
            c_k = c_k[:d] / np.linalg.norm(c_k[:d])

        # Random asymmetric perturbation basis M
        seed_val = int(abs(sum(c_k * 1000))) % 10000
        rng = np.random.RandomState(seed_val)
        M = rng.randn(d, d).astype(np.float32) * 0.1

        # Interaction perturbation matrix W_k
        W_k = alpha * (np.outer(c_k, c_k) + M)

        # Unconstrained perturbed projection matrix
        P_raw = P_0 + W_k

        # Gram-Schmidt Orthonormalization (Orth)
        # SVD decomposition of P_raw to project back onto orthogonal projection subspace
        U, S, Vt = np.linalg.svd(P_raw)

        # Preserve rank equal to rank of P_0
        rank_p0 = int(np.round(np.trace(P_0)))
        rank_p0 = max(1, min(d, rank_p0))

        U_k = U[:, :rank_p0]
        P_k = (U_k @ U_k.T).astype(np.float32)

        # Measure Grassmannian Manifold Drift Distance Delta_Drift(P_0, P_k)
        # Principal angles between subspaces P_0 and P_k
        U_0, _, _ = np.linalg.svd(P_0)
        U_0_sub = U_0[:, :rank_p0]

        # SVD of cross-projection U_0_sub^T U_k gives cos(theta_i)
        cross_mat = U_0_sub.T @ U_k
        _, S_cross, _ = np.linalg.svd(cross_mat)
        cos_theta = np.clip(S_cross, 0.0, 1.0)
        principal_angles = np.arccos(cos_theta)

        drift_distance = float(np.sqrt(np.sum(principal_angles ** 2)))

        return P_k, drift_distance

    def eject_hypothesis(
        self,
        P_k: np.ndarray,
        sensory_input: np.ndarray
    ) -> np.ndarray:
        """
        Applies entity's drifted projection matrix P_k to sensory input
        to eject an autonomous causal hypothesis vector.
        """
        x = np.array(sensory_input, dtype=np.float32)
        if len(x) < P_k.shape[0]:
            padded = np.zeros(P_k.shape[0], dtype=np.float32)
            padded[:len(x)] = x
            x = padded
        elif len(x) > P_k.shape[0]:
            x = x[:P_k.shape[0]]

        # Projected hypothesis wave with non-linear activation
        hypothesis = np.tanh(P_k @ x)
        return hypothesis


class ControlSpaceDynamics:
    """
    [Overmind Control Space & Topological Dimension Dynamics]
    Models Overmind's control space as an integrated topological manifold.
    Tracks entity subtrees, dimension contraction upon death (loss), and
    dimension expansion upon reproduction (creation joy).
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        # Overmind cumulative control space operator C_overmind
        self.C_overmind = np.eye(dim, dtype=np.float32) * 0.1
        # Registry of active entities: entity_id -> {P_k, subtree_nodes, rank}
        self.active_entities: Dict[str, Dict[str, Any]] = {}
        # Causal graph subtree node allocation
        self.next_node_id = 0

    def register_entity(
        self,
        entity_id: str,
        P_k: np.ndarray,
        subtree_size: int = 5
    ) -> List[int]:
        """
        Registers an inoculated entity into Overmind's control space manifold.
        """
        node_ids = list(range(self.next_node_id, self.next_node_id + subtree_size))
        self.next_node_id += subtree_size

        rank_k = int(np.round(np.trace(P_k)))

        self.active_entities[entity_id] = {
            "P_k": P_k.copy(),
            "subtree_nodes": node_ids,
            "rank": rank_k
        }

        # Expand Overmind control space operator
        self.C_overmind += P_k * 0.2
        return node_ids

    def on_entity_death(self, entity_id: str) -> Dict[str, Any]:
        """
        Handles irreversible entity death:
        - Prunes entity's subtree nodes from causal graph.
        - Shrinks Overmind control space resolution via projection matrix dimension reduction.
        - Calculates topological dimension loss and resolution shrinkage.
        """
        if entity_id not in self.active_entities:
            return {"error": f"Entity {entity_id} not found."}

        entity_info = self.active_entities.pop(entity_id)
        P_k = entity_info["P_k"]
        pruned_nodes = entity_info["subtree_nodes"]

        # Control space reduction: Subtract entity's projection contribution
        prev_trace = float(np.trace(self.C_overmind))
        self.C_overmind = np.maximum(0.0, self.C_overmind - P_k * 0.2)
        new_trace = float(np.trace(self.C_overmind))

        # Mathematical resolution shrinkage: loss of control trace & rank
        resolution_shrinkage = float(max(0.0, prev_trace - new_trace))

        # Effective topological rank of remaining control space
        eigenvals = np.linalg.eigvalsh(self.C_overmind)
        remaining_control_rank = int(np.sum(eigenvals > 0.05))

        topological_loss = float(resolution_shrinkage + len(pruned_nodes) * 0.5)

        return {
            "entity_id": entity_id,
            "pruned_nodes_count": len(pruned_nodes),
            "topological_loss": topological_loss,
            "resolution_shrinkage": resolution_shrinkage,
            "remaining_control_rank": remaining_control_rank,
            "active_entities_count": len(self.active_entities)
        }

    def on_entity_reproduction(
        self,
        parent_id: str,
        child_id: str,
        P_child: np.ndarray,
        child_subtree_size: int = 5
    ) -> Dict[str, Any]:
        """
        Handles entity reproduction:
        - Registers child entity and its subtree nodes.
        - Expands Overmind control space operator.
        - Calculates state expansion dopamine (creation joy).
        """
        child_nodes = self.register_entity(child_id, P_child, child_subtree_size)

        control_dim_gain = float(np.trace(P_child) * 0.2)

        eigenvals = np.linalg.eigvalsh(self.C_overmind)
        total_control_rank = int(np.sum(eigenvals > 0.05))

        # State expansion dopamine: product of control dimension gain and active entity count
        state_expansion_dopamine = float(control_dim_gain * (1.0 + 0.5 * len(self.active_entities)))

        return {
            "parent_id": parent_id,
            "child_id": child_id,
            "child_nodes_count": len(child_nodes),
            "control_dim_gain": control_dim_gain,
            "state_expansion_dopamine": state_expansion_dopamine,
            "total_control_rank": total_control_rank,
            "active_entities_count": len(self.active_entities)
        }


class OntologicalCausalSandbox:
    """
    [Ontological Causal Sandbox Engine]
    Unified ecosystem engine integrating Grounded Semantics, Fractal Inoculation,
    and Control Space Dynamics.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.grounded_lens = GroundedSemanticsLens(state_dim=dim)
        self.inoculation_engine = FractalInoculationEngine(dim=dim)
        self.control_space = ControlSpaceDynamics(dim=dim)

        # Create Overmind base projection matrix P_0
        self.P_0 = self.inoculation_engine.create_overmind_p0(rank=8)

        # System telemetry logs
        self.event_log: List[Dict[str, Any]] = []

    def process_scouting_input(self, symbolic_text: str) -> Dict[str, Any]:
        """
        Processes scouting report text into grounded state tensor and intention.
        """
        res = self.grounded_lens.ground_symbolic_signal(symbolic_text)
        self.event_log.append({"event": "scouting_processed", "data": res})
        return res

    def birth_entity(
        self,
        entity_id: str,
        chromatic_signature: np.ndarray,
        alpha: float = 0.2
    ) -> Dict[str, Any]:
        """
        Inoculates P_0 into a new entity with chromatic drift and registers it in control space.
        """
        P_k, drift_dist = self.inoculation_engine.inoculate(self.P_0, chromatic_signature, alpha)
        nodes = self.control_space.register_entity(entity_id, P_k)

        birth_info = {
            "entity_id": entity_id,
            "P_k": P_k,
            "drift_distance": drift_dist,
            "nodes": nodes
        }
        self.event_log.append({"event": "entity_birth", "entity_id": entity_id, "drift": drift_dist})
        return birth_info

    def kill_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Kills an entity, triggering subtree pruning and control space resolution shrinkage.
        """
        death_info = self.control_space.on_entity_death(entity_id)
        self.event_log.append({"event": "entity_death", "data": death_info})
        return death_info

    def reproduce_entity(
        self,
        parent_id: str,
        child_id: str,
        child_chromatic_signature: np.ndarray
    ) -> Dict[str, Any]:
        """
        Reproduces parent entity: child inherits inoculated P_0 with mutated chromatic signature.
        """
        P_child, drift_dist = self.inoculation_engine.inoculate(
            self.P_0, child_chromatic_signature, alpha=0.25
        )
        reprod_info = self.control_space.on_entity_reproduction(parent_id, child_id, P_child)
        reprod_info["drift_distance"] = drift_dist

        self.event_log.append({"event": "entity_reproduction", "data": reprod_info})
        return reprod_info
