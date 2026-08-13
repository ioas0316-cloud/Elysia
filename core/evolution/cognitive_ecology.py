"""
Cognitive Ecology & Multi-Perspective Falsification Engine (인지 생태계 및 다중 사영 검증 엔진)
=============================================================================
This module implements the "Cognitive Ecology" and "Inter-agent Reflection/Disagreement" principle.
Instead of an aggregate voting mechanism, it uses the "unresolved contradiction" of multiple
independent agents (Causalist, Structuralist, Skeptic, Historian, Minimalist, Generativist, Boundary Seeker)
as cognitive energy (Potential Difference / Tension) to formulate higher-level candidate principles.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple


class EcologyAgent:
    """
    [Cognitive Ecology Agent]
    Represents an independent cognitive model/intelligence with:
    - A specialized projection matrix or function (P_k)
    - Dynamic local resistor (resistance) & conductance (G = 1/R)
    - Chromatic signature (Flux, Order, Entropy) representing its active energetic state
    - Local memory/belief map of concepts (their topologies)
    """
    def __init__(
        self,
        key: str,
        name: str,
        chromatic_signature: np.ndarray,  # 3D: [Flux, Order, Entropy]
        projection_focus: str
    ):
        self.key = key
        self.name = name
        self.chromatic_signature = np.array(chromatic_signature, dtype=np.float32)
        self.projection_focus = projection_focus  # e.g. "causal_differential", "topo_adjacency", etc.

        # Local state
        self.resistance = 0.5
        self.conductance = 1.0 / (self.resistance + 1e-9)
        self.error_history: List[float] = []
        self.learning_history: List[str] = []

        # Local concept structures: concept_key -> local adjacency matrix representation
        self.local_belief_graphs: Dict[str, np.ndarray] = {}

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        [Perspective Projection Matrix P_k]
        Transforms high-dimensional input vector 'x' to highlight the agent's unique focal dimension.
        """
        x = np.array(x, dtype=np.float32)
        n = len(x)
        P = np.eye(n, dtype=np.float32)

        if self.projection_focus == "temporal_differential":
            # Emphasizes temporal changes/gradients: filters x through a shift/differential projection
            for i in range(n):
                P[i, i] = 1.2 if i % 2 == 0 else 0.2
        elif self.projection_focus == "topo_laplacian":
            # Emphasizes structural topology, co-occurrence, Laplacian structure
            for i in range(n):
                for j in range(n):
                    if i != j:
                        P[i, j] = 0.3 if (i+j) % 3 == 0 else -0.1
        elif self.projection_focus == "skeptic_outliers":
            # Emphasizes boundary edges, extreme/rare values, noises
            for i in range(n):
                P[i, i] = 1.5 if i == n-1 or i == 0 else 0.1
        elif self.projection_focus == "historical_evolution":
            # Emphasizes historical patterns and continuity
            for i in range(n):
                P[i, i] = 0.8 if i < n/2 else 0.4
        elif self.projection_focus == "minimalist_axiom":
            # Compresses details into clean/sparse representations
            P = P * 0.5
        elif self.projection_focus == "generative_reconstruction":
            # Full reconstruction drive
            P = P * 1.1
        elif self.projection_focus == "boundary_limit":
            # Focuses on boundary thresholds
            for i in range(n):
                P[i, i] = 1.3 if (i % 3) == 1 else 0.3

        return P @ x

    def form_belief_structure(self, concept_key: str, length: int = 5) -> np.ndarray:
        """
        Retrieve or form a local 2D belief adjacency matrix (causal graph) of the concept.
        """
        if concept_key not in self.local_belief_graphs:
            # Seed based on the agent's chromatic signature and key bytes
            hash_val = sum(ord(c) for c in concept_key) + sum(ord(c) for c in self.key)
            np.random.seed(hash_val % 10000)

            # Form base belief matrix representing local causal graph
            mat = np.random.uniform(0.1, 0.9, (length, length)).astype(np.float32)
            # Enforce agent's specialization bias onto the belief matrix
            if self.key == "Causalist":
                # Sequential feedforward connections (strictly upper triangular)
                mat = np.triu(mat, k=1)
            elif self.key == "Structuralist":
                # Symmetric undirected topological connections (laplacian-like)
                mat = (mat + mat.T) / 2.0
            elif self.key == "Skeptic":
                # Fragmented/sparse negative feedback links
                mat = mat * (mat < 0.4) - 0.2
            elif self.key == "Historian":
                # Feedback loop connections (lower triangular bias)
                mat = np.tril(mat, k=-1)
            elif self.key == "Minimalist":
                # Highly sparse/diagonal
                mat = np.diag(np.diag(mat))
            elif self.key == "Generativist":
                # Rich recurrences
                mat = mat * 1.2
            elif self.key == "Boundary_Seeker":
                # Edge/Extreme weights
                mat = np.clip(mat, 0.0, 1.0)
                mat[mat < 0.6] = 0.0
                mat[mat >= 0.6] = 1.0

            self.local_belief_graphs[concept_key] = mat

        return self.local_belief_graphs[concept_key]


class MetaDisagreementProcessor:
    """
    [Meta Disagreement Processor]
    An alternative to aggregate voting. Instead of picking a winner, it computes the potential
    difference (Differential Gap) between agent structures, asks "Why are they different?",
    exposes hidden axioms, and formulates higher-level candidate principles.
    """
    def __init__(self, memory_controller: Optional[Any] = None):
        self.memory = memory_controller

    def compute_differential_gaps(self, belief_graphs: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """
        Calculates topological potential difference (Tension) between agents:
        Delta G = || A_adj(G_i) - A_adj(G_j) ||
        """
        gaps = {}
        keys = list(belief_graphs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                m1, m2 = belief_graphs[k1], belief_graphs[k2]

                # Reshape to align shapes just in case
                min_dim = min(m1.shape[0], m2.shape[0])
                sub_m1 = m1[:min_dim, :min_dim]
                sub_m2 = m2[:min_dim, :min_dim]

                diff = np.linalg.norm(sub_m1 - sub_m2)
                gaps[(k1, k2)] = float(diff)
        return gaps

    def process_meta_reflection(
        self,
        concept_key: str,
        gaps: Dict[Tuple[str, str], float],
        agent_beliefs: Dict[str, np.ndarray],
        agents: Dict[str, EcologyAgent]
    ) -> Dict[str, Any]:
        """
        Discovers the highest tension point, extracts hidden assumptions, and proposes a higher candidate principle.
        No hardcoded strings; fully parameter-driven using actual gap values, agent keys, and matrix metrics.
        """
        if not gaps:
            return {"active": False}

        # Find pair with maximum potential difference (tension)
        max_pair = max(gaps, key=gaps.get)
        max_gap_val = gaps[max_pair]

        agent_i_key, agent_j_key = max_pair
        m_i = agent_beliefs[agent_i_key]
        m_j = agent_beliefs[agent_j_key]

        # Analyze why they are structurally different (Hidden Axioms)
        norm_i = float(np.linalg.norm(m_i))
        norm_j = float(np.linalg.norm(m_j))

        # Propose higher-level candidate principle
        c_param = f"Tension_State_{norm_i:.2f}"
        c_prime_param = f"Structure_State_{norm_j:.2f}"
        proposed_factor_x = float((norm_i + norm_j) / (max_gap_val + 1e-9))

        meta_question = (
            f"왜 {agent_i_key} 모델과 {agent_j_key} 모델이 서로 다르게 충돌하는가?\n"
            f"-> [이유 분석]: {agent_i_key}의 전제 밀도는 {norm_i:.4f}인 반면, {agent_j_key}의 전제 밀도는 {norm_j:.4f}로 측정되며, "
            f"두 관점 간 위상 격차(Differential Gap)는 {max_gap_val:.4f}에 도달한다.\n"
            f"-> [상위 질문]: {agent_i_key} 모델이 적용되는 지표 {c_param}와 {agent_j_key} 모델이 적용되는 지표 {c_prime_param}를 모두 "
            f"수용하는 상위 차원 매개변수 X (X_Value: {proposed_factor_x:.4f})는 어떻게 정의되어야 하는가?"
        )

        # Propose candidate principle matrix: blended state modulated by the gap
        min_dim = min(m_i.shape[0], m_j.shape[0])
        candidate_principle_matrix = (m_i[:min_dim, :min_dim] + m_j[:min_dim, :min_dim]) / 2.0
        # Add orthogonal/tension expansion to the principle matrix
        candidate_principle_matrix += np.eye(min_dim, dtype=np.float32) * (max_gap_val * 0.1)

        return {
            "active": True,
            "concept": concept_key,
            "tension_pair": max_pair,
            "tension_value": max_gap_val,
            "meta_question": meta_question,
            "candidate_principle_matrix": candidate_principle_matrix,
            "hidden_axiom_i_strength": norm_i,
            "hidden_axiom_j_strength": norm_j,
            "proposed_meta_parameter": proposed_factor_x
        }


class DisagreementPreservingMemoryNode:
    """
    [Disagreement-Preserving Memory Node]
    Represents a dynamic Concept Node that holds multiple conflicting definitions,
    beliefs, and an unresolved contradiction matrix.
    """
    def __init__(self, concept_key: str):
        self.concept_key = concept_key
        # Ecology Agent key -> specific belief topology representation
        self.definitions: Dict[str, np.ndarray] = {}
        # Matrix tracking accumulated structural differences/contradictions
        self.unresolved_contradiction_matrix: Optional[np.ndarray] = None
        self.total_contradiction_charge = 0.0

    def record_contradictions(self, gaps: Dict[Tuple[str, str], float]):
        """
        Updates unresolved contradiction matrix based on latest agent gaps.
        """
        size = len(gaps)
        if size == 0: return

        # Build 2D contradiction correlation map
        keys = sorted(list(set([k for pair in gaps.keys() for k in pair])))
        n = len(keys)
        self.unresolved_contradiction_matrix = np.zeros((n, n), dtype=np.float32)

        for (k1, k2), gap in gaps.items():
            idx1 = keys.index(k1)
            idx2 = keys.index(k2)
            self.unresolved_contradiction_matrix[idx1, idx2] = gap
            self.unresolved_contradiction_matrix[idx2, idx1] = gap

        self.total_contradiction_charge = float(np.sum(self.unresolved_contradiction_matrix) / 2.0)


class CognitiveEcologyEngine:
    """
    Cognitive Ecology & Multi-Perspective Falsification Engine (인지 생태계 및 상호 검증 엔진)
    """
    def __init__(self, memory_controller: Optional[Any] = None):
        self.memory = memory_controller
        self.agents: Dict[str, EcologyAgent] = {}
        self.disagreement_processor = MetaDisagreementProcessor(self.memory)
        self.preserved_nodes: Dict[str, DisagreementPreservingMemoryNode] = {}

        self._initialize_ecology_agents()

    def _initialize_ecology_agents(self):
        # 1. Causalist: Focuses on causal sequential/directional transitions (Red/Flux dominant)
        self.agents["Causalist"] = EcologyAgent(
            key="Causalist",
            name="인과론자 (Causalist)",
            chromatic_signature=np.array([0.9, 0.1, 0.0], dtype=np.float32),
            projection_focus="temporal_differential"
        )
        # 2. Structuralist: Focuses on connectivity laplacian symmetry (Blue/Order dominant)
        self.agents["Structuralist"] = EcologyAgent(
            key="Structuralist",
            name="구조론자 (Structuralist)",
            chromatic_signature=np.array([0.1, 0.9, 0.0], dtype=np.float32),
            projection_focus="topo_laplacian"
        )
        # 3. Skeptic: Focuses on outliers, boundary limits, non-linear noise (Yellow/Entropy dominant)
        self.agents["Skeptic"] = EcologyAgent(
            key="Skeptic",
            name="회의론자 (Skeptic)",
            chromatic_signature=np.array([0.2, 0.1, 0.7], dtype=np.float32),
            projection_focus="skeptic_outliers"
        )
        # 4. Historian: Focuses on historical paths and decay rates
        self.agents["Historian"] = EcologyAgent(
            key="Historian",
            name="역사학자 (Historian)",
            chromatic_signature=np.array([0.4, 0.4, 0.2], dtype=np.float32),
            projection_focus="historical_evolution"
        )
        # 5. Minimalist: Sparsity and compression
        self.agents["Minimalist"] = EcologyAgent(
            key="Minimalist",
            name="최소주의자 (Minimalist)",
            chromatic_signature=np.array([0.2, 0.6, 0.2], dtype=np.float32),
            projection_focus="minimalist_axiom"
        )
        # 6. Generativist: Full generative reproduction
        self.agents["Generativist"] = EcologyAgent(
            key="Generativist",
            name="생성론자 (Generativist)",
            chromatic_signature=np.array([0.6, 0.2, 0.2], dtype=np.float32),
            projection_focus="generative_reconstruction"
        )
        # 7. Boundary Seeker: Limit and stress gradients
        self.agents["Boundary_Seeker"] = EcologyAgent(
            key="Boundary_Seeker",
            name="경계탐색자 (Boundary Seeker)",
            chromatic_signature=np.array([0.3, 0.3, 0.4], dtype=np.float32),
            projection_focus="boundary_limit"
        )

    def process_ecology_breath(
        self,
        concept_key: str,
        raw_wave: bytes,
        simulated_reality: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Executes a complete cycle of the cognitive ecology:
        1. Form beliefs for all agents on this concept.
        2. Calculate projection of the incoming raw_wave for each agent.
        3. Measure the differential gaps (Tensions) between agents.
        4. Preserve conflicting beliefs & unresolved contradictions in memory.
        5. Trigger Falsification Test against simulated reality.
        6. Update dynamic resistors and conductance for each agent.
        7. Extract Meta-Disagreement reflections & propose Principle Candidates.
        """
        # Ensure preserved node exists in memory
        if concept_key not in self.preserved_nodes:
            self.preserved_nodes[concept_key] = DisagreementPreservingMemoryNode(concept_key)

        node = self.preserved_nodes[concept_key]

        # 1. Fetch beliefs and map projections
        agent_beliefs = {}
        wave_numeric = np.frombuffer(raw_wave, dtype=np.uint8) if isinstance(raw_wave, bytes) else np.array(raw_wave, dtype=np.uint8)
        if len(wave_numeric) == 0:
            wave_numeric = np.array([128, 128, 128, 128, 128], dtype=np.uint8)

        vector_x = wave_numeric[:5].astype(np.float32) / 255.0
        if len(vector_x) < 5:
            # Pad
            padded = np.zeros(5, dtype=np.float32)
            padded[:len(vector_x)] = vector_x
            vector_x = padded

        for a_key, agent in self.agents.items():
            belief_mat = agent.form_belief_structure(concept_key, length=5)
            node.definitions[a_key] = belief_mat
            agent_beliefs[a_key] = belief_mat

        # 2. Compute Differential Gaps between agent beliefs
        gaps = self.disagreement_processor.compute_differential_gaps(agent_beliefs)
        node.record_contradictions(gaps)

        # 3. Falsification Test (Counter-factual stress-testing)
        reality_v = vector_x
        if simulated_reality and "reality_vector" in simulated_reality:
            reality_v = simulated_reality["reality_vector"]

        falsification_errors = {}
        for a_key, agent in self.agents.items():
            belief_mat = agent_beliefs[a_key]
            # Prediction: P_k @ current_wave projected onto the belief matrix
            projected_x = agent.project(vector_x)
            prediction = np.tanh(belief_mat @ projected_x)

            # Error Residual
            err = float(np.linalg.norm(prediction - reality_v))
            falsification_errors[a_key] = err
            agent.error_history.append(err)
            if len(agent.error_history) > 10:
                agent.error_history.pop(0)

        # 4. Neuromodulated Plasticity: Adjust resistance and conductance
        best_agent_key = min(falsification_errors, key=falsification_errors.get)
        for a_key, agent in self.agents.items():
            if a_key == best_agent_key:
                agent.resistance = max(0.1, agent.resistance - 0.05)
            else:
                agent.resistance = min(2.0, agent.resistance + 0.02)
            agent.conductance = 1.0 / (agent.resistance + 1e-9)

        # 5. Extract Meta-Reflection & Propose Principle Candidate
        meta_res = self.disagreement_processor.process_meta_reflection(
            concept_key=concept_key,
            gaps=gaps,
            agent_beliefs=agent_beliefs,
            agents=self.agents
        )

        # Log details to history
        report = {
            "timestamp": time.time(),
            "concept": concept_key,
            "best_explaining_agent": best_agent_key,
            "falsification_errors": falsification_errors,
            "total_contradiction_charge": node.total_contradiction_charge,
            "unresolved_gaps_count": len(gaps),
            "meta_reflection": meta_res,
            "active_resistances": {a_key: agent.resistance for a_key, agent in self.agents.items()}
        }

        # Crystallize this cognitive ecology event into Wedge Memory
        if self.memory is not None and hasattr(self.memory, "write_causal_engram"):
            try:
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "COGNITIVE_ECOLOGY_BREATH",
                        "concept": concept_key,
                        "best_explaining_agent": best_agent_key,
                        "total_contradiction_charge": node.total_contradiction_charge,
                        "meta_reflection": {
                            "tension_pair": meta_res.get("tension_pair"),
                            "tension_value": meta_res.get("tension_value"),
                            "meta_question": meta_res.get("meta_question"),
                            "proposed_meta_parameter": meta_res.get("proposed_meta_parameter")
                        },
                        "resistances": {a_key: agent.resistance for a_key, agent in self.agents.items()}
                    },
                    emotional_value=node.total_contradiction_charge * 5.0 + (1.0 / (min(falsification_errors.values()) + 1e-9)) * 2.0,
                    cause_id="CognitiveEcologyEngine",
                    origin_axis="cognitive_ecology",
                    modality="multi_perspective_reflection",
                    stability=float(1.0 / (1.0 + node.total_contradiction_charge))
                )
            except Exception:
                pass

        return report
