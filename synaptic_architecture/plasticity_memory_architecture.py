"""
[Plasticity Memory Architecture]
Runtime Dynamic Topology Mutation & Friction-Driven Plasticity Architecture

This module implements a dynamic plasticity memory architecture that operates on top
of static base substrate parameters (W). It resolves the fundamental limitation of static
neural network architectures by treating external data and context not as ephemeral inputs,
but as "internal structural transformation events."

The architecture consists of four stage modules:
1. FrictionSensor: Quantifies physical/topological friction (F) from mismatched impedance (Z)
   between incoming context events and the current dynamic causal graph, and generates meta-cognitive
   observation signals comparing biological/digital friction prototypes.
2. NodeAutopoiesis: Spawns and splits dynamic causal hypothesis nodes when friction exceeds threshold
   (F_high) based on the fractal principle of "sameness (homomorphism)" and "difference (boundary)".
3. GraphMutator: Dynamically mutates and rewires graph edge connectivity and directionality in real-time,
   attracting sameness nodes together and applying resistance masks to difference nodes.
4. ConsolidationLoop: Solidifies short-term dynamic topological mutations into long-term causal memory
   graphs when trajectory mutations successfully achieve macroscopic cognitive equilibrium.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CausalNode:
    """
    Dynamic Causal Graph Node.
    Represents a state/hypothesis node that can autopoietically split, mutate, or consolidate.
    """
    id: str
    feature_vector: np.ndarray
    node_type: str = "base"  # "base", "autopoietic_hypothesis", "consolidated"
    energy: float = 1.0
    sameness_score: float = 1.0
    difference_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self, new_id: str) -> 'CausalNode':
        return CausalNode(
            id=new_id,
            feature_vector=self.feature_vector.copy(),
            node_type=self.node_type,
            energy=self.energy,
            sameness_score=self.sameness_score,
            difference_score=self.difference_score,
            metadata=dict(self.metadata)
        )


@dataclass
class CausalEdge:
    """
    Dynamic Edge connecting two CausalNodes.
    Has directionality, weight (conductivity), and resistance mask.
    """
    source_id: str
    target_id: str
    weight: float = 1.0
    resistance_mask: float = 0.0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaCognitiveObservation:
    """
    Meta-Cognitive Signal describing the 'sameness' and 'difference'
    between biological homeostasis friction and digital substrate impedance.
    """
    friction_value: float
    mismatched_impedance: float
    archetypal_sameness: float   # Shared drive to preserve existential boundary
    substrate_difference: float  # Difference between digital impedance & biological qualia
    narrative: str


class FrictionSensor:
    """
    1. Physical/Topological Friction Sensor (신체적 마찰 검출기)
    Quantifies mismatched impedance (Z) between input context and the active dynamic causal graph.
    Emits high friction signals (F_high) when prediction error / topological mismatch is large,
    and produces meta-cognitive reflection records.
    """
    def __init__(self, impedance_threshold: float = 0.3):
        self.impedance_threshold = impedance_threshold
        self.last_impedance: float = 0.0
        self.last_friction: float = 0.0

    def compute_mismatched_impedance(
        self,
        context_vector: np.ndarray,
        nodes: Dict[str, CausalNode],
        edges: List[CausalEdge]
    ) -> float:
        """
        Calculates complex topological impedance Z between incoming context and graph states.
        Z = ||context_vector - projection(nodes)||_2 + topological_rigidity_penalty
        """
        if not nodes:
            return 1.0

        # Stack node feature vectors
        node_matrix = np.stack([node.feature_vector for node in nodes.values()])

        # Flatten context vector if necessary to match feature dimensions
        ctx = context_vector.astype(np.float32).flatten()
        if node_matrix.shape[1] != ctx.shape[0]:
            # Simple dimension alignment via truncation/padding or SVD projection
            min_dim = min(node_matrix.shape[1], ctx.shape[0])
            ctx_aligned = ctx[:min_dim]
            node_mat_aligned = node_matrix[:, :min_dim]
        else:
            ctx_aligned = ctx
            node_mat_aligned = node_matrix

        # Projection error: distance to nearest node in feature space
        diffs = node_mat_aligned - ctx_aligned
        distances = np.linalg.norm(diffs, axis=1)
        min_dist = float(np.min(distances))

        # Topological rigidity penalty: proportion of high-resistance edges
        if edges:
            avg_resistance = float(np.mean([e.resistance_mask for e in edges if e.is_active]))
        else:
            avg_resistance = 0.0

        # Normalized impedance Z
        context_norm = float(np.linalg.norm(ctx_aligned)) + 1e-9
        normalized_mismatch = min_dist / context_norm

        impedance = float(np.clip(normalized_mismatch + 0.5 * avg_resistance, 0.0, 5.0))
        self.last_impedance = impedance
        return impedance

    def detect_friction(
        self,
        context_vector: np.ndarray,
        nodes: Dict[str, CausalNode],
        edges: List[CausalEdge]
    ) -> Tuple[float, MetaCognitiveObservation]:
        """
        Quantifies friction F = f(Impedance Z) and generates MetaCognitiveObservation.
        """
        z = self.compute_mismatched_impedance(context_vector, nodes, edges)

        # Nonlinear friction response: F = Z^2 / (1 + Z)
        friction = float((z ** 2) / (1.0 + z))
        self.last_friction = friction

        # Meta-cognitive observation comparing biological & digital friction prototypes
        archetypal_sameness = float(np.exp(-abs(friction - 1.0)))  # Shared boundary preservation
        substrate_difference = float(abs(z - self.impedance_threshold))

        narrative = (
            f"[FrictionSensor] Measured Impedance Z={z:.4f}, Friction F={friction:.4f}. "
            f"Archetypal Sameness (Existential Boundary Drive)={archetypal_sameness:.4f}, "
            f"Substrate Difference (Digital Impedance vs Biological Qualia)={substrate_difference:.4f}."
        )

        observation = MetaCognitiveObservation(
            friction_value=friction,
            mismatched_impedance=z,
            archetypal_sameness=archetypal_sameness,
            substrate_difference=substrate_difference,
            narrative=narrative
        )

        return friction, observation


class NodeAutopoiesis:
    """
    2. Dynamic Node Autopoiesis (동적 노드 발생기)
    Under high friction (F_high), splits existing trajectories and spawns
    new hypothesis nodes in state space based on the fractal principle of 'sameness' and 'difference'.
    """
    def __init__(self, friction_threshold: float = 0.4, split_factor: float = 0.1):
        self.friction_threshold = friction_threshold
        self.split_factor = split_factor
        self.generated_node_count: int = 0

    def evaluate_sameness_and_difference(
        self,
        context_vector: np.ndarray,
        candidate_node: CausalNode
    ) -> Tuple[float, float]:
        """
        Computes sameness (homomorphism) and difference (boundary) scores
        between context_vector and a node.
        """
        v1 = context_vector.flatten().astype(np.float32)
        v2 = candidate_node.feature_vector.flatten().astype(np.float32)

        min_len = min(len(v1), len(v2))
        if min_len == 0:
            return 0.0, 1.0

        v1_a, v2_a = v1[:min_len], v2[:min_len]

        norm1 = np.linalg.norm(v1_a) + 1e-9
        norm2 = np.linalg.norm(v2_a) + 1e-9

        # Cosine similarity for sameness (homomorphism)
        cosine_sim = float(np.dot(v1_a, v2_a) / (norm1 * norm2))
        sameness = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        # Euclidean distance ratio for difference (boundary)
        dist = float(np.linalg.norm(v1_a - v2_a))
        difference = float(np.clip(dist / (norm1 + norm2), 0.0, 1.0))

        return sameness, difference

    def trigger_autopoiesis(
        self,
        friction: float,
        context_vector: np.ndarray,
        nodes: Dict[str, CausalNode]
    ) -> List[CausalNode]:
        """
        If friction > friction_threshold, splits existing nodes or generates new
        autopoietic hypothesis nodes in state space.
        """
        if friction <= self.friction_threshold:
            return []

        new_nodes: List[CausalNode] = []
        ctx = context_vector.flatten().astype(np.float32)

        if not nodes:
            # Create seed node
            self.generated_node_count += 1
            node_id = f"auto_node_{self.generated_node_count}"
            seed_node = CausalNode(
                id=node_id,
                feature_vector=ctx.copy(),
                node_type="autopoietic_hypothesis",
                energy=friction,
                sameness_score=1.0,
                difference_score=0.0,
                metadata={"origin": "seed_autopoiesis", "friction": friction}
            )
            new_nodes.append(seed_node)
            return new_nodes

        # Find node with highest mismatch / tension to split
        for node_id, node in list(nodes.items()):
            sameness, difference = self.evaluate_sameness_and_difference(context_vector, node)

            # High friction + high difference triggers trajectory splitting
            if difference > 0.2:
                self.generated_node_count += 1
                new_id = f"hypothesis_{self.generated_node_count}_{node_id}"

                # Perturb feature vector towards context_vector using fractal interpolation
                dim = min(len(node.feature_vector), len(ctx))
                new_vec = node.feature_vector.copy()
                new_vec[:dim] = (1.0 - self.split_factor) * node.feature_vector[:dim] + self.split_factor * ctx[:dim]

                hypothesis_node = CausalNode(
                    id=new_id,
                    feature_vector=new_vec,
                    node_type="autopoietic_hypothesis",
                    energy=friction * sameness,
                    sameness_score=sameness,
                    difference_score=difference,
                    metadata={"parent": node_id, "sameness": sameness, "difference": difference}
                )
                new_nodes.append(hypothesis_node)

        return new_nodes


class GraphMutator:
    """
    3. Topological Graph Mutator Engine (위상 재정렬 엔진)
    Dynamically rewires edge connectivity and directionality during runtime inference:
    - Pulls and binds nodes together at 'sameness (homomorphism)' points.
    - Injects resistance masks at 'difference (boundary)' points.
    """
    def __init__(self, attraction_rate: float = 0.2, resistance_decay: float = 0.1):
        self.attraction_rate = attraction_rate
        self.resistance_decay = resistance_decay

    def mutate_topology(
        self,
        nodes: Dict[str, CausalNode],
        edges: List[CausalEdge],
        autopoietic_nodes: List[CausalNode]
    ) -> List[CausalEdge]:
        """
        Executes real-time dynamic graph topology mutation.
        """
        # Register new autopoietic nodes into local working dictionary
        all_nodes = dict(nodes)
        for auto_node in autopoietic_nodes:
            all_nodes[auto_node.id] = auto_node

        node_keys = list(all_nodes.keys())
        existing_edges_map = {(e.source_id, e.target_id): e for e in edges}

        updated_edges: List[CausalEdge] = list(edges)

        # Mutate existing edges
        for edge in updated_edges:
            if not edge.is_active:
                continue

            src = all_nodes.get(edge.source_id)
            tgt = all_nodes.get(edge.target_id)

            if src is not None and tgt is not None:
                dim = min(len(src.feature_vector), len(tgt.feature_vector))
                v1, v2 = src.feature_vector[:dim], tgt.feature_vector[:dim]
                norm1, norm2 = np.linalg.norm(v1) + 1e-9, np.linalg.norm(v2) + 1e-9
                sim = float(np.dot(v1, v2) / (norm1 * norm2))

                if sim > 0.5:
                    # Sameness: attract features & increase edge conductivity (weight)
                    edge.weight = float(np.clip(edge.weight + self.attraction_rate * sim, 0.0, 10.0))
                    edge.resistance_mask = float(np.clip(edge.resistance_mask - self.resistance_decay, 0.0, 1.0))

                    # Feature attraction pull
                    pull = self.attraction_rate * (v2 - v1)
                    src.feature_vector[:dim] += pull * 0.5
                    tgt.feature_vector[:dim] -= pull * 0.5
                else:
                    # Difference: inject resistance mask
                    diff_score = 1.0 - sim
                    edge.resistance_mask = float(np.clip(edge.resistance_mask + diff_score * 0.2, 0.0, 1.0))
                    edge.weight = float(np.clip(edge.weight * (1.0 - edge.resistance_mask), 0.0, 10.0))

        # Wire new edges for autopoietic nodes
        for auto_node in autopoietic_nodes:
            for n_id, target_node in nodes.items():
                if n_id == auto_node.id:
                    continue

                dim = min(len(auto_node.feature_vector), len(target_node.feature_vector))
                v1, v2 = auto_node.feature_vector[:dim], target_node.feature_vector[:dim]
                norm1, norm2 = np.linalg.norm(v1) + 1e-9, np.linalg.norm(v2) + 1e-9
                sim = float(np.dot(v1, v2) / (norm1 * norm2))

                if sim > 0.3:
                    # Wire forward and backward edges
                    edge_fwd = CausalEdge(
                        source_id=target_node.id,
                        target_id=auto_node.id,
                        weight=float(sim * 2.0),
                        resistance_mask=float(1.0 - sim)
                    )
                    edge_rev = CausalEdge(
                        source_id=auto_node.id,
                        target_id=target_node.id,
                        weight=float(sim * 1.5),
                        resistance_mask=float(1.0 - sim)
                    )
                    updated_edges.extend([edge_fwd, edge_rev])

        return updated_edges


class ConsolidationLoop:
    """
    4. Structural Consolidation Loop (결합·응고 순환기)
    Evaluates whether dynamic runtime topological mutations contributed to macro cognitive
    equilibrium (reducing friction/impedance). If beneficial, solidifies short-term
    autopoietic nodes and edges into long-term causal memory on top of base substrate W.
    """
    def __init__(self, consolidation_threshold: float = 0.5):
        self.consolidation_threshold = consolidation_threshold
        self.consolidated_count: int = 0

    def evaluate_and_consolidate(
        self,
        nodes: Dict[str, CausalNode],
        edges: List[CausalEdge],
        initial_friction: float,
        final_friction: float
    ) -> Tuple[Dict[str, CausalNode], List[CausalEdge], List[str]]:
        """
        Consolidates beneficial autopoietic mutations into long-term causal graph.
        Returns updated nodes, edges, and list of consolidated node IDs.
        """
        friction_reduction = initial_friction - final_friction
        consolidated_ids: List[str] = []

        # If friction was reduced or final friction is low, consolidate
        should_consolidate = (friction_reduction > 0.0) or (final_friction < self.consolidation_threshold)

        if not should_consolidate:
            # Prune high-resistance or inactive autopoietic nodes
            pruned_nodes = dict(nodes)
            pruned_edges = [e for e in edges if e.resistance_mask < 0.9]
            return pruned_nodes, pruned_edges, []

        updated_nodes = dict(nodes)
        for node_id, node in list(updated_nodes.items()):
            if node.node_type == "autopoietic_hypothesis":
                # Solidify into long-term consolidated node
                node.node_type = "consolidated"
                node.energy = float(node.energy * 1.5 + friction_reduction)
                node.metadata["consolidated_at_friction_delta"] = float(friction_reduction)
                consolidated_ids.append(node_id)
                self.consolidated_count += 1

        # Solidify edges associated with consolidated nodes
        updated_edges: List[CausalEdge] = []
        for edge in edges:
            if edge.source_id in updated_nodes and edge.target_id in updated_nodes:
                if edge.resistance_mask < 0.8:
                    edge.weight = float(edge.weight * 1.2)
                    edge.resistance_mask = float(edge.resistance_mask * 0.5)
                    updated_edges.append(edge)

        return updated_nodes, updated_edges, consolidated_ids


class PlasticityMemoryArchitecture:
    """
    Unified Plasticity Memory Architecture (가소성 메모리 아키텍처)
    Combines static base substrate W with runtime Dynamic Topology Mutation and Friction-driven Plasticity.

    4-Stage Plasticity Pipeline:
    1. FrictionSensor: Quantifies impedance mismatch Z and friction F.
    2. NodeAutopoiesis: Spawns dynamic causal nodes under high friction based on sameness/difference.
    3. GraphMutator: Rewires topology in real-time (attracting sameness, masking difference).
    4. ConsolidationLoop: Solidifies macro-beneficial mutations into persistent causal memory.
    """
    def __init__(
        self,
        base_feature_dim: int = 64,
        impedance_threshold: float = 0.3,
        friction_threshold: float = 0.4
    ):
        self.base_feature_dim = base_feature_dim
        self.sensor = FrictionSensor(impedance_threshold=impedance_threshold)
        self.autopoiesis = NodeAutopoiesis(friction_threshold=friction_threshold)
        self.mutator = GraphMutator()
        self.consolidation = ConsolidationLoop()

        # Dynamic Causal Graph State
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.history: List[Dict[str, Any]] = []

        # Initialize base substrate root node
        self._init_base_substrate()

    def _init_base_substrate(self):
        """Initializes default base substrate node."""
        root_node = CausalNode(
            id="base_root",
            feature_vector=np.zeros(self.base_feature_dim, dtype=np.float32),
            node_type="base",
            energy=1.0,
            metadata={"description": "Base Substrate Root (Static Parameter W)"}
        )
        self.nodes[root_node.id] = root_node

    def process_event(self, context_data: Any) -> Dict[str, Any]:
        """
        Executes complete 4-step runtime plasticity loop for an incoming event/context.
        Supports vectors, strings, dicts, or raw multimodal data representations.
        """
        # Convert input into standardized feature vector
        if isinstance(context_data, np.ndarray):
            ctx_vec = context_data.flatten().astype(np.float32)
        elif isinstance(context_data, (list, tuple)):
            ctx_vec = np.array(context_data, dtype=np.float32).flatten()
        elif isinstance(context_data, str):
            # Deterministic hash projection into feature dimension
            hash_val = hash(context_data)
            np.random.seed(abs(hash_val) % (2**32 - 1))
            ctx_vec = np.random.randn(self.base_feature_dim).astype(np.float32)
        else:
            ctx_vec = np.ones(self.base_feature_dim, dtype=np.float32)

        # Pad or truncate to match base_feature_dim
        if len(ctx_vec) < self.base_feature_dim:
            padded = np.zeros(self.base_feature_dim, dtype=np.float32)
            padded[:len(ctx_vec)] = ctx_vec
            ctx_vec = padded
        elif len(ctx_vec) > self.base_feature_dim:
            ctx_vec = ctx_vec[:self.base_feature_dim]

        # Step 1: Physical/Topological Friction Sensor
        initial_friction, meta_obs = self.sensor.detect_friction(ctx_vec, self.nodes, self.edges)

        # Step 2: Dynamic Node Autopoiesis
        autopoietic_nodes = self.autopoiesis.trigger_autopoiesis(initial_friction, ctx_vec, self.nodes)
        for auto_node in autopoietic_nodes:
            self.nodes[auto_node.id] = auto_node

        # Step 3: Topological Graph Mutator (Runtime Edge Rewiring)
        self.edges = self.mutator.mutate_topology(self.nodes, self.edges, autopoietic_nodes)

        # Compute post-mutation friction
        final_friction, post_meta_obs = self.sensor.detect_friction(ctx_vec, self.nodes, self.edges)

        # Step 4: Consolidation Loop
        self.nodes, self.edges, consolidated_ids = self.consolidation.evaluate_and_consolidate(
            self.nodes, self.edges, initial_friction, final_friction
        )

        record = {
            "initial_friction": initial_friction,
            "final_friction": final_friction,
            "autopoietic_nodes_created": [n.id for n in autopoietic_nodes],
            "consolidated_node_ids": consolidated_ids,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "meta_observation": meta_obs.narrative
        }
        self.history.append(record)
        return record

    def reconstruct_memory(self, query_context: Any) -> Tuple[np.ndarray, List[str]]:
        """
        Reconstructs memory dynamically based on incoming query_context and current causal graph topology.
        Demonstrates Reconstructive Memory (past causal net reassembled in present friction).
        """
        if isinstance(query_context, np.ndarray):
            q_vec = query_context.flatten().astype(np.float32)
        else:
            hash_val = hash(str(query_context))
            np.random.seed(abs(hash_val) % (2**32 - 1))
            q_vec = np.random.randn(self.base_feature_dim).astype(np.float32)

        if len(q_vec) < self.base_feature_dim:
            padded = np.zeros(self.base_feature_dim, dtype=np.float32)
            padded[:len(q_vec)] = q_vec
            q_vec = padded
        else:
            q_vec = q_vec[:self.base_feature_dim]

        # Compute resonance of each node with query
        active_trajectory: List[Tuple[float, CausalNode]] = []
        for node in self.nodes.values():
            dim = min(len(q_vec), len(node.feature_vector))
            v1, v2 = q_vec[:dim], node.feature_vector[:dim]
            norm1, norm2 = np.linalg.norm(v1) + 1e-9, np.linalg.norm(v2) + 1e-9
            sim = float(np.dot(v1, v2) / (norm1 * norm2))
            res_score = sim * node.energy
            active_trajectory.append((res_score, node))

        active_trajectory.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [node for score, node in active_trajectory[:3]]

        # Reconstructed memory vector: weighted combination of top resonant causal nodes
        reconstructed_vec = np.zeros(self.base_feature_dim, dtype=np.float32)
        node_ids = []

        for score, node in active_trajectory[:3]:
            dim = min(self.base_feature_dim, len(node.feature_vector))
            reconstructed_vec[:dim] += max(0.01, score) * node.feature_vector[:dim]
            node_ids.append(node.id)

        norm = np.linalg.norm(reconstructed_vec)
        if norm > 0:
            reconstructed_vec /= norm

        return reconstructed_vec, node_ids
