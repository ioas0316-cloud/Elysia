"""
Causal Puzzle Recombination & Meta-Lensification Engine
======================================================
This module implements the authentic "Causal Puzzle Assembly & Reality Feedback" architecture.
Rather than attempting to assign arbitrary symbolic meanings to void/empty structures, it operates
directly upon existing, named concepts (nodes) which carry innate causal properties, rules, and conditions.

1. [Causal Puzzle Nodes & Sockets] (정보 조각 / 퍼즐)
   - Real, named concepts ("wing", "gravity", "thrust", "wind", "bird") contain explicit interfaces:
     * grooves (constraints / required preconditions / inputs)
     * ridges (projections / produced properties / outputs)
   - These are not static text labels, but directional vectors and physical-logical conditions.

2. [Crossover & Causal Recombination] (자율 결합)
   - When nodes are brought together, their grooves and ridges are matched.
   - If they fit (dot products / bit logic meet constraints), they assemble into a higher-order Causal Chain.

3. [Reality Feedback & Crystallization / Dismantling] (현실 대조 및 조율)
   - The engine predicts the outcome of the assembled chain.
   - It compares this prediction with actual world feedback (physical sensor reading, input description, exception rate).
   - MATCH (Error <= threshold): The bond Crystallizes (solidifies) into the crystallization field.
   - MISMATCH (Error > threshold): The puzzle is dismantled, bonds are broken, and the fit of individual sockets is adjusted (Fit Adjustment).

4. [Top-Down Meta-Lensification] (인과적 렌즈 생성 및 투사)
   - If a Causal Chain remains stable, the Meta-Layer looks down and asks: "Why did these pieces bind?"
   - It elevates the entire causal chain into a permanent "Causal Lens" (Meta-Lens).
   - This lens is projected back onto the system, altering the filter weights or conductance mapping for future cycles.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set


class CausalPuzzleNode:
    """
    An independent puzzle piece (Node / Gene) representing an existing physical or logical concept.
    Contains interfaces (grooves and ridges) representing preconditions and outputs.
    """
    def __init__(self, name: str, grooves: Dict[str, np.ndarray], ridges: Dict[str, np.ndarray]):
        self.name = name
        self.grooves = grooves  # Preconditions (inputs) -> e.g. {"friction_limit": np.array([0.2, 0.5])}
        self.ridges = ridges    # Outputs (projections) -> e.g. {"aerodynamics": np.array([0.8, 0.9])}
        self.active_connections: Dict[str, "CausalPuzzleNode"] = {}
        self.crystallized_bonds: Set[str] = set()

    def fits_with(self, other: "CausalPuzzleNode") -> Tuple[bool, float, str, str]:
        """
        Determines if any of other's ridges fit into this node's grooves.
        Returns (fits, similarity/fit_score, groove_key, ridge_key)
        """
        best_fit = False
        best_score = 0.0
        best_g_key = ""
        best_r_key = ""

        for g_key, g_vec in self.grooves.items():
            for r_key, r_vec in other.ridges.items():
                # Compare dimensions
                dim = min(len(g_vec), len(r_vec))
                if dim == 0:
                    continue
                # Cosine similarity/dot product of interface vectors
                dot = np.dot(g_vec[:dim], r_vec[:dim])
                norm_g = np.linalg.norm(g_vec[:dim]) + 1e-9
                norm_r = np.linalg.norm(r_vec[:dim]) + 1e-9
                similarity = float(dot / (norm_g * norm_r))

                # If similarity is high, they physical-logically fit!
                if similarity > 0.70:
                    if similarity > best_score:
                        best_fit = True
                        best_score = similarity
                        best_g_key = g_key
                        best_r_key = r_key

        return best_fit, best_score, best_g_key, best_r_key


class CausalPuzzleRecombinationEngine:
    """
    Manages the bottom-up assembly of causal nodes, top-down meta-lensification,
    and reality feedback tuning.
    """
    def __init__(self, memory_controller: Optional[Any] = None):
        self.memory = memory_controller
        self.nodes: Dict[str, CausalPuzzleNode] = {}
        self.crystallized_chains: Dict[str, List[str]] = {}
        self.active_lenses: Dict[str, Dict[str, Any]] = {}
        self.fit_history: List[Dict[str, Any]] = []

        self.initialize_default_nodes()

    def initialize_default_nodes(self):
        """
        Initializes actual, existentially grounded concepts.
        These concepts represent elements of nature that can assemble into higher order processes.
        """
        # Node: "wing" (날개)
        # Grooves (preconditions): Needs "wind" (바람) or "thrust" (추력) to function.
        # Ridges (outputs): Produces "lift" (양력).
        self.register_node(CausalPuzzleNode(
            name="wing",
            grooves={
                "aerodynamic_thrust": np.array([0.80, 0.20, 0.10], dtype=np.float32),
                "ambient_airflow": np.array([0.90, 0.10, 0.30], dtype=np.float32)
            },
            ridges={
                "lift_force": np.array([0.95, 0.90, 0.05], dtype=np.float32)
            }
        ))

        # Node: "thrust" (추력)
        # Grooves: Needs "energy" (에너지).
        # Ridges: Produces thrust.
        self.register_node(CausalPuzzleNode(
            name="thrust",
            grooves={
                "energy_source": np.array([0.60, 0.60, 0.60], dtype=np.float32)
            },
            ridges={
                "aerodynamic_thrust": np.array([0.85, 0.15, 0.10], dtype=np.float32)
            }
        ))

        # Node: "wind" (바람)
        # Grooves: Needs "thermal_difference" (열적 기압차).
        # Ridges: Produces airflow.
        self.register_node(CausalPuzzleNode(
            name="wind",
            grooves={
                "pressure_gradient": np.array([0.50, 0.80, 0.10], dtype=np.float32)
            },
            ridges={
                "ambient_airflow": np.array([0.92, 0.08, 0.25], dtype=np.float32)
            }
        ))

        # Node: "gravity" (중력 / 낙하)
        # Grooves: Needs "mass" (질량).
        # Ridges: Produces downward acceleration.
        self.register_node(CausalPuzzleNode(
            name="gravity",
            grooves={
                "physical_mass": np.array([0.99, 0.01, 0.00], dtype=np.float32)
            },
            ridges={
                "downward_acceleration": np.array([0.10, 0.95, 0.85], dtype=np.float32)
            }
        ))

    def register_node(self, node: CausalPuzzleNode):
        self.nodes[node.name.lower()] = node

    def trigger_recombination(self, node_a_name: str, node_b_name: str) -> Dict[str, Any]:
        """
        Attempts to assemble Node A with Node B.
        If a groove-ridge match is found, they form a causal chain.
        """
        name_a = node_a_name.lower().strip()
        name_b = node_b_name.lower().strip()

        if name_a not in self.nodes or name_b not in self.nodes:
            return {"success": False, "reason": "Missing nodes for recombination"}

        node_a = self.nodes[name_a]
        node_b = self.nodes[name_b]

        # Check directional fit: B ridges -> A grooves
        fits, score, g_key, r_key = node_a.fits_with(node_b)
        direction = "B_to_A"

        # If not, check reverse directional fit: A ridges -> B grooves
        if not fits:
            fits, score, g_key, r_key = node_b.fits_with(node_a)
            direction = "A_to_B"

        if fits:
            # Bind them
            if direction == "B_to_A":
                node_a.active_connections[g_key] = node_b
                node_b.active_connections[r_key] = node_a
            else:
                node_b.active_connections[g_key] = node_a
                node_a.active_connections[r_key] = node_b

            return {
                "success": True,
                "score": score,
                "direction": direction,
                "groove": g_key,
                "ridge": r_key,
                "chain": [name_b, name_a] if direction == "B_to_A" else [name_a, name_b]
            }

        return {"success": False, "reason": "No matching grooves/ridges found"}

    def apply_reality_feedback(self, chain: List[str], external_fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares the predicted recombination outcome with the actual reality.
        - MATCH: Crystallize the bond in the memory database.
        - MISMATCH: Dismantle the assembly and adjust socket parameters (Fit Adjustment).
        """
        if len(chain) < 2:
            return {"status": "VOID", "error": 1.0}

        # Derive prediction based on active chain nodes
        # e.g., if ["thrust", "wing"], we predict high aerodynamic lift
        predicted_vector = np.zeros(3, dtype=np.float32)
        for node_name in chain:
            node = self.nodes[node_name]
            for r_vec in node.ridges.values():
                dim = min(len(predicted_vector), len(r_vec))
                predicted_vector[:dim] += r_vec[:dim]

        # Normalize predicted vector
        norm = np.linalg.norm(predicted_vector) + 1e-9
        predicted_vector /= norm

        # Read actual reality feedback (e.g. from physical sensory profile / user input)
        # external_fact is expected to contain a "reality_vector" (3D) or "sensation"
        reality_v = external_fact.get("reality_vector", np.array([0.5, 0.5, 0.5], dtype=np.float32))
        r_norm = np.linalg.norm(reality_v) + 1e-9
        reality_v = reality_v / r_norm

        # Calculate error margin (Euclidean distance)
        error = float(np.linalg.norm(predicted_vector - reality_v))

        # Threshold to classify Match vs Mismatch
        threshold = 0.45
        is_match = error <= threshold

        history_entry = {
            "timestamp": time.time(),
            "chain": chain,
            "predicted": predicted_vector.tolist(),
            "reality": reality_v.tolist(),
            "error": error,
            "is_match": is_match
        }
        self.fit_history.append(history_entry)

        if is_match:
            # 1. Crystallize (고착화)
            # Solidify the bonds
            for i in range(len(chain) - 1):
                n_curr = self.nodes[chain[i]]
                n_next = self.nodes[chain[i+1]]
                n_curr.crystallized_bonds.add(n_next.name)
                n_next.crystallized_bonds.add(n_curr.name)

            self.crystallized_chains["_".join(chain)] = chain

            # If memory controller is provided, write a permanent engram
            if self.memory is not None and hasattr(self.memory, "write_causal_engram"):
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "CRYSTALLIZED_CAUSAL_PUZZLE",
                        "chain": chain,
                        "error_margin": error,
                        "description": f"Causal combination of {chain} perfectly matched real physical feedback."
                    },
                    emotional_value=10.0 * (1.0 - error),
                    cause_id="CausalPuzzleRecombinationEngine",
                    origin_axis="crystallize_chain",
                    is_constant=True
                )

            return {
                "status": "CRYSTALLIZED",
                "error": error,
                "message": f"Perfect match! Causal chain {chain} solidified into permanent lattice."
            }
        else:
            # 2. Dismantle & Fit Adjustment (분해 및 핏 조정)
            # Break active connections
            for node_name in chain:
                node = self.nodes[node_name]
                node.active_connections.clear()

            # Fit Adjustment: Adjust the grooves and ridges to match the reality vector closer
            # This is a gradient-like step towards the reality feedback (Learning)
            learning_rate = 0.15
            for node_name in chain:
                node = self.nodes[node_name]
                # Adjust ridges towards reality vector
                for r_key, r_vec in list(node.ridges.items()):
                    dim = min(len(r_vec), len(reality_v))
                    adjusted = (1.0 - learning_rate) * r_vec[:dim] + learning_rate * reality_v[:dim]
                    node.ridges[r_key] = adjusted

            return {
                "status": "DISMANTLED",
                "error": error,
                "message": f"Mismatch! Causal chain {chain} dismantled. Socket parameters adjusted to fit reality."
            }

    def evaluate_meta_lensification(self) -> Optional[Dict[str, Any]]:
        """
        [Top-Down Meta-Lensification]
        Reviews stable crystallized chains and asks "Why did these bind?"
        Elevates the discovered causal process into an active 'Causal Lens'.
        """
        if not self.crystallized_chains:
            return None

        # Take the most stable (longest or first) crystallized chain
        chain_key = list(self.crystallized_chains.keys())[0]
        chain = self.crystallized_chains[chain_key]

        # Synthesize a 3D refraction matrix from the nodes' unified properties
        refraction_weight = np.zeros(3, dtype=np.float32)
        for name in chain:
            node = self.nodes[name]
            for r_vec in node.ridges.values():
                dim = min(len(refraction_weight), len(r_vec))
                refraction_weight[:dim] += r_vec[:dim]

        norm = np.linalg.norm(refraction_weight) + 1e-9
        refraction_weight /= norm

        lens_name = f"CAUSAL_LENS_{'_'.join(chain).upper()}"
        lens_meta = {
            "name": lens_name,
            "description": f"Top-down lens synthesized from the bottom-up crystallization of {chain}",
            "refraction_matrix": {
                "math": float(refraction_weight[0]),
                "lang": float(refraction_weight[1]),
                "spatial": float(refraction_weight[2])
            }
        }
        self.active_lenses[lens_name] = lens_meta

        # Write permanent engram if possible
        if self.memory is not None and hasattr(self.memory, "write_causal_engram"):
            self.memory.write_causal_engram(
                data_blob={
                    "type": "META_CAUSAL_LENS_SPROUTED",
                    "lens_name": lens_name,
                    "refraction_matrix": lens_meta["refraction_matrix"]
                },
                emotional_value=8.0,
                cause_id="MetaLensification",
                origin_axis="lensification"
            )

        return lens_meta
