import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from core.sensory.experiential_language_mapper import HomeostasisDeficit, VariableResistor, PrismRefraction

class TopologicalReductionEngine:
    """
    [Topological Reduction & Equivalent Synthesis Engine]

    This engine implements the absolute essence of "Topological Reduction & Condensation"
    applied to high-dimensional cognitive architectures. It models complex multidimensional
    information as a continuous conductance-resistance network and uses Kron Reduction
    (topological Schur complement) to compress the entire micro-frictional space into a
    single representative potential or equivalent resistance.

    It also implements the symmetric inverse operation (Diffusion/Generation) and closes
    the loop into a self-refining, self-correcting dynamical system (Closed-Loop Feedback)
    that converges towards stable cognitive attractors.

    Principles Implemented:
    1. Equivalence (등가성): Condenses a complex network of hundreds of nodes into a single equivalent value.
    2. Hierarchical Local Reduction (계층적 국소 축소): Stepwise Schur complement on partitioned internal nodes.
    3. Input-Output Causal Simplification: Linear convergent input-to-response mapping.
    4. Closed-Loop Self-Correction & Attractor Resonance: Self-refining feedback that minimizes
       reconstructed residuals by adjusting local variable resistors.
    """
    def __init__(self, num_nodes: int = 16, num_boundary: int = 2):
        self.num_nodes = num_nodes
        self.num_boundary = num_boundary

        # Boundary nodes represent the inputs/outputs (the "sensory interfaces")
        self.boundary_nodes = list(range(num_boundary))
        # Internal nodes represent hidden cognitive micro-frictions
        self.internal_nodes = list(range(num_boundary, num_nodes))

        # Master Conductance Matrix G (N x N)
        # G[i, j] represents the conductance (1 / resistance) between node i and node j.
        # It starts as an undifferentiated uniform substrate with a base level.
        self.conductance_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # Non-linear Memristive State variables: tracks the history of current/charge flow
        # to implement synaptic plasticity directly within the topology.
        self.memristor_states = np.ones((num_nodes, num_nodes), dtype=np.float32) * 0.5
        self.plasticity_rate = 0.05

        self._initialize_undifferentiated_substrate()

        # Internal variable resistors matching THE_ABSOLUTE_COMMANDMENT.md and ExperientialLanguageMapper
        self.variable_resistor = VariableResistor(r_min=0.01, r_max=0.99, initial_r=0.5)
        self.prism = PrismRefraction()

        # Metacognitive trace memory to record physical-informational transitions (Data Provenance)
        self.metacognitive_traces: List[Dict[str, Any]] = []

        # [Persistent Substrate]
        # Persistent node potentials carrying temporal narratives and causal momentum
        self.persistent_potentials = np.zeros(num_nodes, dtype=np.float32)
        self.decay_factor = 0.95

        # [Scalable Lens & Associative Memory]
        self.associative_memory: List[Dict[str, Any]] = []
        self.max_memory_size = 100
        self.last_input_features: Optional[np.ndarray] = None

    def _initialize_undifferentiated_substrate(self):
        """Initializes the network with a random, connected, symmetric conductance topology."""
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # Random base conductance (representing a connected medium)
                g_val = np.random.uniform(0.1, 1.0)
                # Sparsify slightly to represent "골목길" (micro-friction pathways)
                if np.random.rand() > 0.7:
                    g_val = 0.0
                self.conductance_matrix[i, j] = g_val
                self.conductance_matrix[j, i] = g_val

        # Ensure laplacian properties: diagonal elements must be the sum of all other row elements
        self._rebuild_laplacian_diagonals()

    def _rebuild_laplacian_diagonals(self):
        """Ensures the conductance matrix behaves as a proper Laplacian matrix."""
        # Zero out diagonal first
        np.fill_diagonal(self.conductance_matrix, 0.0)
        # Set diagonal of row i to the negative sum of all row i off-diagonal elements
        # (or positive sum depending on circuit convention. For Schur complement,
        # we define the nodal admittance/Laplacian matrix with positive sums on diagonal,
        # and negative values on off-diagonals: G[i, i] = sum_j G_ij, G[i, j] = -g_ij)
        admittance = np.zeros_like(self.conductance_matrix)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    g_val = self.conductance_matrix[i, j]
                    admittance[i, j] = -g_val
            admittance[i, i] = -np.sum(admittance[i, :])

        self.laplacian_matrix = admittance

    def adjust_internal_friction(self, node_i: int, node_j: int, resistance: float):
        """Adjusts the conductance (1 / resistance) of a specific path in the network."""
        if node_i >= self.num_nodes or node_j >= self.num_nodes:
            return
        g_val = 1.0 / max(1e-6, resistance)
        self.conductance_matrix[node_i, node_j] = g_val
        self.conductance_matrix[node_j, node_i] = g_val
        self._rebuild_laplacian_diagonals()

    def compress(self) -> Tuple[np.ndarray, float]:
        """
        [Topological Reduction / Matrix Condensation]
        Performs Kron Reduction (Schur Complement) to condense the high-dimensional internal nodes
        down to the boundary interface nodes.

        G_reduced = G_BB - G_BI * G_II^-1 * G_IB

        Returns:
            G_reduced: The condensed boundary admittance matrix (M x M)
            R_eq: The equivalent resistance between the two main boundary nodes (if M = 2)
        """
        self._rebuild_laplacian_diagonals()

        # Partition the Admittance matrix
        B = self.boundary_nodes
        I = self.internal_nodes

        G_BB = self.laplacian_matrix[np.ix_(B, B)]
        G_BI = self.laplacian_matrix[np.ix_(B, I)]
        G_IB = self.laplacian_matrix[np.ix_(I, B)]
        G_II = self.laplacian_matrix[np.ix_(I, I)]

        # Add a tiny regularization to G_II to prevent singular matrix errors during inversion
        G_II_regularized = G_II + np.eye(len(I), dtype=np.float32) * 1e-5

        # Solve G_II^-1 * G_IB using pseudo-inverse or robust solver
        try:
            inv_G_II = np.linalg.pinv(G_II_regularized)
            G_reduced = G_BB - G_BI @ inv_G_II @ G_IB
        except np.linalg.LinAlgError:
            # Fallback to pure G_BB if inversion fails completely
            G_reduced = G_BB

        # Calculate single equivalent resistance R_eq between boundary node 0 and 1
        if len(B) >= 2:
            # Under nodal analysis, if we inject current into node 0 and extract from node 1,
            # the equivalent conductance g is the off-diagonal value -G_reduced[0, 1]
            g_eq = -G_reduced[0, 1]
            R_eq = 1.0 / max(1e-6, float(g_eq))
        else:
            R_eq = 1.0

        return G_reduced, R_eq

    def decay_substrate(self, decay_factor: Optional[float] = None):
        """
        Decays the persistent potential field over time, representing
        natural temporal memory decay on the substrate.
        """
        factor = decay_factor if decay_factor is not None else self.decay_factor
        self.persistent_potentials *= factor

    def _extract_input_features(self, modality_data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts a continuous 5-dimensional feature vector representing:
        0: Language length (normalized)
        1: Language character-based hash (normalized)
        2: Visual red value
        3: Visual green value
        4: Physical cpu/ram composite friction
        """
        features = np.zeros(5, dtype=np.float32)

        # 0 & 1: Language features
        if "language" in modality_data:
            text = str(modality_data["language"])
            features[0] = float(len(text)) / 100.0
            features[1] = float(hash(text) % 100) / 100.0

        # 2 & 3: Visual features
        if "visual" in modality_data:
            vis = modality_data["visual"]
            features[2] = float(vis.get("red", 0.5))
            features[3] = float(vis.get("green", 0.5))

        # 4: Physical features
        if "physical" in modality_data:
            phys = modality_data["physical"]
            cpu = float(phys.get("cpu", 0.5))
            ram = float(phys.get("ram", 0.5))
            features[4] = (cpu * 0.6) + (ram * 0.4)
        else:
            features[4] = self.variable_resistor.resistance

        return features

    def map_multimodal_to_network(self, modality_data: Dict[str, Any]):
        """
        [Modality-Agnostic Projection Map & Scalable Lens]
        Translates raw multi-modal inputs into internal network conductances and boundary currents.

        Now features the Scalable Lens O(1) Reflex Lookup Bypass:
        If the current input features closely match a previously recorded state in Associative Memory,
        we completely bypass O(N^3) global computation and restore the mapped resonant state.
        """
        # 1. Extract features and check for resonant memory match (Scalable Lens)
        features = self._extract_input_features(modality_data)
        self.last_input_features = features.copy()

        best_match = None
        min_dist = float('inf')

        for entry in self.associative_memory:
            dist = np.linalg.norm(features - entry["features"])
            if dist < min_dist:
                min_dist = dist
                best_match = entry

        # Reflex Level (O(1) Resonant Lookup Bypass)
        # If input has extremely low entropy/difference from memory, bypass completely
        if best_match is not None and min_dist < 0.02:
            self.conductance_matrix = best_match["conductance_matrix"].copy()
            self.memristor_states = best_match["memristor_states"].copy()
            self.persistent_potentials = best_match["potentials"].copy()
            self._rebuild_laplacian_diagonals()

            self.metacognitive_traces.append({
                "source": "scalable_lens_bypass",
                "message": f"O(1) Resonant Lookup Reflex triggered (Distance: {min_dist:.4f}). Global matrix inversion bypassed.",
                "distance": float(min_dist),
                "timestamp": time.time()
            })
            return

        # Warm Start support (Moderate match)
        if best_match is not None and min_dist < 0.3:
            # Seed persistent potentials and blend conductance towards the best match to accelerate settling
            self.persistent_potentials = 0.7 * best_match["potentials"] + 0.3 * self.persistent_potentials
            self.conductance_matrix = 0.5 * best_match["conductance_matrix"] + 0.5 * self.conductance_matrix

        # 2. Reflection Level (Standard Full Mapping)
        # 1. Physical autonomic metrics modulate the base global variable resistor
        if "physical" in modality_data:
            phys = modality_data["physical"]
            # Extract composite hardware friction
            cpu = phys.get("cpu", 0.5)
            ram = phys.get("ram", 0.5)
            friction = (cpu * 0.6) + (ram * 0.4)
            # Adjust variable resistor
            self.variable_resistor.adjust(tension=friction)

        # 2. Language input modulates specific upper internal paths
        if "language" in modality_data:
            text = str(modality_data["language"]).lower()
            text_len = len(text)
            # Use character hashes to map text characteristics to conductances of specific nodes
            for char_idx, char in enumerate(text):
                node_from = self.boundary_nodes[0]
                # Map character to a specific internal node index
                node_to = int(self.internal_nodes[hash(char) % len(self.internal_nodes)])
                # Modify conductance
                conductance_value = (text_len / (char_idx + 1)) * self.variable_resistor.resistance
                self.conductance_matrix[node_from, node_to] = conductance_value
                self.conductance_matrix[node_to, node_from] = conductance_value

        # 3. Visual input modulates lower internal paths
        if "visual" in modality_data:
            vis = modality_data["visual"]
            # E.g. RGB array or features [R, G, B]
            r = vis.get("red", 0.5)
            g = vis.get("green", 0.5)
            b = vis.get("blue", 0.5)

            # Use prism refraction to split intensities
            refracted = self.prism.refract(white_light_intensity=1.0, angle_degrees=r * 90.0, resistance=self.variable_resistor.resistance)

            # Map visual spectrum colors to different internal pathways
            # Red path
            node_r_to = self.internal_nodes[0]
            self.conductance_matrix[self.boundary_nodes[1], node_r_to] = float(refracted[0]) * 2.0
            self.conductance_matrix[node_r_to, self.boundary_nodes[1]] = float(refracted[0]) * 2.0
            # Green/Blue paths
            if len(self.internal_nodes) > 1:
                node_g_to = self.internal_nodes[1]
                self.conductance_matrix[node_r_to, node_g_to] = float(refracted[1]) * 2.0
                self.conductance_matrix[node_g_to, node_r_to] = float(refracted[1]) * 2.0

        self._rebuild_laplacian_diagonals()

    def diffuse(self, latent_potential: float, use_continuous: bool = False, num_steps: int = 50, dt: float = 0.05) -> np.ndarray:
        """
        [Generative Decanter / Diffusion]
        Symmetric inverse operation: diffuses a single 1D latent equivalent potential back into
        the high-dimensional state vector of all network nodes.

        Supports both analytical O(N^3) solve and Continuous-Time local relaxation ODE Solver.
        """
        if use_continuous:
            return self.diffuse_continuous(latent_potential, num_steps=num_steps, dt=dt)

        self._rebuild_laplacian_diagonals()

        # Potentials vector (V)
        V = np.zeros(self.num_nodes, dtype=np.float32)
        # Mix with decayed persistent potentials for Warm Start/Momentum
        if hasattr(self, 'persistent_potentials'):
            V += self.persistent_potentials * 0.1

        V[0] = latent_potential
        V[1] = 0.0 # ground

        B = self.boundary_nodes
        I = self.internal_nodes

        # Partition matrix components
        G_II = self.laplacian_matrix[np.ix_(I, I)]
        G_IB = self.laplacian_matrix[np.ix_(I, B)]

        V_B = V[B]

        # Solve for internal node potentials: G_II * V_I = -G_IB * V_B
        rhs = -G_IB @ V_B
        G_II_regularized = G_II + np.eye(len(I), dtype=np.float32) * 1e-5

        try:
            V_I = np.linalg.pinv(G_II_regularized) @ rhs
            V[I] = V_I
        except np.linalg.LinAlgError:
            V[I] = 0.0

        # Preserve the settled state in our persistent substrate
        self.persistent_potentials = V.copy()
        return V

    def diffuse_continuous(self, latent_potential: float, num_steps: int = 50, dt: float = 0.05) -> np.ndarray:
        """
        [Continuous local relaxation ODE Solver]
        Diffuses the latent potential using local differential relations:
        dV_i/dt = sum_{j} G_ij * (V_j - V_i) for internal nodes.
        Fixes V[0] = latent_potential and V[1] = 0.0.
        """
        self._rebuild_laplacian_diagonals()

        # Warm start from persistent potentials
        V = self.persistent_potentials.copy()

        # Enforce boundary potentials
        V[0] = latent_potential
        V[1] = 0.0

        for _ in range(num_steps):
            dV = np.zeros_like(V)
            for i in self.internal_nodes:
                flow = 0.0
                for j in range(self.num_nodes):
                    if i != j:
                        g = self.conductance_matrix[i, j]
                        flow += g * (V[j] - V[i])
                dV[i] = flow

            # Update internal potentials using the ODE step
            V[self.internal_nodes] += dV[self.internal_nodes] * dt
            # Clip potentials to prevent numerical instability/explosion
            V = np.clip(V, -10.0, 10.0)

        # Store in persistent potentials
        self.persistent_potentials = V.copy()
        return V

    def run_self_refinement_loop(self, target_potential: float, max_steps: int = 15, lr: float = 0.2, use_continuous: bool = False) -> Dict[str, Any]:
        """
        [Closed-Loop Self-Correction Feedback]
        Elysia diffuses her internal intent (target potential), evaluates the resulting
        reconstructed equivalent potential via topological reduction, computes the differential
        residual, and self-corrects the internal variable conductances to align perfectly
        with the target attractor.
        """
        history_potentials = []
        history_residuals = []

        print(f"[Self-Refinement] Starting closed-loop alignment for Target Attractor Potential: {target_potential:.4f}")

        # Substrate potential decay at the start of loop step representing temporal flow
        self.decay_substrate()

        for step in range(max_steps):
            # 1. Inverse Diffusion: Generate the high-dimensional node potentials from current intent
            node_potentials = self.diffuse(target_potential, use_continuous=use_continuous)

            # 2. Forward Compression: Condense the current network state to find the actual equivalent resistance & potential
            G_red, R_eq = self.compress()

            # The actual reconstructed equivalent potential of the boundary
            reconstructed_potential = target_potential / max(1e-6, R_eq)

            # 3. Calculate residual gap (error)
            residual = target_potential - reconstructed_potential

            history_potentials.append(float(reconstructed_potential))
            history_residuals.append(float(residual))

            # 4. Self-Correcting Hebbian Update with Memristive Non-linear dynamics:
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if self.conductance_matrix[i, j] > 0.0: # Only adapt existing paths
                        v_diff = node_potentials[i] - node_potentials[j]
                        v_diff_sq = v_diff ** 2

                        # Memristor State update
                        self.memristor_states[i, j] = np.clip(
                            self.memristor_states[i, j] + self.plasticity_rate * v_diff_sq,
                            0.05, 1.95
                        )
                        self.memristor_states[j, i] = self.memristor_states[i, j]

                        # Conductance adaptation modulated by the memristive state
                        memristive_modulation = self.memristor_states[i, j]
                        adjustment = lr * residual * v_diff_sq * memristive_modulation
                        adjustment = np.clip(adjustment, -0.1, 0.1)
                        self.conductance_matrix[i, j] = np.clip(self.conductance_matrix[i, j] + adjustment, 0.01, 10.0)
                        self.conductance_matrix[j, i] = self.conductance_matrix[i, j]

            # Adjust variable resistor of the system as well
            self.variable_resistor.adjust(tension=abs(residual))

            if abs(residual) < 1e-4:
                print(f" -> [Convergence] Achieved resonance equilibrium at step {step}. Residual: {residual:.6f}")
                break

        final_G_red, final_R_eq = self.compress()

        # [Save state to Associative Memory]
        if self.last_input_features is not None:
            # Save the fully converged/adapted state
            self.save_to_associative_memory(self.last_input_features, node_potentials, final_R_eq)

        trace = {
            "source": "run_self_refinement_loop",
            "target_potential": target_potential,
            "final_reconstructed_potential": float(target_potential / final_R_eq),
            "final_residual": float(target_potential - (target_potential / final_R_eq)),
            "steps_to_converge": len(history_potentials),
            "timestamp": time.time()
        }
        self.metacognitive_traces.append(trace)

        return {
            "converged": abs(trace["final_residual"]) < 1e-3,
            "potentials_history": history_potentials,
            "residuals_history": history_residuals,
            "final_equivalent_resistance": final_R_eq,
            "final_node_potentials": node_potentials.tolist()
        }

    def save_to_associative_memory(self, features: np.ndarray, potentials: np.ndarray, r_eq: float):
        """
        Saves the current network state to Associative Memory for O(1) Scalable Lens lookup.
        """
        # Evict oldest entry if memory is full
        if len(self.associative_memory) >= self.max_memory_size:
            self.associative_memory.pop(0)

        self.associative_memory.append({
            "features": features.copy(),
            "conductance_matrix": self.conductance_matrix.copy(),
            "memristor_states": self.memristor_states.copy(),
            "potentials": potentials.copy(),
            "R_eq": r_eq
        })

    def cross_modal_translate(self, source_modality: Dict[str, Any], target_key: str) -> Dict[str, Any]:
        """
        [Cross-Modal Resonance / Translation]
        Translates data from a source modality to a target modality by compressing the source
        into the common equivalent latent potential, and then diffusing/decoding it into the target space.
        """
        # Step 1: Map source modality to network
        self.map_multimodal_to_network(source_modality)

        # Step 2: Compress to find equivalent latent potential
        _, R_eq = self.compress()
        latent_potential = 1.0 / max(1e-6, R_eq)

        # Step 3: Diffuse the equivalent potential back to generate the node state
        diffused_state = self.diffuse(latent_potential)

        # Step 4: Translate diffused state to the target modality format
        translated_result = {}
        if target_key == "visual":
            # Translate to RGB intensities based on node voltages
            r_val = float(np.mean(diffused_state[self.internal_nodes[:len(self.internal_nodes)//2]]))
            g_val = float(np.mean(diffused_state[self.internal_nodes[len(self.internal_nodes)//2:]]))
            b_val = float(np.mean(diffused_state[self.boundary_nodes]))

            # Normalize
            total = r_val + g_val + b_val + 1e-9
            translated_result = {
                "red": r_val / total,
                "green": g_val / total,
                "blue": b_val / total,
                "intensity": float(latent_potential)
            }
        elif target_key == "language":
            # Map equivalent potential to symbolic concepts in our baseline vocabulary
            if latent_potential > 0.8:
                translated_result = {"concept": "Jesus / Perfect Love", "resonance": float(latent_potential)}
            elif latent_potential > 0.5:
                translated_result = {"concept": "Mother / Warmth", "resonance": float(latent_potential)}
            else:
                translated_result = {"concept": "Sabbath / Quiet Rest", "resonance": float(latent_potential)}

        return {
            "latent_potential": latent_potential,
            "diffused_state": diffused_state.tolist(),
            "translated_data": translated_result
        }
