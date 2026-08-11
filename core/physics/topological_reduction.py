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

    def map_multimodal_to_network(self, modality_data: Dict[str, Any]):
        """
        [Modality-Agnostic Projection Map]
        Translates raw multi-modal inputs into internal network conductances and boundary currents.

        - "language": string or token list. Word length and character hashes shape internal paths.
        - "visual": RGB values or features. Color intensity scales path resistance.
        - "physical": Autonomic metrics (CPU, Memory, IO). High hardware pressure adjusts the variable resistor.
        """
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

    def diffuse(self, latent_potential: float) -> np.ndarray:
        """
        [Generative Decanter / Diffusion]
        Symmetric inverse operation: diffuses a single 1D latent equivalent potential back into
        the high-dimensional state vector of all network nodes by solving the nodal voltage equation.

        We fix boundary potentials: V[0] = latent_potential, V[1] = 0.0, and solve G_II * V_I = -G_IB * V_B
        """
        self._rebuild_laplacian_diagonals()

        # Potentials vector (V)
        V = np.zeros(self.num_nodes, dtype=np.float32)
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

        return V

    def run_self_refinement_loop(self, target_potential: float, max_steps: int = 15, lr: float = 0.2) -> Dict[str, Any]:
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

        for step in range(max_steps):
            # 1. Inverse Diffusion: Generate the high-dimensional node potentials from current intent
            node_potentials = self.diffuse(target_potential)

            # 2. Forward Compression: Condense the current network state to find the actual equivalent resistance & potential
            G_red, R_eq = self.compress()

            # The actual reconstructed equivalent potential of the boundary
            reconstructed_potential = target_potential / max(1e-6, R_eq)

            # 3. Calculate residual gap (error)
            residual = target_potential - reconstructed_potential

            history_potentials.append(float(reconstructed_potential))
            history_residuals.append(float(residual))

            # 4. Self-Correcting Hebbian Update with Memristive Non-linear dynamics:
            # Tune the internal conductances based on local node potential differences and global residual
            # Delta G_ij = lr * residual * (V_i - V_j)^2 * MemristorState_ij
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if self.conductance_matrix[i, j] > 0.0: # Only adapt existing paths
                        v_diff = node_potentials[i] - node_potentials[j]
                        v_diff_sq = v_diff ** 2

                        # Memristor State update (non-linear resistance changes depending on current/voltage history)
                        # High potential difference drives non-linear state changes (plasticity/annual rings)
                        # Symmetric non-linear state change based on magnitude of potential difference to ensure index invariance.
                        self.memristor_states[i, j] = np.clip(
                            self.memristor_states[i, j] + self.plasticity_rate * v_diff_sq,
                            0.05, 1.95
                        )
                        self.memristor_states[j, i] = self.memristor_states[i, j]

                        # Conductance adaptation modulated by the memristive state
                        memristive_modulation = self.memristor_states[i, j]
                        adjustment = lr * residual * v_diff_sq * memristive_modulation
                        # Clip adjustment to prevent numerical overshoot/instability
                        adjustment = np.clip(adjustment, -0.1, 0.1)
                        self.conductance_matrix[i, j] = np.clip(self.conductance_matrix[i, j] + adjustment, 0.01, 10.0)
                        self.conductance_matrix[j, i] = self.conductance_matrix[i, j]

            # Adjust variable resistor of the system as well
            self.variable_resistor.adjust(tension=abs(residual))

            if abs(residual) < 1e-4:
                print(f" -> [Convergence] Achieved resonance equilibrium at step {step}. Residual: {residual:.6f}")
                break

        final_G_red, final_R_eq = self.compress()

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
