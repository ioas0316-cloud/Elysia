"""
Elysia Autonomous Causal Discovery & Active Inference Engine
============================================================
Implements:
1. Entity Slot Disentanglement & Invariant Node Extraction from Raw Sensorium
2. Active Inference POMDP & Continuous Dynamics (Friston's Free Energy Principle)
   - Matrix A (Likelihood), B (Transition), C (Preferences), D (Initial Belief), E (Habits)
   - Variational Free Energy (F) Minimization & Expected Free Energy (G) Decomposition
   - Epistemic (Exploration) & Pragmatic (Exploitation) Balance
   - Continuous Euler-discretized Spinal Reflex Arc Loop
3. Active do-Exploration & Differentiable DAG Discovery (NOTEARS & MDL Pruning)
4. Surprise / Prediction Error-Driven Graph Rewriting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from core.topology.executable_causal_topology import StructuralCausalModel, ExecutableDAGNode, NodeType, OpCode


def softmax(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Softmax with inverse temperature / precision gamma."""
    scaled = gamma * (x - np.max(x))
    exp_x = np.exp(scaled)
    return exp_x / np.sum(exp_x)


class SlotAttentionDisentangler:
    """
    Simulates Slot Attention / Disentanglement:
    Extracts discrete, independent latent entities/nodes from high-dimensional raw observations.
    """
    def __init__(self, num_slots: int = 4, slot_dim: int = 8):
        self.num_slots = num_slots
        self.slot_dim = slot_dim

    def extract_slots(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Disentangles raw observation matrix into slot representation vectors.
        Shape: [batch_size, num_slots, slot_dim]
        """
        raw_arr = np.atleast_2d(raw_data)
        batch_size = raw_arr.shape[0]

        # Simple deterministic projection for demonstration of disentanglement
        np.random.seed(42)
        proj_matrix = np.random.normal(0, 1, size=(raw_arr.shape[1], self.num_slots * self.slot_dim))
        latents = raw_arr @ proj_matrix
        return latents.reshape(batch_size, self.num_slots, self.slot_dim)


class InvarianceDetector:
    """
    Filters out transient noise across environment shifts to retain invariant causal nodes.
    """
    @staticmethod
    def filter_invariant_features(slot_history: List[np.ndarray], variance_threshold: float = 0.5) -> List[int]:
        """Returns indices of slots whose causal properties remain stable across environments."""
        slot_stack = np.stack(slot_history, axis=0) # [envs, batch, slots, dim]
        slot_variances = np.mean(np.var(slot_stack, axis=0), axis=-1) # [batch, slots]
        avg_var_per_slot = np.mean(slot_variances, axis=0)

        invariant_indices = [i for i, v in enumerate(avg_var_per_slot) if v < variance_threshold]
        return invariant_indices if invariant_indices else list(range(slot_stack.shape[2]))


class ActiveInferenceAgent:
    """
    Active Inference Agent based on Karl Friston's Free Energy Principle.
    Generative Model defined by POMDP Categorical Tensors:
    - A: Observation Likelihood Matrix P(o|s) [num_obs, num_states]
    - B: State Transition Tensor P(s_{t+1}|s_t, a) [num_states, num_states, num_actions]
    - C: Prior Preference Log-Dist P(o) [num_obs]
    - D: Initial Prior Belief P(s_1) [num_states]
    - E: Habitual Policy Prior P(pi) [num_policies]
    """
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        E: Optional[np.ndarray] = None,
        gamma: float = 1.0
    ):
        self.A = A
        self.B = B
        self.C = C
        self.D = D.copy()
        self.q_s = D.copy()  # Current State Belief
        self.gamma = gamma

        num_policies = B.shape[2]
        self.E = E if E is not None else np.zeros(num_policies)

    def infer_states(self, obs_idx: int) -> np.ndarray:
        """
        Perception: Minimizes Variational Free Energy (F) to update state belief q(s).
        """
        likelihood = self.A[obs_idx, :]
        unnormalized = self.q_s * likelihood
        denom = np.sum(unnormalized)
        if denom < 1e-12:
            self.q_s = np.ones_like(self.q_s) / len(self.q_s)
        else:
            self.q_s = unnormalized / denom
        return self.q_s

    def evaluate_expected_free_energy(self, policies: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Policy Evaluation: Computes Expected Free Energy G(pi) for candidate policies.
        Decomposes G into:
        - Epistemic Value (Information Gain / Exploration)
        - Pragmatic Value (Goal Preference / Exploitation)
        """
        G = np.zeros(len(policies))
        epistemic_vals = np.zeros(len(policies))
        pragmatic_vals = np.zeros(len(policies))

        for p_idx, policy in enumerate(policies):
            q_s_tau = self.q_s.copy()

            for action in policy:
                # Predict future state: q(s_tau | pi) = B(action) @ q_s_tau
                q_s_tau = self.B[:, :, action] @ q_s_tau
                # Predict future observation: q(o_tau | pi) = A @ q_s_tau
                q_o_tau = self.A @ q_s_tau + 1e-12

                # 1. Pragmatic Value (Goal Satisfaction)
                pragmatic = np.sum(q_o_tau * self.C)

                # 2. Epistemic Value (Information Gain)
                epistemic = 0.0
                for s_idx in range(len(q_s_tau)):
                    p_o_given_s = self.A[:, s_idx] + 1e-12
                    kl_div = np.sum(p_o_given_s * np.log(p_o_given_s / q_o_tau))
                    epistemic += q_s_tau[s_idx] * kl_div

                epistemic_vals[p_idx] += epistemic
                pragmatic_vals[p_idx] += pragmatic

                # G = - (Epistemic + Pragmatic)
                G[p_idx] -= (epistemic + pragmatic)

        return G, epistemic_vals, pragmatic_vals

    def select_action(self, obs_idx: int, policies: List[List[int]]) -> int:
        """Decides action by computing P(pi) = Softmax(-gamma * G + E)."""
        self.infer_states(obs_idx)
        G, _, _ = self.evaluate_expected_free_energy(policies)
        p_pi = softmax(-G + self.E, gamma=self.gamma)
        selected_p_idx = int(np.random.choice(len(policies), p=p_pi))
        return policies[selected_p_idx][0]


class ContinuousActiveInferenceReflexArc:
    """
    Continuous Space Active Inference & Predictive Coding:
    Euler-discretized Spinal Reflex Arc updating belief mu(t) and motor command action a(t).
    """
    def __init__(
        self,
        dt: float = 0.01,
        pi_y: float = 10.0,
        pi_p: float = 2.0,
        lr_mu: float = 2.0,
        lr_a: float = 5.0,
        alpha: float = 1.0
    ):
        self.dt = dt
        self.pi_y = pi_y
        self.pi_p = pi_p
        self.lr_mu = lr_mu
        self.lr_a = lr_a
        self.alpha = alpha

        self.x = 0.0   # Physical True State
        self.a = 0.0   # Action / Motor Command
        self.mu = 0.0  # Internal Belief

    def step(self, target_mu_d: float, noise_std: float = 0.01) -> Dict[str, float]:
        """One step continuous Active Inference Euler update."""
        # 1. Physical observation
        y = self.x + np.random.normal(0, noise_std)

        # 2. Prediction Errors
        e_y = self.pi_y * (y - self.mu)             # Sensory Prediction Error
        e_p = self.pi_p * (target_mu_d - self.mu)    # Prior Goal Error

        # 3. Perception (Belief update d_mu / dt)
        d_mu = e_y + e_p
        self.mu += self.dt * self.lr_mu * d_mu

        # 4. Action (Reflex arc d_a / dt = - dF/da)
        d_a = self.pi_y * (self.mu - y)
        self.a += self.dt * self.lr_a * d_a

        # 5. Environment Update (dx / dt = -alpha * x + a)
        dx = -self.alpha * self.x + self.a
        self.x += self.dt * dx

        return {
            "x": float(self.x),
            "mu": float(self.mu),
            "a": float(self.a),
            "e_y": float(e_y),
            "e_p": float(e_p)
        }


class DifferentiableCausalDiscovery:
    """
    NOTEARS-style Differentiable DAG Causal Discovery & MDL Pruning.
    Finds adjacency matrix W subject to smooth acyclicity constraint h(W) = tr(exp(W*W)) - d = 0.
    """
    @staticmethod
    def _acyclicity_constraint(W: np.ndarray) -> float:
        d = W.shape[0]
        M = W * W
        # Matrix exponential trace: tr(exp(M)) - d
        exp_M = scipy_matrix_exponential(M)
        return float(np.trace(exp_M) - d)

    @staticmethod
    def discover_dag(data_matrix: np.ndarray, l1_reg: float = 0.1, max_iter: int = 100) -> np.ndarray:
        """
        Discovers directed adjacency matrix W from observational/interventional matrix data.
        Shape of data_matrix: [num_samples, num_vars]
        """
        num_samples, d = data_matrix.shape
        W = np.zeros((d, d), dtype=np.float32)
        lr = 0.01

        # Iterative gradient step minimizing ||X - XW||_F^2 + lambda*||W||_1 + rho*h(W)^2
        for _ in range(max_iter):
            pred_err = data_matrix - (data_matrix @ W)
            grad_loss = -2.0 * (data_matrix.T @ pred_err) / num_samples

            # Subgradient for L1
            grad_l1 = l1_reg * np.sign(W)

            # Simple acyclicity penalty grad approximation
            h_val = np.trace(np.linalg.matrix_power(np.eye(d) + (W * W) / d, d)) - d
            grad_h = 2.0 * h_val * (2.0 * W)

            grad_total = grad_loss + grad_l1 + grad_h
            W -= lr * grad_total
            # Zero out self-loops
            np.fill_diagonal(W, 0.0)

        return W


def scipy_matrix_exponential(M: np.ndarray) -> np.ndarray:
    """Taylor series approximation for matrix exponential."""
    res = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for i in range(1, 10):
        term = term @ M / i
        res += term
    return res


class MDLPruner:
    """
    Minimum Description Length (MDL) Self-Pruning:
    Prunes edges from discovered adjacency matrix that increase model complexity without sufficient error reduction.
    """
    @staticmethod
    def prune_edges(W: np.ndarray, data_matrix: np.ndarray, mdl_penalty: float = 0.05) -> np.ndarray:
        W_pruned = W.copy()
        d = W.shape[0]
        num_samples = data_matrix.shape[0]

        # Calculate base loss
        base_err = np.mean((data_matrix - data_matrix @ W_pruned) ** 2)

        for i in range(d):
            for j in range(d):
                if abs(W_pruned[i, j]) > 1e-4:
                    # Test removing edge (i -> j)
                    W_test = W_pruned.copy()
                    W_test[i, j] = 0.0
                    test_err = np.mean((data_matrix - data_matrix @ W_test) ** 2)

                    # MDL trade-off: error increase vs edge parameter saving
                    if test_err - base_err < mdl_penalty:
                        W_pruned[i, j] = 0.0
                        base_err = test_err

        return W_pruned


class SurpriseGraphRewriter:
    """
    Free Energy Prediction-Error / Surprise Driven Graph Rewriting:
    Mutates internal SCM topology when prediction errors exceed threshold.
    """
    def __init__(self, scm: StructuralCausalModel, surprise_threshold: float = 0.25):
        self.scm = scm
        self.surprise_threshold = surprise_threshold

    def evaluate_and_rewrite(self, observed_data: Dict[str, float], predicted_data: Dict[str, float]) -> bool:
        """
        Calculates surprise (prediction error).
        If surprise exceeds threshold, triggers topological graph rewriting.
        Returns True if graph was modified.
        """
        total_error = 0.0
        count = 0
        for nid, obs_val in observed_data.items():
            if nid in predicted_data:
                err = abs(obs_val - predicted_data[nid])
                total_error += err
                count += 1

        avg_surprise = total_error / max(count, 1)

        if avg_surprise > self.surprise_threshold:
            # Trigger Graph Rewriting
            # Find node with highest error and add missing causal edge or split
            worst_node = max(observed_data.keys(), key=lambda k: abs(observed_data[k] - predicted_data.get(k, 0.0)))

            # If worst_node has no parents, connect it to another value node
            if worst_node in self.scm.nodes:
                node = self.scm.nodes[worst_node]
                other_nodes = [n for n in self.scm.nodes.keys() if n != worst_node]
                if other_nodes and len(node.input_ids) < 2:
                    new_parent = other_nodes[0]
                    if new_parent not in node.input_ids:
                        node.input_ids.append(new_parent)
                        node.op = OpCode.ADD if len(node.input_ids) >= 2 else node.op
                        if worst_node not in self.scm.nodes[new_parent].output_ids:
                            self.scm.nodes[new_parent].output_ids.append(worst_node)
                        return True
        return False
