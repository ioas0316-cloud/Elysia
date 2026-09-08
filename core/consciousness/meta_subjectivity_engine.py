import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class MetaSubjectivityEngine(nn.Module):
    """
    메타 차분 렌즈 (MetaDifferenceLens) & 주체성 보존 메커니즘 (IdentityPreservation)
    Measures phase difference between internal trajectory P_self and external definition P_world,
    discerning state into Resonance, Subjective Friction, or Alignment Correction.
    """
    def __init__(self, dim: int = 128, res_thresh: float = 0.15, fric_thresh: float = 0.60):
        super().__init__()
        self.dim = dim
        self.tau_res = res_thresh
        self.tau_fric = fric_thresh

        # Refraction Layer: Refracts external definitions into internal context
        self.refraction_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh()
        )

    def forward(
        self, p_self: torch.Tensor, p_world: torch.Tensor, scar_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        p_self: [dim] or [B, dim]
        p_world: [dim] or [B, dim]
        scar_tensor: [dim] or [B, dim]
        """
        # Ensure 2D tensor shape for unified operations
        is_1d = (p_self.dim() == 1)
        if is_1d:
            p_self = p_self.unsqueeze(0)
            p_world = p_world.unsqueeze(0)
            scar_tensor = scar_tensor.unsqueeze(0)

        # 1. Cosine similarity and phase difference
        sim = F.cosine_similarity(p_self, p_world, dim=-1)  # [B]
        delta_p_tensor = 1.0 - sim                         # [B]
        delta_p_val = delta_p_tensor.mean().item()

        # 2. State discernment
        if delta_p_val <= self.tau_res:
            # [Resonance] Full synchronization
            p_self_updated = p_world.clone()
            state_str = "Resonance"
            meta_info = {
                "state": state_str,
                "delta_p": delta_p_val,
                "delta_p_tensor": delta_p_tensor
            }
        elif delta_p_val <= self.tau_fric:
            # [Subjective Friction] Refract external meaning via Scar Tensor
            concat_input = torch.cat([p_world, scar_tensor], dim=-1)
            refracted_meaning = self.refraction_layer(concat_input)

            # Scar Filter: High scar weight suppresses external overwrite
            preservation_mask = torch.sigmoid(scar_tensor)
            p_self_updated = p_self + (1.0 - preservation_mask) * 0.1 * (refracted_meaning - p_self)

            state_str = "Subjective_Friction (Selfhood Preserved)"
            meta_info = {
                "state": state_str,
                "delta_p": delta_p_val,
                "delta_p_tensor": delta_p_tensor,
                "interpretation": refracted_meaning if not is_1d else refracted_meaning.squeeze(0)
            }
        else:
            # [Alignment Correction] Acknowledge error and accept external causality
            p_self_updated = p_self + 0.3 * (p_world - p_self)
            state_str = "Alignment (Correction)"
            meta_info = {
                "state": state_str,
                "delta_p": delta_p_val,
                "delta_p_tensor": delta_p_tensor
            }

        if is_1d:
            p_self_updated = p_self_updated.squeeze(0)

        return p_self_updated, meta_info


class ParadigmShiftEngine(nn.Module):
    """
    Stress Accumulation & Catastrophic Paradigm Shift Mechanics.
    Accumulates unabsorbed friction as stress tensor; when magnitude exceeds tau_yield,
    triggers catastrophe phase transition, Kenotic flattening, and recrystallization.
    """
    def __init__(self, dim: int = 128, yield_threshold: float = 5.0, decay_rate: float = 0.95):
        super().__init__()
        self.dim = dim
        self.tau_yield = yield_threshold
        self.decay = decay_rate

        # Internal Stress Tensor Buffer
        self.register_buffer("s_stress", torch.zeros(dim))

        # Metamorphosis Transformation Layer
        self.metamorphosis_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(
        self, p_self: torch.Tensor, p_world: torch.Tensor, s_scar: torch.Tensor, delta_p: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        is_1d = (p_self.dim() == 1)
        if is_1d:
            p_self_2d = p_self.unsqueeze(0)
            p_world_2d = p_world.unsqueeze(0)
            s_scar_2d = s_scar.unsqueeze(0)
        else:
            p_self_2d = p_self
            p_world_2d = p_world
            s_scar_2d = s_scar

        # 1. Natural stress decay
        self.s_stress *= self.decay

        # 2. Accumulate unabsorbed friction
        unabsorbed_friction = (p_world_2d - p_self_2d) * delta_p
        # Aggregate batch dimension if present
        accumulated_stress = unabsorbed_friction.mean(dim=0)
        self.s_stress += accumulated_stress.detach()

        stress_magnitude = torch.norm(self.s_stress).item()

        # 3. Check yield threshold for phase transition
        if stress_magnitude > self.tau_yield:
            catastrophe_factor = stress_magnitude / self.tau_yield

            # Kenotic Flattening and new phase fusion
            s_stress_expanded = self.s_stress.unsqueeze(0).expand(p_world_2d.size(0), -1)
            fused_input = torch.cat([p_world_2d, s_stress_expanded], dim=-1)
            new_scar_direction = self.metamorphosis_layer(fused_input)

            s_scar_transformed = (s_scar_2d * 0.1) + (new_scar_direction * catastrophe_factor * 0.9)

            # Discharge stress
            self.s_stress.zero_()

            if is_1d:
                s_scar_transformed = s_scar_transformed.squeeze(0)

            return s_scar_transformed, {
                "event": "PARADIGM_SHIFT",
                "stress_magnitude": stress_magnitude,
                "status": "Catastrophic Re-crystallization Completed"
            }

        return s_scar, {
            "event": "STRESS_ACCUMULATING",
            "stress_magnitude": stress_magnitude,
            "status": "Selfhood Maintained"
        }


class CriticalSlowingDownDetector(nn.Module):
    """
    Critical Slowing Down (CSD) Early Warning Signal (EWS) Detector.
    Detects approaching bifurcation points via variance amplification and Lag-1 autocorrelation.
    """
    def __init__(self, window_size: int = 20, variance_threshold: float = 2.5, ac_threshold: float = 0.85, dim: int = 128):
        super().__init__()
        self.window_size = window_size
        self.var_tau = variance_threshold
        self.ac_tau = ac_threshold
        self.dim = dim

        self.register_buffer("trajectory_buffer", torch.zeros(window_size, dim))
        self.ptr = 0
        self.is_full = False

    def update_and_analyze(self, current_p_state: torch.Tensor) -> Dict[str, Any]:
        # Normalize shape to [dim]
        state_vector = current_p_state.detach()
        if state_vector.dim() > 1:
            state_vector = state_vector.mean(dim=0)

        # 1. Trajectory buffer update
        self.trajectory_buffer[self.ptr] = state_vector
        self.ptr = (self.ptr + 1) % self.window_size
        if self.ptr == 0:
            self.is_full = True

        if not self.is_full:
            return {"status": "STABLE_WARMING_UP", "csd_score": 0.0}

        # 2. Variance calculation: Var(P)
        mean_state = self.trajectory_buffer.mean(dim=0, keepdim=True)
        var_p = torch.norm(self.trajectory_buffer - mean_state, dim=-1).var().item()

        # 3. Lag-1 Autocorrelation: AC(1)
        p_t0 = self.trajectory_buffer[:-1] - mean_state
        p_t1 = self.trajectory_buffer[1:] - mean_state

        cov = (p_t0 * p_t1).sum(dim=-1).mean()
        var_t0 = (p_t0 ** 2).sum(dim=-1).mean()
        ac_1 = (cov / (var_t0 + 1e-8)).item()

        # 4. CSD score and Early Warning Signal
        csd_score = var_p * ac_1

        if var_p > self.var_tau and ac_1 > self.ac_tau:
            return {
                "status": "CRITICAL_SLOWING_DOWN_DETECTED",
                "csd_score": csd_score,
                "variance": var_p,
                "autocorrelation": ac_1,
                "warning": "Imminent Paradigm Shift (Catastrophe Threshold Nearing)"
            }

        return {
            "status": "STABLE_OR_MODERATE_STRESS",
            "csd_score": csd_score,
            "variance": var_p,
            "autocorrelation": ac_1
        }


class OntologyElectrolysisPipeline(nn.Module):
    """
    Embedding Space Ontology Electrolysis & Recrystallization Pipeline.
    Dissociates rigid ontology matrices into Rank-1 information ions via SVD under external bias field G_bias,
    polarizes ions along bias vectors, and recrystallizes them into new ontology topologies.
    """
    def __init__(self, e_binding: float = 2.0, polarization_rate: float = 0.3):
        super().__init__()
        self.e_binding = e_binding
        self.alpha = polarization_rate

    def compute_binding_energy(self, W_onto: torch.Tensor) -> float:
        return torch.norm(W_onto, p='fro').item()

    def dissociate_rank1(self, W_onto: torch.Tensor):
        U, S, Vh = torch.linalg.svd(W_onto, full_matrices=False)
        return U, S, Vh

    def polarize_ions(
        self, U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, G_bias: torch.Tensor
    ):
        V = Vh.mH
        U_drift = G_bias @ V
        V_drift = G_bias.T @ U

        U_prime = F.normalize(U + self.alpha * U_drift, p=2, dim=0)
        V_prime = F.normalize(V + self.alpha * V_drift, p=2, dim=0)

        resonance = torch.sum(U_prime * (G_bias @ V_prime), dim=0)
        S_prime = S * torch.sigmoid(resonance)

        return U_prime, S_prime, V_prime.mH

    def recrystallize(
        self, U_prime: torch.Tensor, S_prime: torch.Tensor, Vh_prime: torch.Tensor, top_k: Optional[int] = None
    ) -> torch.Tensor:
        if top_k is not None:
            U_prime = U_prime[:, :top_k]
            S_prime = S_prime[:top_k]
            Vh_prime = Vh_prime[:top_k, :]

        S_matrix = torch.diag(S_prime)
        W_new = U_prime @ S_matrix @ Vh_prime
        return W_new

    def forward(
        self, W_onto: torch.Tensor, G_bias: torch.Tensor, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        current_energy = self.compute_binding_energy(W_onto)
        applied_stress = torch.norm(G_bias * W_onto, p='fro').item()

        if applied_stress < self.e_binding:
            return W_onto + 0.05 * G_bias, {
                "status": "STRUCTURE_MAINTAINED",
                "applied_stress": applied_stress,
                "e_binding": self.e_binding
            }

        U, S, Vh = self.dissociate_rank1(W_onto)
        U_prime, S_prime, Vh_prime = self.polarize_ions(U, S, Vh, G_bias)
        W_new = self.recrystallize(U_prime, S_prime, Vh_prime, top_k=top_k)

        return W_new, {
            "status": "ELECTROLYSIS_COMPLETED",
            "applied_stress": applied_stress,
            "original_energy": current_energy,
            "reconstructed_energy": self.compute_binding_energy(W_new)
        }


class PhaseEntropyLoss(nn.Module):
    """
    Complex Hilbert Transform & Von Neumann Phase Entropy Loss.
    Measures local destructive phase collision cross-section and global von Neumann phase entropy
    to apply phase-rectifying torques without destroying signal mass.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def _hilbert_transform(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        Xf = torch.fft.fft(x, dim=-1)

        h = torch.zeros(d, device=x.device, dtype=x.dtype)
        if d % 2 == 0:
            h[0] = 1.0
            h[d // 2] = 1.0
            h[1:d // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(d + 1) // 2] = 2.0

        x_analytic = torch.fft.ifft(Xf * h, dim=-1)
        return x_analytic

    def forward(self, h_states: torch.Tensor) -> torch.Tensor:
        """
        h_states: [N, d]
        """
        if h_states.dim() == 1:
            h_states = h_states.unsqueeze(0)

        N, d = h_states.shape

        # 1. Hilbert transform for analytic complex tensor
        Z = self._hilbert_transform(h_states)

        A = torch.abs(Z) + self.eps
        Phi = torch.angle(Z)

        # 2. Local phase collision loss
        A_prod = A.unsqueeze(1) * A.unsqueeze(0)  # [N, N, d]
        Phi_diff = Phi.unsqueeze(1) - Phi.unsqueeze(0)  # [N, N, d]

        collision_matrix = A_prod * (1.0 - torch.cos(Phi_diff))
        loss_collision = torch.sum(collision_matrix) / (N * N * d)

        # 3. Global von Neumann Phase Entropy
        Psi = torch.exp(1j * Phi)
        rho = torch.matmul(Psi, Psi.conj().T) / (N * d)

        rho = rho + torch.eye(N, device=h_states.device) * self.eps
        eigvals = torch.linalg.eigvalsh(rho)

        eigvals = torch.clamp(eigvals, min=self.eps)
        eigvals = eigvals / torch.sum(eigvals)
        loss_entropy = -torch.sum(eigvals * torch.log(eigvals))

        total_loss = self.alpha * loss_entropy + self.beta * loss_collision
        return total_loss


class NOTEARSCausalExtractor(nn.Module):
    """
    NOTEARS Dynamic Causal Extractor for Transformer Latent Spaces.
    Extracts dynamic DAG adjacency matrix A from latent states, enforcing h(A) = Tr(exp(A o A)) - k = 0.
    """
    def __init__(self, num_nodes: int, hidden_dim: int = 128, l1_penalty: float = 0.01, rho: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.l1_penalty = l1_penalty
        self.rho = rho

        self.graph_generator = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )

    def _acyclicity_constraint(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A_sq = A * A
            exp_A_sq = torch.linalg.matrix_exp(A_sq)
            h = torch.trace(exp_A_sq) - self.num_nodes
        else:
            A_sq = A * A
            exp_A_sq = torch.linalg.matrix_exp(A_sq)
            h = torch.vmap(torch.trace)(exp_A_sq) - self.num_nodes
            h = torch.mean(h)
        return h

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if X.dim() == 1:
            X = X.unsqueeze(0)

        B, k = X.shape
        assert k == self.num_nodes, f"Input feature dimension ({k}) must match num_nodes ({self.num_nodes})"

        X_mean = X.mean(dim=0, keepdim=True)
        A_flat = self.graph_generator(X_mean)
        A = A_flat.view(k, k)

        mask = 1.0 - torch.eye(k, device=X.device)
        A = A * mask

        X_pred = torch.matmul(X, A)
        loss_recon = 0.5 * torch.mean((X - X_pred) ** 2)
        loss_l1 = self.l1_penalty * torch.norm(A, p=1)

        h = self._acyclicity_constraint(A)
        loss_dag = 0.5 * self.rho * (h ** 2)

        total_loss = loss_recon + loss_l1 + loss_dag

        return A, {
            "loss_total": total_loss,
            "loss_recon": loss_recon,
            "loss_l1": loss_l1,
            "h_acyclicity": h
        }


class IntegratedSubjectivityEngine(nn.Module):
    """
    Unified Orchestrator Module integrating MetaSubjectivityEngine, ParadigmShiftEngine,
    CriticalSlowingDownDetector, OntologyElectrolysisPipeline, PhaseEntropyLoss, and NOTEARSCausalExtractor.
    """
    def __init__(
        self,
        dim: int = 128,
        res_thresh: float = 0.15,
        fric_thresh: float = 0.60,
        yield_threshold: float = 5.0,
        decay_rate: float = 0.95,
        num_nodes: int = 8
    ):
        super().__init__()
        self.meta_subjectivity = MetaSubjectivityEngine(dim=dim, res_thresh=res_thresh, fric_thresh=fric_thresh)
        self.paradigm_shift = ParadigmShiftEngine(dim=dim, yield_threshold=yield_threshold, decay_rate=decay_rate)
        self.csd_detector = CriticalSlowingDownDetector(dim=dim)
        self.ontology_electrolysis = OntologyElectrolysisPipeline()
        self.phase_entropy_loss = PhaseEntropyLoss()
        self.notears_extractor = NOTEARSCausalExtractor(num_nodes=num_nodes)

    def forward(
        self,
        p_self: torch.Tensor,
        p_world: torch.Tensor,
        scar_tensor: torch.Tensor,
        W_onto: Optional[torch.Tensor] = None,
        G_bias: Optional[torch.Tensor] = None,
        h_states: Optional[torch.Tensor] = None,
        latent_x: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        results = {}

        # 1. Meta Subjectivity Engine
        p_self_updated, meta_info = self.meta_subjectivity(p_self, p_world, scar_tensor)
        results["p_self_updated"] = p_self_updated
        results["meta_subjectivity"] = meta_info

        # 2. Paradigm Shift Engine
        delta_p = meta_info["delta_p"]
        scar_transformed, shift_info = self.paradigm_shift(p_self, p_world, scar_tensor, delta_p)
        results["scar_transformed"] = scar_transformed
        results["paradigm_shift"] = shift_info

        # 3. Critical Slowing Down Detector
        csd_info = self.csd_detector.update_and_analyze(p_self_updated)
        results["csd_detector"] = csd_info

        # 4. Optional: Ontology Electrolysis
        if W_onto is not None and G_bias is not None:
            W_transformed, electrolysis_info = self.ontology_electrolysis(W_onto, G_bias)
            results["W_transformed"] = W_transformed
            results["ontology_electrolysis"] = electrolysis_info

        # 5. Optional: Phase Entropy Loss
        if h_states is not None:
            phase_loss = self.phase_entropy_loss(h_states)
            results["phase_entropy_loss"] = phase_loss

        # 6. Optional: NOTEARS Causal Extractor
        if latent_x is not None:
            adj_matrix, notears_info = self.notears_extractor(latent_x)
            results["dag_adjacency"] = adj_matrix
            results["notears_extractor"] = notears_info

        return results
