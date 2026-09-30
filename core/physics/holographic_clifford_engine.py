"""
Holographic Clifford Autopoietic Engine (홀로그래픽 클리포드 자기생성적 인과 엔진)
================================================================================
Implements the unified holographic cognitive physics framework based on:
1. Clifford Algebra Cl(3,0) Multivector Fields (Scalar, Vector, Bivector, Trivector)
2. Transformer-to-Clifford Multivector Bridge & Clifford Phase-Coherence Attention
3. Spontaneous Symbol Emergence & Dynamic Rule Update Matrix (Symbol Sprouting)
4. MERA-style Hierarchical Renormalization Group (RG) Coarse-Graining & Macro-Rotor Extraction
5. Non-Equilibrium Spacetime Field Dynamics & Dual Action Engine (Contraction/Emission)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class CliffordAlgebra3D:
    """
    Clifford / Geometric Algebra Cl(3,0) structure & Cayley Table tensor engine.
    8-dimensional basis blades: [1, e1, e2, e3, e12, e13, e23, e123]
    Grade-0: 1 scalar (index 0)
    Grade-1: 3 vectors (indices 1, 2, 3 -> e1, e2, e3)
    Grade-2: 3 bivectors (indices 4, 5, 6 -> e12, e13, e23)
    Grade-3: 1 trivector / pseudoscalar (index 7 -> e123)
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.dim = 8
        self.cayley = self._build_cayley_table().to(device)
        # Reversion signs for grades [0, 1, 2, 3]: (+1, +1, -1, -1)
        self.reversion_signs = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            dtype=torch.float32,
            device=device
        )

    def _build_cayley_table(self) -> torch.Tensor:
        """Constructs 8x8x8 Structure Constant Cayley Tensor C[i, j, k] for Cl(3,0)."""
        C = torch.zeros((8, 8, 8), dtype=torch.float32)
        mult_rules = {
            (0,0):(0,1), (0,1):(1,1), (0,2):(2,1), (0,3):(3,1), (0,4):(4,1), (0,5):(5,1), (0,6):(6,1), (0,7):(7,1),
            (1,0):(1,1), (1,1):(0,1), (1,2):(4,1), (1,3):(5,1), (1,4):(2,1), (1,5):(3,1), (1,6):(7,1), (1,7):(6,1),
            (2,0):(2,1), (2,1):(4,-1),(2,2):(0,1), (2,3):(6,1), (2,4):(1,-1),(2,5):(7,-1),(2,6):(3,1), (2,7):(5,-1),
            (3,0):(3,1), (3,1):(5,-1),(3,2):(6,-1),(3,3):(0,1), (3,4):(7,1), (3,5):(1,-1),(3,6):(2,-1),(3,7):(4,1),
            (4,0):(4,1), (4,1):(2,-1),(4,2):(1,1), (4,3):(7,1), (4,4):(0,-1),(4,5):(6,-1),(4,6):(5,1), (4,7):(3,-1),
            (5,0):(5,1), (5,1):(3,-1),(5,2):(7,-1),(5,3):(1,1), (5,4):(6,1), (5,5):(0,-1),(5,6):(4,-1),(5,7):(2,1),
            (6,0):(6,1), (6,1):(7,1), (6,2):(3,-1),(6,3):(2,1), (6,4):(5,-1),(6,5):(4,1), (6,6):(0,-1),(6,7):(1,-1),
            (7,0):(7,1), (7,1):(6,1), (7,2):(5,-1),(7,3):(4,1), (7,4):(3,-1),(7,5):(2,1), (7,6):(1,-1),(7,7):(0,-1)
        }
        for (i, j), (k, sign) in mult_rules.items():
            C[i, j, k] = float(sign)
        return C

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Parallel Geometric Product A x B via einsum with Cayley tensor."""
        cayley = self.cayley.to(A.device)
        return torch.einsum('...i, ...j, ijk -> ...k', A, B, cayley)

    def revert(self, A: torch.Tensor) -> torch.Tensor:
        """Reversion Operator A~ (reverses order of basis blade multiplication)."""
        rev_signs = self.reversion_signs.to(A.device)
        return A * rev_signs

    def scalar_part(self, A: torch.Tensor) -> torch.Tensor:
        """Extracts Grade-0 scalar component."""
        return A[..., 0]

    def bivector_part(self, A: torch.Tensor) -> torch.Tensor:
        """Extracts Grade-2 bivector components (indices 4, 5, 6)."""
        return A[..., 4:7]

    def norm(self, A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculates scalar magnitude norm: sqrt(|<A A~>_0|)."""
        # For multivectors in Cl(3,0), <A A~>_0 = sum_k A_k^2
        return torch.sqrt(torch.sum(A * A, dim=-1) + eps)


class TransformerToCliffordBridge(nn.Module):
    """
    Converts Standard Transformer Embeddings (d_model) into Cl(3,0) Clifford Multivectors (8 dims)
    and fuses with existing Clifford Multivector Fields.
    """

    def __init__(self, d_model: int, device: str = 'cpu'):
        super().__init__()
        self.ca = CliffordAlgebra3D(device=device)
        self.d_model = d_model

        # Grade-wise Projection Heads
        self.proj_scalar = nn.Linear(d_model, 1)    # Grade 0: Energy / Density
        self.proj_vector = nn.Linear(d_model, 3)    # Grade 1: Attribute Directions (e1, e2, e3)
        self.proj_bivector = nn.Linear(d_model, 3)  # Grade 2: Rotor Planes (e12, e13, e23)
        self.proj_trivector = nn.Linear(d_model, 1) # Grade 3: Context Volume (e123)

        self.norm = nn.LayerNorm(d_model)
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def project_to_multivector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (Batch, Seq_Len, d_model)
        Output: Psi_trans (Batch, Seq_Len, 8)
        """
        x_norm = self.norm(x)
        g0 = F.softplus(self.proj_scalar(x_norm))
        g1 = self.proj_vector(x_norm)
        g2 = torch.tanh(self.proj_bivector(x_norm))
        g3 = self.proj_trivector(x_norm)

        # Concatenate 8 multivector blade components
        return torch.cat([g0, g1, g2, g3], dim=-1)

    def forward(self, x_transformer: torch.Tensor, psi_field: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        psi_trans = self.project_to_multivector(x_transformer)

        if psi_field is None:
            return {"psi_multivector": psi_trans, "phase_coherence": None}

        # Geometric Product Fusion
        psi_interaction = self.ca.geometric_product(psi_trans, psi_field)
        gate = torch.sigmoid(self.fusion_gate)
        psi_fused = (1.0 - gate) * psi_field + gate * psi_interaction

        phase_coherence = psi_interaction[..., 0]
        return {
            "psi_multivector": psi_fused,
            "psi_trans": psi_trans,
            "phase_coherence": phase_coherence
        }


class CliffordAttention(nn.Module):
    """
    Clifford Phase-Coherence Attention.
    Replaces Softmax Self-Attention with Geometric Product & Reversion-based Phase Coherence.
    """

    def __init__(self, num_heads: int, device: str = 'cpu'):
        super().__init__()
        self.ca = CliffordAlgebra3D(device=device)
        self.num_heads = num_heads

        self.proj_q = nn.Linear(8, 8, bias=False)
        self.proj_k = nn.Linear(8, 8, bias=False)
        self.proj_v = nn.Linear(8, 8, bias=False)
        self.proj_out = nn.Linear(8, 8, bias=False)

        self.beta = nn.Parameter(torch.tensor(5.0))   # Phase sensitivity
        self.gamma = nn.Parameter(torch.tensor(0.0))  # Phase threshold

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  x (Batch, Seq_Len, Num_Heads, 8)
        Output: (output_field, phase_attn_weights)
        """
        B, L, H, C = x.shape
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)

        Q_perm = Q.permute(0, 2, 1, 3)  # (B, H, L, 8)
        K_perm = K.permute(0, 2, 1, 3)  # (B, H, L, 8)

        # Grade-0 Scalar inner product of Q_i and K_j~: sum_a Q_i,a * K_j,a * sign_a
        raw_coherence = torch.einsum(
            'bhia, bhja, a -> bhij',
            Q_perm, K_perm, self.ca.reversion_signs.to(x.device)
        )

        norm_Q = self.ca.norm(Q_perm).unsqueeze(-1)    # (B, H, L, 1)
        norm_K = self.ca.norm(K_perm).unsqueeze(-2)    # (B, H, 1, L)

        normalized_coherence = raw_coherence / (norm_Q * norm_K + 1e-8)

        # Phase-Locking Gate Activation
        phase_attn_weights = torch.sigmoid(self.beta * normalized_coherence - self.gamma)

        if mask is not None:
            phase_attn_weights = phase_attn_weights.masked_fill(mask == 0, 0.0)

        V_perm = V.permute(0, 2, 1, 3)
        out_perm = torch.einsum('bhij, bhja -> bhia', phase_attn_weights, V_perm)
        out = out_perm.permute(0, 2, 1, 3)

        return self.proj_out(out), phase_attn_weights


class SymbolEmergenceEngine(nn.Module):
    """
    Spontaneous Symbol Emergence & Dynamic Rule Update Layer.
    Triggered when Phase Mismatch / Tension exceeds critical threshold.
    Sprouts new symbol tile t_new and expands compatibility rule matrix M.
    """

    def __init__(self, initial_num_tiles: int = 4, max_tiles: int = 64, threshold: float = 0.6, device: str = 'cpu'):
        super().__init__()
        self.ca = CliffordAlgebra3D(device=device)
        self.threshold = threshold
        self.num_tiles = initial_num_tiles
        self.max_tiles = max_tiles

        # Registered parameter buffer for tile rotors (max_tiles, 8) to preserve optimizer references
        self.tile_rotors = nn.Parameter(torch.randn(max_tiles, 8) * 0.1)

        # Compatibility matrix registered as buffer
        self.register_buffer("compatibility_matrix", torch.zeros(max_tiles, max_tiles))
        self._update_compatibility_matrix()

    def _update_compatibility_matrix(self):
        """Recomputes active sub-matrix M(i, j) = (<R_i~ R_j>_0)^2."""
        rotors = self.tile_rotors[:self.num_tiles]
        scalars = torch.einsum('ia, ja, a -> ij', rotors, rotors, self.ca.reversion_signs.to(rotors.device))
        norm_i = self.ca.norm(rotors).unsqueeze(1)
        norm_j = self.ca.norm(rotors).unsqueeze(0)
        coherence = scalars / (norm_i * norm_j + 1e-8)
        self.compatibility_matrix[:self.num_tiles, :self.num_tiles] = torch.pow(coherence, 2)

    def check_and_sprout_symbol(self, current_field: torch.Tensor) -> Dict[str, Any]:
        """
        Checks Emergence Criterion:
        Psi > Psi_threshold AND min_k E_rotor(x_i, k) > E_mismatch
        If triggered, sprouts a new symbol in place and updates compatibility matrix M.
        """
        field_flat = current_field.view(-1, 8)  # (N, 8)
        active_rotors = self.tile_rotors[:self.num_tiles]

        scalars = torch.einsum('na, ta, a -> nt', field_flat, active_rotors, self.ca.reversion_signs.to(current_field.device))
        norm_field = self.ca.norm(field_flat).unsqueeze(1)
        norm_tiles = self.ca.norm(active_rotors).unsqueeze(0)

        alignment = scalars / (norm_field * norm_tiles + 1e-8)
        max_alignment, _ = torch.max(alignment, dim=-1)

        mismatch_score = torch.mean(1.0 - max_alignment).item()

        sprouted = False
        new_tile_idx = None

        if mismatch_score > self.threshold and self.num_tiles < self.max_tiles:
            sprouted = True
            mean_state = torch.mean(field_flat, dim=0, keepdim=True)  # (1, 8)
            mean_norm = self.ca.norm(mean_state) + 1e-8
            normalized_new_rotor = mean_state / mean_norm

            # In-place parameter update preserving optimizer parameter reference
            self.tile_rotors.data[self.num_tiles:self.num_tiles + 1] = normalized_new_rotor
            new_tile_idx = self.num_tiles
            self.num_tiles += 1

            self._update_compatibility_matrix()

        return {
            "sprouted": sprouted,
            "mismatch_score": mismatch_score,
            "new_num_tiles": self.num_tiles,
            "new_tile_idx": new_tile_idx,
            "compatibility_matrix": self.compatibility_matrix[:self.num_tiles, :self.num_tiles]
        }


class MERAIsometryRG(nn.Module):
    """
    Holographic Coarse-Graining Layer (MERA Isometry) for Multivector Fields.
    Compresses micro 2x2 local spatial grid or sequence pairs into macro multivector states while preserving rotor flux.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.ca = CliffordAlgebra3D(device=device)
        self.isometry_weights = nn.Parameter(torch.randn(4, 8, 8) * 0.05)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Input:  field (Batch, Height, Width, 8) or (Batch, Seq_Len, 8)
        Output: macro_field (Batch, Height//2, Width//2, 8) or (Batch, Seq_Len//2, 8)
        """
        if field.dim() == 4:
            B, H, W, C = field.shape
            pad_h = H % 2
            pad_w = W % 2
            if pad_h > 0 or pad_w > 0:
                field = F.pad(field, (0, 0, 0, pad_w, 0, pad_h))
                H, W = field.shape[1], field.shape[2]

            patches = field.view(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, W // 2, 4, C)
            macro_field = torch.einsum('bhwpc, pck -> bhwk', patches, self.isometry_weights)
        elif field.dim() == 3:
            B, L, C = field.shape
            if L % 2 != 0:
                field = F.pad(field, (0, 0, 0, 1))
                L = field.shape[1]
            patches = field.view(B, L // 2, 2, C)
            sub_weights = self.isometry_weights[:2]
            macro_field = torch.einsum('blpc, pck -> blk', patches, sub_weights)
        else:
            raise ValueError(f"Unsupported field dimension: {field.dim()}")

        norm = self.ca.norm(macro_field).unsqueeze(-1)
        return macro_field / (norm + 1e-8)


class NonEquilibriumSpacetimeField(nn.Module):
    """
    Non-Equilibrium Spacetime Field Dynamics & Dual Action Engine.
    Simulates field evolution under phase diffusion, causal relaxation, and external drives:
    dPsi/dt = -gamma * grad^2(Psi) - eta * (delta V_causal / delta Psi~) + Omega_ext
    And Dual Action Engine:
    - Contraction Phase: RG coarse-graining of external inputs
    - Expansion Phase: Emission of multivector causal wave via outer product with source field J_source
    """

    def __init__(self, gamma: float = 0.1, eta: float = 0.05, device: str = 'cpu'):
        super().__init__()
        self.ca = CliffordAlgebra3D(device=device)
        self.gamma = gamma
        self.eta = eta
        self.source_j = nn.Parameter(torch.randn(1, 1, 8) * 0.1)

    def compute_causal_potential(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Calculates Causal Potential Energy V_causal(x) = 1/2 || (Grade1 ^ grad(Grade1)) . Grade3 ||^2.
        For discrete grid/sequence, approximates via bivector-trivector inner contraction.
        """
        g1 = psi[..., 1:4]  # 1-blade (vector)
        g2 = psi[..., 4:7]  # 2-blade (bivector)
        g3 = psi[..., 7:8]  # 3-blade (trivector)

        tension_plane = torch.sum(g1 * g2, dim=-1, keepdim=True)
        causal_potential = 0.5 * torch.pow(tension_plane * g3, 2)
        return torch.mean(causal_potential)

    def evolve_step(self, psi: torch.Tensor, omega_ext: Optional[torch.Tensor] = None, dt: float = 0.01) -> torch.Tensor:
        """Executes one time step dPsi/dt evolution."""
        if psi.dim() == 3:
            laplacian = torch.roll(psi, shifts=1, dims=1) + torch.roll(psi, shifts=-1, dims=1) - 2.0 * psi
        elif psi.dim() == 4:
            laplacian = (
                torch.roll(psi, shifts=1, dims=1) + torch.roll(psi, shifts=-1, dims=1) +
                torch.roll(psi, shifts=1, dims=2) + torch.roll(psi, shifts=-1, dims=2) - 4.0 * psi
            )
        else:
            laplacian = torch.zeros_like(psi)

        psi_rev = self.ca.revert(psi)
        relaxation = -self.eta * psi_rev

        ext_drive = omega_ext if omega_ext is not None else torch.zeros_like(psi)

        dpsi_dt = -self.gamma * laplacian + relaxation + ext_drive
        psi_next = psi + dpsi_dt * dt

        norm = self.ca.norm(psi_next).unsqueeze(-1)
        return psi_next / (norm + 1e-8)

    def emit_causal_wave(self, internal_psi: torch.Tensor) -> torch.Tensor:
        """
        Dual Action Engine: Expansion Phase.
        Emits causal wave via outer product with internal multivector state and source field J_source.
        Psi_emission = Psi_int x J_source
        """
        j_exp = self.source_j.expand_as(internal_psi)
        emission_wave = self.ca.geometric_product(internal_psi, j_exp)
        return emission_wave


class HolographicCognitiveEngine(nn.Module):
    """
    Unified Holographic Cognitive Engine.
    Combines Bridge, Attention, Symbol Emergence, MERA RG, and Non-Equilibrium Spacetime Field.
    """

    def __init__(self, d_model: int, num_heads: int = 4, grid_size: int = 16, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.bridge = TransformerToCliffordBridge(d_model=d_model, device=device)
        self.attention = CliffordAttention(num_heads=num_heads, device=device)
        self.symbol_emergence = SymbolEmergenceEngine(device=device)
        self.mera_rg = MERAIsometryRG(device=device)
        self.field_dynamics = NonEquilibriumSpacetimeField(device=device)

    def forward(self, x_transformer: torch.Tensor, psi_field: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Full Pipeline Forward Pass:
        1. Vector Embedding -> Clifford Multivector Conversion & Fusion
        2. Clifford Phase-Coherence Attention
        3. Check Emergence Criterion & Sprout Symbols if needed
        4. Holographic MERA-RG Coarse-Graining Flow
        5. Non-Equilibrium Spacetime Field Step & Dual Action Emission
        """
        # 1. Bridge Conversion & Fusion
        bridge_res = self.bridge(x_transformer, psi_field)
        psi_multivector = bridge_res["psi_multivector"]  # (B, L, 8)

        # Reshape for Multi-Head Attention: (B, L, Num_Heads, 8)
        B, L, C = psi_multivector.shape
        psi_heads = psi_multivector.unsqueeze(2).expand(B, L, self.attention.num_heads, C)

        # 2. Clifford Phase-Coherence Attention
        attn_out, attn_weights = self.attention(psi_heads)
        psi_attended = torch.mean(attn_out, dim=2)  # Aggregate heads back to (B, L, 8)

        # 3. Symbol Emergence Check
        emergence_res = self.symbol_emergence.check_and_sprout_symbol(psi_attended)

        # 4. MERA Hierarchical RG Coarse-Graining Flow
        macro_field = self.mera_rg(psi_attended)

        # 5. Spacetime Field Dynamics Step & Dual Action Emission
        psi_evolved = self.field_dynamics.evolve_step(psi_attended)
        emitted_wave = self.field_dynamics.emit_causal_wave(psi_evolved)

        causal_potential = self.field_dynamics.compute_causal_potential(psi_evolved)

        return {
            "psi_multivector": psi_multivector,
            "psi_attended": psi_attended,
            "attn_weights": attn_weights,
            "emergence": emergence_res,
            "macro_field": macro_field,
            "psi_evolved": psi_evolved,
            "emitted_wave": emitted_wave,
            "causal_potential": causal_potential.item()
        }
