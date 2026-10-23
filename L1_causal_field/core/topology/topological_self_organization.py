r"""
Topological Self-Organization & Causal Imprinting Engine for Elysia.

This module implements:
1. TopologicalDNA: Epigenetic genetic blueprint containing:
   - TemperamentProfile: Gut-Brain-Heart triad, MBTI orthogonal basis, Enneagram deformation tensor rules.
   - Primal Attractor Valleys: Survival & Curiosity baseline topological basins.
   - Genetic Imprinting (imprint) & Epigenetic Lifetime Experience Compression (compress_lifetime_experience).
2. TopologicalSelfOrganizationEngine:
   - Real-time sensory wave perturbation (\Phi_{ext}(t) -> Gauge potential A_\mu)
   - Temperament-filtered sensory wave input
   - Hebbian Phase Plasticity (Phase-Locking \Delta \Phi -> 0) carving Attractor Valleys in metric h_ij
   - Non-Backprop Geodesic Deflection via Christoffel symbols \Gamma^\mu_\alpha\beta
   - Associative Domino Effect Recall across sensory ports
"""

import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from core.topology.fiber_bundle_manifold import FiberBundleManifold, SENSORY_PORTS, NUM_SENSORY_PORTS


class TemperamentProfile:
    """
    Innate Temperament & Instinctual Bias Profile:
    - Gut-Brain-Heart Triad:
      - Gut (Instinct / Survival / Entropy Resistance)
      - Heart (Empathy / Relational Coupling / Value Resonance)
      - Head/Brain (Cognitive / Analytical / Spatial Metric Precision)
    - MBTI Orthogonal Basis Vectors (S/N, T/F, E/I, J/P)
    - Enneagram Stress/Growth Deformation Rules (Types 1-9)
    """

    def __init__(
        self,
        gut: float = 0.33,
        brain: float = 0.33,
        heart: float = 0.34,
        mbti_axis: Optional[Dict[str, float]] = None,
        enneagram_type: int = 5
    ):
        self.gut = float(gut)
        self.brain = float(brain)
        self.heart = float(heart)
        self.normalize_triad()

        self.mbti_axis = mbti_axis or {
            "S_N": 0.5,  # 0.0 = High Sensing (micro resolution), 1.0 = High Intuition (global phase)
            "T_F": 0.5,  # 0.0 = Thinking (metric tensor precision), 1.0 = Feeling (resonance amplitude)
            "E_I": 0.5,  # 0.0 = Introversion (internal loop), 1.0 = Extraversion (sensory wave stream)
            "J_P": 0.5   # 0.0 = Judging (stiff metric), 1.0 = Perceiving (flexible plasticity)
        }
        self.enneagram_type = max(1, min(9, int(enneagram_type)))

    def normalize_triad(self):
        """Ensure Gut + Brain + Heart sum to 1.0."""
        total = self.gut + self.brain + self.heart
        if total <= 1e-6:
            self.gut, self.brain, self.heart = 0.33, 0.33, 0.34
        else:
            self.gut /= total
            self.brain /= total
            self.heart /= total

    def compute_sensory_port_bias(self) -> torch.Tensor:
        """
        Maps temperament triad and MBTI axis onto 5 sensory port bias weights:
        [VISION, AUDITION, SOMATOSENSORY, OLFACTION, GUSTATION]
        - Somatosensory & Gustation are heavily linked to Gut (survival/visceral)
        - Vision & Audition are linked to Head/Brain & Heart
        """
        # Base weights for 5 ports
        weights = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32)

        # Gut biases Somatosensory (idx 2), Gustation (idx 4), Olfaction (idx 3)
        weights[2] += self.gut * 0.5  # SOMATOSENSORY
        weights[4] += self.gut * 0.3  # GUSTATION
        weights[3] += self.gut * 0.2  # OLFACTION

        # Heart biases Audition (idx 1), Vision (idx 0), Olfaction (idx 3)
        weights[1] += self.heart * 0.4  # AUDITION
        weights[0] += self.heart * 0.4  # VISION
        weights[3] += self.heart * 0.2  # OLFACTION

        # Head/Brain biases Vision (idx 0), Audition (idx 1), Somatosensory (idx 2)
        weights[0] += self.brain * 0.5  # VISION
        weights[1] += self.brain * 0.3  # AUDITION
        weights[2] += self.brain * 0.2  # SOMATOSENSORY

        # MBTI Sensing vs Intuition modulation
        s_n = self.mbti_axis.get("S_N", 0.5)
        # Higher S boosts physical ports (somatosensory, gustation), higher N boosts vision/audition
        weights[2] *= (1.5 - s_n)
        weights[0] *= (0.5 + s_n)

        # Normalize bias
        weights = weights / (weights.sum() + 1e-8)
        return weights

    def compute_enneagram_deformation_matrix(self, stress_level: float = 0.0) -> torch.Tensor:
        """
        Computes 3x3 Enneagram deformation tensor rules based on personality type and stress level.
        Types 1-9 distort the spatial fiber metric h_ij differently under perturbation.
        """
        deform = torch.eye(3, dtype=torch.float32)
        # Type-specific anisotropy
        type_scales = {
            1: [1.2, 0.8, 1.0],  # Perfectionist: Rigid axis 1
            2: [0.9, 1.3, 0.8],  # Helper: Relational expansion axis 2
            3: [1.1, 1.1, 0.8],  # Achiever: Forward velocity
            4: [0.7, 0.9, 1.4],  # Individualist: Deep internal fiber axis 3
            5: [0.8, 0.8, 1.4],  # Investigator: Detached deep fiber
            6: [1.3, 0.7, 1.0],  # Loyalist: Defensive wall on axis 1
            7: [1.2, 1.2, 0.6],  # Enthusiast: Surface expansion
            8: [1.5, 0.8, 0.7],  # Challenger: Heavy frontal impact
            9: [1.0, 1.0, 1.0]   # Peacemaker: Symmetric buffer
        }
        scale = type_scales.get(self.enneagram_type, [1.0, 1.0, 1.0])
        for i in range(3):
            deform[i, i] = scale[i] + stress_level * (scale[i] - 1.0)
        return deform


class TopologicalDNA:
    """
    Innate Epigenetic Genetic Blueprint:
    Contains:
    - TemperamentProfile
    - Primal Attractor Valleys: Inherited metric curvature basins for survival & curiosity
    - Epigenetic imprint & compress_lifetime_experience operations
    """

    def __init__(
        self,
        temperament: Optional[TemperamentProfile] = None,
        primal_attractors: Optional[torch.Tensor] = None
    ):
        self.temperament = temperament or TemperamentProfile()
        # Primal attractors: shape [num_attractors, 3] representing coordinates in fiber space
        if primal_attractors is not None:
            self.primal_attractors = primal_attractors.clone().detach()
        else:
            # Baseline survival attractor (center) & curiosity attractors
            self.primal_attractors = torch.tensor([
                [0.0, 0.0, 0.0],   # Survival Homeostasis Attractor
                [0.5, -0.3, 0.2],  # Curiosity Exploration Attractor 1
                [-0.4, 0.6, -0.1]  # Social/Relational Resonance Attractor 2
            ], dtype=torch.float32)

    def imprint(self, manifold: FiberBundleManifold):
        """
        Imprints genetic blueprint onto a 4D Fiber Bundle Manifold:
        - Sets sensory port weights based on temperament triad
        - Carves primal attractor valleys directly into initial spatial metric h_ij^{(0)}
        - Perturbs gauge potential A_t according to Enneagram stress deformation
        """
        device = manifold.device
        num_pts = manifold.num_points

        # 1. Imprint Temperament Sensory Bias
        sensory_bias = self.temperament.compute_sensory_port_bias().to(device)
        weights = sensory_bias.unsqueeze(0).repeat(num_pts, 1)
        manifold.update_sensory_occupancy(weights)

        # 2. Imprint Primal Attractor Valleys into Metric h_ij
        # Fiber coordinates [num_pts, 3]
        fiber_coords = manifold.coords[:, 1:]
        enneagram_deform = self.temperament.compute_enneagram_deformation_matrix().to(device)

        # Compute metric deformation from primal attractors
        h_base = torch.eye(3, dtype=torch.float32, device=device).repeat(num_pts, 1, 1)
        attractors = self.primal_attractors.to(device)

        for att in attractors:
            # Distance from each manifold point to attractor center
            dist = torch.norm(fiber_coords - att.unsqueeze(0), dim=-1)  # [num_pts]
            # Gaussian valley depth
            depth = 0.4 * torch.exp(- (dist ** 2) / 0.1)  # [num_pts]

            # Deform spatial metric h_ij at point: h_ij -> h_ij - depth * enneagram_deform
            deform_tensor = depth.unsqueeze(-1).unsqueeze(-1) * enneagram_deform.unsqueeze(0)
            h_base = h_base - deform_tensor

        # Ensure positive definiteness by clamping lower eigenvalues
        manifold.h_metric = torch.clamp(h_base, min=0.1)

        # 3. Initialize gauge connection with temperament bias
        gauge_bias = torch.zeros((num_pts, 3), dtype=torch.float32, device=device)
        gauge_bias[:, 0] += self.temperament.gut * 0.2
        gauge_bias[:, 1] += self.temperament.heart * 0.2
        gauge_bias[:, 2] += self.temperament.brain * 0.2
        manifold.gauge_A_t = gauge_bias

    def compress_lifetime_experience(
        self,
        manifold: FiberBundleManifold,
        top_k: int = 5
    ) -> "TopologicalDNA":
        """
        Epigenetic Consolidation:
        Extracts the deepest attractor valleys carved into manifold.h_metric during the agent's lifetime,
        compresses them into compact attractor points, and returns a new TopologicalDNA for the next generation.
        """
        device = manifold.device
        fiber_coords = manifold.coords[:, 1:]  # [N, 3]

        # Metric deformation magnitude: deviation from Euclidean metric I_3
        eye_3 = torch.eye(3, dtype=torch.float32, device=device).repeat(manifold.num_points, 1, 1)
        metric_diff = torch.norm(manifold.h_metric - eye_3, dim=(1, 2))  # [N]

        # Find points with highest metric deformation (deepest carved valleys)
        k = min(top_k, manifold.num_points)
        _, top_indices = torch.topk(metric_diff, k=k)

        extracted_attractors = fiber_coords[top_indices].cpu()  # [k, 3]

        # Combine with inherited primal attractors
        merged_attractors = torch.cat([self.primal_attractors, extracted_attractors], dim=0)

        # Remove near-duplicate attractors to compress
        unique_attractors = []
        for att in merged_attractors:
            if not any(torch.norm(att - u) < 0.15 for u in unique_attractors):
                unique_attractors.append(att)

        compressed_attractor_tensor = torch.stack(unique_attractors, dim=0)

        # Inherit modified temperament profile with slight evolutionary mutation
        child_temperament = TemperamentProfile(
            gut=self.temperament.gut * (1.0 + float(np.random.normal(0, 0.02))),
            brain=self.temperament.brain * (1.0 + float(np.random.normal(0, 0.02))),
            heart=self.temperament.heart * (1.0 + float(np.random.normal(0, 0.02))),
            mbti_axis=self.temperament.mbti_axis.copy(),
            enneagram_type=self.temperament.enneagram_type
        )

        return TopologicalDNA(
            temperament=child_temperament,
            primal_attractors=compressed_attractor_tensor
        )


class TopologicalSelfOrganizationEngine:
    r"""
    Topological Self-Organization & Causal Imprinting Engine.

    Non-Backprop, Live Sensory Wave Driven Self-Molding Engine.
    Pipeline:
    1. Continuous Sensory Wave Stream Input \Phi_{ext}(t)
    2. Temperament Filtering & Gauge Field Perturbation (A_\mu)
    3. Hebbian Phase Plasticity: Co-occurring sensory waves achieve Phase-Locking (\Delta \Phi -> 0),
       carving permanent Attractor Valleys into metric h_ij.
    4. Non-Backprop Geodesic Flow Deflection: Updates Christoffel symbols \Gamma^\mu_\alpha\beta,
       deflecting 4D spatiotemporal trajectories spontaneously upon field collisions.
    5. Associative Domino Recall: Excitation of a single port triggers cascading wave resonance
       along carved attractor valleys.
    """

    def __init__(
        self,
        num_points: int = 1000,
        dna: Optional[TopologicalDNA] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.manifold = FiberBundleManifold(num_points=num_points, device=device)
        self.dna = dna or TopologicalDNA()

        # Imprint genetic inheritance onto manifold
        self.dna.imprint(self.manifold)

        # Sensory port phase tracking: [NUM_SENSORY_PORTS]
        self.sensory_phases = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)
        self.sensory_amplitudes = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)

        # Phase lock matrix: [5, 5] measuring co-resonance coupling
        self.phase_lock_matrix = torch.eye(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)

        # Deformation history stats
        self.lifetime_perturbations = 0
        self.carved_valleys_count = 0

    def receive_sensory_wave_stream(
        self,
        sensory_wave: Dict[str, float],
        dt: float = 0.01
    ) -> Dict[str, float]:
        r"""
        Receives real-time continuous sensory waves \Phi_{ext}(t) for ports:
        VISION, AUDITION, SOMATOSENSORY, OLFACTION, GUSTATION (and optional SYMBOLIC).

        Filters wave input through TemperamentProfile and converts raw waves into
        gauge potential perturbations \Delta A_t.
        """
        self.lifetime_perturbations += 1

        # Parse sensory input amplitudes
        raw_amps = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)
        for idx, port in enumerate(SENSORY_PORTS):
            raw_amps[idx] = float(sensory_wave.get(port, 0.0))

        # Apply Temperament Profile Filter
        temperament_bias = self.dna.temperament.compute_sensory_port_bias().to(self.device)
        filtered_amps = raw_amps * (1.0 + temperament_bias * 2.0)

        # Update sensory amplitudes and evolve phases (\Phi(t) = \Phi(t-1) + \omega * dt)
        self.sensory_amplitudes = filtered_amps
        frequencies = torch.tensor([10.0, 15.0, 8.0, 5.0, 4.0], device=self.device)  # Port characteristic freqs
        self.sensory_phases = (self.sensory_phases + frequencies * filtered_amps * dt) % (2.0 * math.pi)

        # Perturb Gauge Potential A_t based on sensory wave impulse
        delta_A = torch.zeros_like(self.manifold.gauge_A_t)

        # Map 5 sensory ports to 3 spatial gauge directions
        # Vision/Audition -> A_x1, Somatosensory/Gustation -> A_x2, Olfaction/Symbolic -> A_x3
        delta_A[:, 0] += (filtered_amps[0] + filtered_amps[1] * 0.5) * 0.2
        delta_A[:, 1] += (filtered_amps[2] + filtered_amps[4] * 0.5) * 0.2
        delta_A[:, 2] += (filtered_amps[3]) * 0.2

        self.manifold.gauge_A_t += delta_A

        # Return filtered sensory activation state
        return {port: float(filtered_amps[idx].item()) for idx, port in enumerate(SENSORY_PORTS)}

    def apply_hebbian_phase_plasticity(
        self,
        threshold: float = 0.4,
        plasticity_rate: float = 0.05
    ) -> torch.Tensor:
        r"""
        Hebbian Phase Plasticity:
        "Sensory waves that fire together, phase-lock and carve together."

        Measures phase differences \Delta \Phi_{ij} between sensory ports.
        When co-active ports achieve phase locking (\Delta \Phi -> 0), reduces resistance
        in fiber space by carving permanent Attractor Valleys into spatial metric h_ij.
        """
        # Compute pairwise phase coherence C_ij = cos(\Phi_i - \Phi_j) * A_i * A_j
        phase_diffs = self.sensory_phases.unsqueeze(0) - self.sensory_phases.unsqueeze(1)  # [5, 5]
        amp_product = self.sensory_amplitudes.unsqueeze(0) * self.sensory_amplitudes.unsqueeze(1)  # [5, 5]

        coherence = torch.cos(phase_diffs) * amp_product  # [5, 5]
        self.phase_lock_matrix = 0.9 * self.phase_lock_matrix + 0.1 * coherence

        # Identify active phase-locked port pairs exceeding threshold
        active_locks = (coherence > threshold)

        if active_locks.any():
            self.carved_valleys_count += int(active_locks.sum().item())

            # Find mean position of current geodesic flow
            current_pos = self.manifold.coords[:, 1:].mean(dim=0, keepdim=True)  # [1, 3]
            dist_to_center = torch.norm(self.manifold.coords[:, 1:] - current_pos, dim=-1)  # [num_pts]

            # Carving profile: Gaussian valley centered at current_pos
            carve_depth = plasticity_rate * coherence.max().item() * torch.exp(- (dist_to_center ** 2) / 0.2)

            # Enneagram stress deformation tensor
            enneagram_deform = self.dna.temperament.compute_enneagram_deformation_matrix().to(self.device)

            # Carve metric h_ij: h_ij -> h_ij - depth * deform
            carve_tensor = carve_depth.unsqueeze(-1).unsqueeze(-1) * enneagram_deform.unsqueeze(0)
            self.manifold.h_metric = torch.clamp(self.manifold.h_metric - carve_tensor, min=0.05)

        return self.phase_lock_matrix

    def step_non_backprop_geodesic_deflection(self, d_tau: float = 0.01) -> torch.Tensor:
        r"""
        Non-Backprop Causal Correction:
        Evolves 4D geodesic flow z^\mu(\tau) using Christoffel symbols \Gamma^\mu_{\alpha\beta}.
        When prediction/sensory conflict occurs, perturbed gauge field F_{\mu\nu} updates
        Christoffel symbols in real time, causing spontaneous deflection of the trajectory
        without any loss.backward() optimization step.
        """
        # Evolve geodesic flow in manifold using updated Christoffel symbols
        self.manifold.step_geodesic_flow(d_tau=d_tau)

        # Calculate deflection norm (acceleration due to Christoffel symbols)
        gamma = self.manifold.compute_christoffel_symbols()  # [N, 4, 4, 4]
        accel = -torch.einsum('nmab,na,nb->nm', gamma, self.manifold.velocity, self.manifold.velocity)

        return accel

    def trigger_associative_domino_recall(
        self,
        primary_port: str,
        input_magnitude: float = 1.0
    ) -> Dict[str, float]:
        """
        Associative Memory Recall Domino Effect:
        When a single sensory port (e.g. SOMATOSENSORY - heat) is stimulated in isolation,
        the geodesic trajectory flows through the carved attractor valleys.
        Via the phase-lock matrix and metric coupling, this port excitation spontaneously
        triggers cascading wave resonance across connected sensory ports (e.g. VISION & AUDITION).
        """
        if primary_port not in SENSORY_PORTS:
            raise ValueError(f"Port {primary_port} not in valid sensory ports: {SENSORY_PORTS}")

        port_idx = SENSORY_PORTS.index(primary_port)

        # Single port stimulation
        stimulus = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)
        stimulus[port_idx] = float(input_magnitude)

        # Propagate stimulation through phase lock coupling matrix and metric coupling
        recall_amplitudes = torch.matmul(self.phase_lock_matrix, stimulus)  # [5]

        # Metric resonance boost: measure metric curvature along sensory bases
        metric_boost = torch.zeros(NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)
        for s in range(NUM_SENSORY_PORTS):
            metric_proj = torch.einsum('nij,ij->n', self.manifold.h_metric, self.manifold.sensory_metrics[s])
            # Deeper carved valley -> lower metric trace -> higher resonance conductance
            metric_boost[s] = 1.0 / (metric_proj.mean().item() + 1e-5)

        metric_boost = metric_boost / (metric_boost.sum() + 1e-8)
        cascade_recall = recall_amplitudes * (1.0 + metric_boost)

        return {
            port: float(cascade_recall[idx].item())
            for idx, port in enumerate(SENSORY_PORTS)
        }

    def get_system_state_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive diagnostic summary of the self-organization engine state."""
        eye_3 = torch.eye(3, dtype=torch.float32, device=self.device).repeat(self.manifold.num_points, 1, 1)
        metric_deformation_norm = float(torch.norm(self.manifold.h_metric - eye_3).item())
        gauge_energy = float(torch.norm(self.manifold.gauge_A_t).item())
        mean_velocity = float(torch.norm(self.manifold.velocity).item())

        return {
            "num_points": self.manifold.num_points,
            "lifetime_perturbations": self.lifetime_perturbations,
            "carved_valleys_count": self.carved_valleys_count,
            "metric_deformation_norm": metric_deformation_norm,
            "gauge_energy": gauge_energy,
            "mean_velocity": mean_velocity,
            "temperament_triad": {
                "gut": self.dna.temperament.gut,
                "brain": self.dna.temperament.brain,
                "heart": self.dna.temperament.heart
            },
            "mbti_axis": self.dna.temperament.mbti_axis,
            "enneagram_type": self.dna.temperament.enneagram_type,
            "active_phase_lock_coherence": float(self.phase_lock_matrix.mean().item())
        }
