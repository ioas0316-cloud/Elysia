"""
elysia_core/horizon_expansion_spectrometer.py

Horizon Expansion Relational Spectrometer (지평 확장 관계적 분광기)
=====================================================================
Solving the Fixed Container Trap (fixed N-dimensional box) via Scale-Free Manifold Expansion.
1. Relational Spectrometer (관계적 분광 매니폴드):
   Measures phase discrepancy (Phase Torque) and differential tension (Differential Mapping)
   between incoming stimuli and existing cognitive coordinate axes.
2. Generative Axis Construction (새로운 인과적 축 작도):
   When a completely novel stimulus (e.g. 'bitter taste' / 쓴맛) enters, system measures phase
   discrepancy against known axes (sweet, sour, salty). If residual tension exceeds threshold,
   it constructs a new rotational axis (N -> N+1 expansion) rather than forcing it into a pre-baked box.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class HorizonExpansionSpectrometer:
    """
    Dynamic Relational Spectrometer capable of expanding cognitive dimensions (N -> N+1)
    and constructing new coordinate axes upon encountering novel sensory/cognitive friction.
    """

    def __init__(
        self,
        initial_dimension: int = 3,
        expansion_threshold: float = 0.35,
        initial_axis_labels: Optional[List[str]] = None
    ):
        self.dimension = initial_dimension
        self.expansion_threshold = expansion_threshold

        # Initialize base coordinate axes
        labels = initial_axis_labels or ["sweet", "sour", "salty"]
        self.axis_labels: List[str] = list(labels[:initial_dimension])

        # If initial labels fewer than initial_dimension, pad default labels
        while len(self.axis_labels) < initial_dimension:
            self.axis_labels.append(f"axis_{len(self.axis_labels)+1}")

        # Orthonormal basis matrix representing current cognitive axes in R^N
        self.basis_matrix = np.eye(initial_dimension, dtype=np.float64)

        # Relational Spectrum facet registry mapping axis_label -> metadata
        self.facet_registry: Dict[str, Dict[str, Any]] = {}
        for idx, label in enumerate(self.axis_labels):
            self.facet_registry[label] = {
                "axis_index": idx,
                "label": label,
                "creation_step": 0,
                "basis_vector": self.basis_matrix[idx].copy(),
                "accumulated_resonance": 1.0
            }

        self.expansion_events: List[Dict[str, Any]] = []
        self.step_counter = 0

    def analyze_and_expand_spectrum(
        self,
        stimulus_vector: np.ndarray,
        stimulus_label_hint: str = "unknown_stimulus"
    ) -> Dict[str, Any]:
        """
        Analyzes an incoming stimulus vector against current cognitive basis axes in full feature space.
        If residual differential tension exceeds threshold, constructs a new axis (N -> N+1).
        """
        self.step_counter += 1

        # Full feature dimension M
        raw_vec = np.array(stimulus_vector, dtype=np.float64)
        M = max(len(raw_vec), self.dimension)

        full_stimulus = np.pad(raw_vec, (0, max(0, M - len(raw_vec))))[:M]
        norm_stim = np.linalg.norm(full_stimulus)
        if norm_stim > 1e-8:
            unit_stim = full_stimulus / norm_stim
        else:
            unit_stim = np.zeros(M, dtype=np.float64)

        # Embed current basis matrix into R^M
        full_basis = np.zeros((self.dimension, M), dtype=np.float64)
        full_basis[:, :self.dimension] = self.basis_matrix

        # 1. Project stimulus onto current N basis axes
        projections = np.dot(full_basis, unit_stim)  # Shape: (N,)
        reconstructed = np.dot(projections, full_basis)  # Shape: (M,)

        # 2. Compute residual vector & differential tension in R^M
        residual_vector = unit_stim - reconstructed
        differential_tension = float(np.linalg.norm(residual_vector))

        # Compute angular phase torque discrepancy
        phase_torque = np.cross(unit_stim[:3], reconstructed[:3]) if M >= 3 else residual_vector[:3]

        expanded = False
        new_axis_info = None

        # 3. Check if differential tension exceeds expansion threshold
        if differential_tension > self.expansion_threshold:
            expanded = True
            new_axis_info = self._construct_new_coordinate_axis(
                residual_vector=residual_vector,
                stimulus_label_hint=stimulus_label_hint,
                differential_tension=differential_tension
            )

            # Re-evaluate projections after horizon expansion to N+1
            expanded_basis = np.zeros((self.dimension, max(M, self.dimension)), dtype=np.float64)
            expanded_basis[:, :self.dimension] = self.basis_matrix
            padded_unit = np.pad(unit_stim, (0, max(0, self.dimension - M)))
            projections = np.dot(expanded_basis, padded_unit)

        # Update accumulated resonance on matching facet
        matched_axis_idx = int(np.argmax(np.abs(projections)))
        matched_label = self.axis_labels[matched_axis_idx]
        self.facet_registry[matched_label]["accumulated_resonance"] += 0.1

        return {
            "step": self.step_counter,
            "dimension": self.dimension,
            "projections": projections,
            "differential_tension": differential_tension,
            "phase_torque": phase_torque,
            "phase_torque_norm": float(np.linalg.norm(phase_torque)),
            "horizon_expanded": expanded,
            "expansion_info": new_axis_info,
            "matched_axis_label": matched_label,
            "matched_resonance": float(projections[matched_axis_idx]),
            "active_axis_labels": list(self.axis_labels)
        }

    def _construct_new_coordinate_axis(
        self,
        residual_vector: np.ndarray,
        stimulus_label_hint: str,
        differential_tension: float
    ) -> Dict[str, Any]:
        """
        Constructs a brand new coordinate axis in N+1 dimensions orthogonal to existing axes.
        """
        old_dim = self.dimension
        new_dim = old_dim + 1
        self.dimension = new_dim

        # Expand existing basis matrix to (new_dim, new_dim)
        new_basis = np.zeros((new_dim, new_dim), dtype=np.float64)
        new_basis[:old_dim, :old_dim] = self.basis_matrix

        # Normalize residual vector
        res_norm = np.linalg.norm(residual_vector)
        if res_norm > 1e-8:
            norm_res = residual_vector / res_norm
        else:
            norm_res = np.zeros(len(residual_vector), dtype=np.float64)

        # Construct new axis vector in R^new_dim
        new_axis_vec = np.zeros(new_dim, dtype=np.float64)
        overlap = min(len(norm_res), new_dim)
        new_axis_vec[:overlap] = norm_res[:overlap]

        # Orthonormalize against prior basis vectors using Gram-Schmidt
        for k in range(old_dim):
            proj_k = np.dot(new_axis_vec, new_basis[k])
            new_axis_vec -= proj_k * new_basis[k]

        norm_new = np.linalg.norm(new_axis_vec)
        if norm_new > 1e-8:
            new_axis_vec /= norm_new
        else:
            new_axis_vec[new_dim - 1] = 1.0

        new_basis[old_dim] = new_axis_vec
        self.basis_matrix = new_basis

        # Register label and facet
        label = stimulus_label_hint if stimulus_label_hint != "unknown_stimulus" else f"axis_{new_dim}"
        if label in self.axis_labels:
            label = f"{label}_extended_{new_dim}"

        self.axis_labels.append(label)
        self.facet_registry[label] = {
            "axis_index": old_dim,
            "label": label,
            "creation_step": self.step_counter,
            "basis_vector": new_axis_vec.copy(),
            "accumulated_resonance": 1.0
        }

        # Update all prior basis vectors in registry to match new_dim size
        for k, facet in self.facet_registry.items():
            facet["basis_vector"] = np.pad(facet["basis_vector"], (0, max(0, new_dim - len(facet["basis_vector"]))))

        event_record = {
            "step": self.step_counter,
            "previous_dimension": old_dim,
            "new_dimension": new_dim,
            "created_axis_label": label,
            "differential_tension": differential_tension,
            "new_axis_vector": new_axis_vec
        }
        self.expansion_events.append(event_record)
        return event_record
