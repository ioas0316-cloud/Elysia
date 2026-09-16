r"""
Recursive Causal Evolution Loop (재귀적 메타 인과 진화 루프)
============================================================

Implements the fractal 3-tier recursive causal evolution pipeline:
1. Component Level (\Delta_c): Low-level raw byte and sensory impulse tension scan & zero-convergence equilibrium.
2. Principle Level (\Delta_p): Judgment rule tensor (\Theta) evaluation and meta-principle evolution when persistent friction occurs.
3. Structural Level (\Delta_s): Macro topology, SemanticMassEngine, and GenealogicalMemoryUnit deconstruction & re-weaving.

Top-down zero-convergence feedback dynamics (\Delta -> 0) ensure that macro structural equilibrium
propagates back down to stabilize principle and component level tensions.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.ingestion.raw_byte_causal_sensor import (
    RawByteCausalSensor,
    ZeroConvergenceTension,
    ByteStructuralGrounding,
    ByteProvenanceBranch,
    TransformationProvenanceLog,
    StructuralViolationError,
)
from core.physics.semantic_mass_engine import SemanticMassEngine
from core.meta.algorithm_tensor import AlgorithmTensor
from core.memory.genealogical_memory_unit import (
    GenealogicalMemoryUnit,
    DynamicDeconstructionEngine,
)


@dataclass
class MetaCausalPrinciple:
    r"""
    [Meta-Causal Principle (메타 인과 판별 원리)]
    Encodes the judgment principle/rules as a dynamic tensor parameter structure (\Theta)
    and AlgorithmTensor graph rather than a static if-else code block.

    Can evolve when persistent principle tension (\Delta_p) occurs.
    """
    principle_id: str
    name: str
    param_vector: np.ndarray  # Shape (D,) parameter vector Theta
    algorithm_tensor: AlgorithmTensor
    evolution_version: int = 1
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def evaluate_principle_tension(
        self,
        component_delta: float,
        feature_vector: np.ndarray,
    ) -> float:
        r"""
        Evaluates principle tension (\Delta_p) by measuring the resonance friction
        between input feature vector and principle parameter vector Theta.
        """
        if len(feature_vector) != len(self.param_vector):
            # Pad or truncate feature vector to match param_vector
            dim = len(self.param_vector)
            if len(feature_vector) < dim:
                feature_vector = np.pad(feature_vector, (0, dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:dim]

        # Calculate directional alignment (cosine similarity)
        p_norm = self.param_vector / (np.linalg.norm(self.param_vector) + 1e-9)
        f_norm = feature_vector / (np.linalg.norm(feature_vector) + 1e-9)
        alignment = float(np.dot(p_norm, f_norm))

        # Dissonance friction between principle parameters and input features
        dissonance = 0.5 * (1.0 - alignment)

        # Principle tension is a function of component delta and internal principle dissonance
        principle_delta = float(0.5 * component_delta + 0.5 * (dissonance * 2.0))
        return float(np.clip(principle_delta, 0.0, 1.0))

    def evolve_principle(
        self,
        trigger_delta_p: float,
        feature_vector: np.ndarray,
        reason: str = "Persistent Meta-Tension Resolution",
    ):
        r"""
        Evolves principle parameters Theta towards the new feature manifold,
        reducing principle friction (\Delta_p -> 0) and recording the evolution lineage.
        """
        dim = len(self.param_vector)
        feat = feature_vector[:dim] if len(feature_vector) >= dim else np.pad(feature_vector, (0, dim - len(feature_vector)))

        old_param = self.param_vector.copy()
        # Learning / Adaptation step towards input manifold
        learning_rate = 0.3 * trigger_delta_p
        self.param_vector = (1.0 - learning_rate) * self.param_vector + learning_rate * feat
        self.param_vector = self.param_vector / (np.linalg.norm(self.param_vector) + 1e-9)

        self.evolution_version += 1
        record = {
            "version": self.evolution_version,
            "trigger_delta_p": trigger_delta_p,
            "reason": reason,
            "param_shift_norm": float(np.linalg.norm(self.param_vector - old_param)),
            "timestamp": time.time(),
        }
        self.evolution_history.append(record)

        # Update AlgorithmTensor graph
        node_id = f"eval_node_v{self.evolution_version}"
        self.algorithm_tensor.add_node(node_id, op_type="evolved_transform", param_vector=self.param_vector)
        prev_node_id = f"eval_node_v{self.evolution_version - 1}"
        if prev_node_id in self.algorithm_tensor.nodes:
            self.algorithm_tensor.add_connection(prev_node_id, node_id, weight=1.0)


@dataclass
class FractalTensionReport:
    r"""
    Comprehensive report of the fractal 3-tier tension state:
    - delta_c: Component Level Delta (\Delta_c)
    - delta_p: Principle Level Delta (\Delta_p)
    - delta_s: Structural Level Delta (\Delta_s)
    - total_fractal_tension: Integrated scalar tension (\Delta_{total})
    - is_zero_converged: True if total tension < 1e-3
    - principle_evolved: True if principle evolution occurred
    - memory_deconstructed: True if structural deconstruction occurred
    """
    delta_c: float
    delta_p: float
    delta_s: float
    total_fractal_tension: float
    is_zero_converged: bool
    principle_evolved: bool
    memory_deconstructed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RecursiveCausalEvolutionLoop:
    r"""
    [Recursive Causal Evolution Loop]
    Integrates Component, Principle, and Structural tiers in a continuous fractal loop:
    1. Component Tier: Raw byte / sensory scan via ByteStructuralGrounding.
    2. Principle Tier: Evaluates MetaCausalPrinciple (\Theta). Evolves principle if \Delta_p > principle_threshold.
    3. Structural Tier: Integrates with SemanticMassEngine & DynamicDeconstructionEngine. Deconstructs/re-weaves if \Delta_s > structural_threshold.
    4. Top-Down Feedback: Macro structural equilibrium reduces lower-level residual tension (\Delta -> 0).
    """

    def __init__(
        self,
        dimensions: int = 16,
        component_threshold: float = 0.3,
        principle_threshold: float = 0.4,
        structural_threshold: float = 0.65,
    ):
        self.dimensions = dimensions
        self.component_threshold = component_threshold
        self.principle_threshold = principle_threshold
        self.structural_threshold = structural_threshold

        # Core Engine Subsystems
        self.byte_grounder = ByteStructuralGrounding(sensor_dim=dimensions)
        self.semantic_engine = SemanticMassEngine(dimensions=dimensions)
        self.deconstruction_engine = DynamicDeconstructionEngine(deconstruction_threshold=structural_threshold)

        # Meta-Causal Principle Initialization
        alg_tensor = AlgorithmTensor("MetaPrincipleGraph")
        initial_param = np.ones(dimensions, dtype=np.float32) / np.sqrt(dimensions)
        alg_tensor.add_node("eval_node_v1", op_type="initial_transform", param_vector=initial_param)

        self.principle = MetaCausalPrinciple(
            principle_id="Principle_ZeroConvergence",
            name="Equilibrium & Spec Compliance Principle",
            param_vector=initial_param,
            algorithm_tensor=alg_tensor,
        )

        # Loop Cycle State
        self.cycle_count = 0
        self.last_report: Optional[FractalTensionReport] = None

    def process_cycle(
        self,
        raw_input: Union[bytes, bytearray, np.ndarray, Any],
        cycle_id: str,
        expected_format: Optional[str] = None,
        enforce_strict_byte: bool = False,
        target_memory_unit_id: Optional[str] = None,
    ) -> FractalTensionReport:
        """
        Executes one complete cycle of the recursive causal evolution loop across all 3 tiers.
        """
        self.cycle_count += 1
        details = {"cycle_id": cycle_id, "cycle_count": self.cycle_count}

        # ----------------------------------------------------------------------
        # Tier 1: Component Level Delta (\Delta_c)
        # ----------------------------------------------------------------------
        provenance_log = None
        grounding_payload = None

        if isinstance(raw_input, (bytes, bytearray)):
            try:
                provenance_log, grounding_payload = self.byte_grounder.process_and_ground_bytes(
                    raw_bytes=raw_input,
                    provenance_id=f"Prov_{cycle_id}",
                    expected_format=expected_format,
                    enforce_strict=enforce_strict_byte,
                )
                delta_c = float(grounding_payload["residual_tension"])
                feature_vec = grounding_payload["wave_spectrum"]
            except StructuralViolationError as e:
                delta_c = 1.0
                feature_vec = np.ones(self.dimensions, dtype=np.float32) / np.sqrt(self.dimensions)
                details["component_error"] = str(e)
        elif isinstance(raw_input, np.ndarray):
            feat = raw_input.flatten()
            feature_vec = feat[:self.dimensions] if len(feat) >= self.dimensions else np.pad(feat, (0, self.dimensions - len(feat)))
            feature_vec = feature_vec / (np.linalg.norm(feature_vec) + 1e-9)
            delta_c = float(np.clip(1.0 - np.linalg.norm(feature_vec), 0.0, 1.0))
        else:
            feature_vec = np.ones(self.dimensions, dtype=np.float32) / np.sqrt(self.dimensions)
            delta_c = 0.5

        details["delta_c"] = delta_c

        # ----------------------------------------------------------------------
        # Tier 2: Principle Level Delta (\Delta_p) & Meta-Principle Evolution
        # ----------------------------------------------------------------------
        delta_p = self.principle.evaluate_principle_tension(
            component_delta=delta_c,
            feature_vector=feature_vec,
        )
        details["delta_p_before"] = delta_p

        principle_evolved = False
        if delta_p >= self.principle_threshold:
            # Trigger Meta-Principle Evolution!
            self.principle.evolve_principle(
                trigger_delta_p=delta_p,
                feature_vector=feature_vec,
                reason=f"Cycle {cycle_id} persistent principle friction",
            )
            principle_evolved = True

            # Recalculate delta_p after evolution (Equilibrium restored)
            delta_p = self.principle.evaluate_principle_tension(
                component_delta=delta_c,
                feature_vector=feature_vec,
            )

        details["delta_p_after"] = delta_p
        details["principle_version"] = self.principle.evolution_version
        details["principle_evolved"] = principle_evolved

        # ----------------------------------------------------------------------
        # Tier 3: Structural Level Delta (\Delta_s) & Topology Evolution
        # ----------------------------------------------------------------------
        # Interaction with SemanticMassEngine & GenealogicalMemoryUnit
        sem_res = self.semantic_engine.process_interaction(
            external_friction=feature_vec,
            trinitarian_contrast=1.0 + delta_p,
        )

        semantic_mass = sem_res["semantic_mass"]
        causal_curvature = sem_res["causal_curvature"]

        # Structural delta is derived from curvature and combined friction
        delta_s = float(np.clip(0.5 * delta_p + 0.5 * (1.0 - np.exp(-causal_curvature)), 0.0, 1.0))
        details["delta_s_before"] = delta_s

        memory_deconstructed = False
        rewoven_unit_id = None

        if target_memory_unit_id and target_memory_unit_id in self.deconstruction_engine.memory_units:
            decon_triggered, rewoven_unit, decon_report = self.deconstruction_engine.evaluate_and_deconstruct(
                target_unit_id=target_memory_unit_id,
                new_raw_friction=feature_vec,
                semantic_engine=self.semantic_engine,
                new_label_if_rewoven=f"EvolvedMemory_C{self.cycle_count}",
            )
            memory_deconstructed = decon_triggered
            if decon_triggered and rewoven_unit:
                rewoven_unit_id = rewoven_unit.unit_id
                details["deconstruction_report"] = decon_report
        else:
            # Auto-register new memory unit
            unit = self.deconstruction_engine.create_and_register_unit(
                unit_id=f"MemUnit_{cycle_id}",
                label=f"StructuralStrata_C{self.cycle_count}",
                raw_friction=raw_input if isinstance(raw_input, (bytes, bytearray)) else feature_vec,
                semantic_engine=self.semantic_engine,
                expected_format=expected_format,
                enforce_strict_byte_spec=enforce_strict_byte,
            )
            details["created_unit_id"] = unit.unit_id

        # ----------------------------------------------------------------------
        # Top-Down Feedback Dynamics (\Delta -> 0)
        # ----------------------------------------------------------------------
        # Structural stabilization cools down principle and component deltas
        if principle_evolved or memory_deconstructed or rewoven_unit_id:
            # Feedback relaxation
            delta_s *= 0.2
            delta_p *= 0.2
            delta_c *= 0.2

        total_fractal_tension = float(np.sqrt((delta_c ** 2 + delta_p ** 2 + delta_s ** 2) / 3.0))
        is_zero_converged = total_fractal_tension < 1e-3

        report = FractalTensionReport(
            delta_c=delta_c,
            delta_p=delta_p,
            delta_s=delta_s,
            total_fractal_tension=total_fractal_tension,
            is_zero_converged=is_zero_converged,
            principle_evolved=principle_evolved,
            memory_deconstructed=memory_deconstructed,
            details=details,
        )

        self.last_report = report
        return report
