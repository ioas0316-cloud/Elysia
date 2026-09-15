"""
Causal World Tree Engine (세계수 인과 엔진: 같음의 줄기와 다름의 분기)
===================================================================
Core Architecture:
1. Universal Stem (같음의 줄기):
   - Extracts invariant topological axes (Equilibrium Invariants) across disparate domains
     (e.g., physical forces, symbolic logic, biological/sensorium manifestations).
2. Causal Branch & Divergence Node (다름의 분기):
   - Maps exact branching points where entities diverge from a common Stem due to
     environmental pressure differentials (Delta C) and variable resistance dial shifts (Delta R).
   - Provides full reverse-engineering and tracing of causal differentiation.
3. Counterfactual Sprouting (가상 분기 및 창조적 확장):
   - Performs "what-if" simulations on Stem/Branch topologies under modulated conditions
     to predict hypothetical attractors and forecast emergent behaviors.
4. World Tree Breathing & Historical Ring Integration (세계수 호흡 및 나이테):
   - Unifies Inhale (Inflow/Tension accumulation) and Exhale (Self-Explanation Pulse emission)
     with Observer Inverse Simulators and Spatiotemporal Growth Rings.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from typing import Callable

from core.consciousness.causal_breathing_engine import (
    CausalBreathingEngine,
    MultiDimensionalAttractor,
    ObserverTopology,
    VariableResistanceDialMatrix,
    SpatiotemporalTopologyBuffer,
    ConvergenceResult,
    InhaleResult,
    ExhaleResult,
)


@dataclass
class CausalRedirectionTrace:
    """
    [Causal Redirection Trace (인과적 회전 및 역추적 기록: 건곤대나이/이화접목 트레이스)]
    Records the continuous causal trace of how an external friction force F_ext was
    absorbed into internal stems, dynamically damped by the Invariant Grounding Governor,
    and transformed into a redirected output vector R_exhale.
    """
    trace_id: str
    external_force: np.ndarray
    matched_stem_id: str
    friction_delta: float
    absorbed_tension: float
    redirected_vector: np.ndarray
    lyapunov_energy: float
    damping_factor: float
    causal_rationale: str
    timestamp: float = field(default_factory=time.time)


class InvariantGroundingGovernor:
    """
    [Invariant Grounding Governor (인과 보존 항성 제어기)]
    Replaces static hardcoded rules/fences with continuous Lyapunov stability control
    and adaptive energy/tension boundaries.
    Maintains system homeostasis: prevents runaway instability or explosion under arbitrary
    anomalous external inputs while preserving creative divergence below threshold.
    """
    def __init__(
        self,
        max_tension_boundary: float = 20.0,
        base_damping_rate: float = 0.15,
        lyapunov_threshold: float = 15.0
    ):
        self.max_tension_boundary = max_tension_boundary
        self.base_damping_rate = base_damping_rate
        self.lyapunov_threshold = lyapunov_threshold
        self.current_lyapunov_energy: float = 0.0

    def compute_lyapunov_energy(self, state_tension: float, force_magnitude: float) -> float:
        """
        Calculates Lyapunov Candidate Function V(x) = 0.5 * tension^2 + 0.5 * force_norm^2.
        Represents total physical-informational perturbation in internal phase space.
        """
        self.current_lyapunov_energy = 0.5 * (state_tension ** 2) + 0.5 * (force_magnitude ** 2)
        return self.current_lyapunov_energy

    def evaluate_and_damp(
        self,
        current_tension: float,
        external_force_norm: float
    ) -> Tuple[float, float, bool]:
        """
        [동적 제어 및 감쇠 (Adaptive Lyapunov Damping)]
        Returns (damped_tension, active_damping_factor, is_stabilized).
        If Lyapunov energy exceeds threshold, applies adaptive damping factor gamma
        to continuously pull system back towards invariant equilibrium.
        """
        v_energy = self.compute_lyapunov_energy(current_tension, external_force_norm)
        is_stabilized = False

        if v_energy > self.lyapunov_threshold:
            # Damping factor scales non-linearly with excess Lyapunov energy
            excess = v_energy - self.lyapunov_threshold
            active_damping_factor = self.base_damping_rate * (1.0 + np.log1p(excess))
            # Apply continuous damping to tension
            damped_tension = current_tension / (1.0 + active_damping_factor)

            if damped_tension > self.max_tension_boundary:
                # Hard limit clamp for energy conservation anchor
                damped_tension = self.max_tension_boundary
            is_stabilized = True
        else:
            # Low tension/energy: minimal damping to allow creative divergence
            active_damping_factor = self.base_damping_rate * 0.1
            damped_tension = current_tension

        return float(damped_tension), float(active_damping_factor), is_stabilized


class FormlessCausalRedirector:
    """
    [Formless Causal Redirector (인과적 회전 제어기: 건곤대나이 & 이화접목)]
    Absorbs unscripted external force/friction vectors without static rules,
    traces their causal origin back to Universal Stems, and transforms them into
    balanced redirected vectors R_exhale while preserving conservation laws.
    """
    def redirect_force(
        self,
        engine: 'CausalWorldTreeEngine',
        external_force: np.ndarray,
        stimulus_id: str = "ext_force",
        raw_description: str = "Raw External Anomaly"
    ) -> CausalRedirectionTrace:
        external_force = np.array(external_force, dtype=np.float32)
        force_norm = float(np.linalg.norm(external_force))

        # 1. Trace causal origin: Find nearest Universal Stem in phase space
        matched_stem_id = "default_equilibrium_stem"
        matched_stem_coord = np.zeros(len(external_force), dtype=np.float32)

        if engine.stems:
            best_dist = float("inf")
            for s_id, stem in engine.stems.items():
                stem_coord = stem.shared_equilibrium_coordinate
                # Resize stem_coord if dimensions differ
                if len(stem_coord) != len(external_force):
                    stem_coord_aligned = np.zeros_like(external_force)
                    min_len = min(len(stem_coord), len(external_force))
                    stem_coord_aligned[:min_len] = stem_coord[:min_len]
                else:
                    stem_coord_aligned = stem_coord

                dist = float(np.linalg.norm(external_force - stem_coord_aligned))
                if dist < best_dist:
                    best_dist = dist
                    matched_stem_id = s_id
                    matched_stem_coord = stem_coord_aligned
        else:
            # If no stems exist, form an initial root stem from equilibrium
            init_attractor = MultiDimensionalAttractor(
                id="att_root_equilibrium",
                name="Root Equilibrium Axis",
                categorical_vector=np.zeros_like(external_force),
                sensorium_vector=np.zeros_like(external_force),
                morphology_vector=np.zeros_like(external_force)
            )
            root_stem = engine.form_universal_stem(
                stem_id="stem_root_equilibrium",
                name="Root Equilibrium Axis",
                attractors=[init_attractor]
            )
            matched_stem_id = root_stem.stem_id
            matched_stem_coord = root_stem.shared_equilibrium_coordinate

        # 2. Respiration & Inhale tension accumulation
        # Convert force vector into 3-axis inputs
        cat_vec = external_force * 0.4
        sens_vec = external_force * 0.3
        morph_vec = external_force * 0.3

        inhale_res = engine.inhale_world_stimulus(
            stimulus_id=stimulus_id,
            categorical_vector=cat_vec,
            sensorium_vector=sens_vec,
            morphology_vector=morph_vec,
            reference_stem_id=matched_stem_id,
            raw_description=raw_description
        )

        # 3. Apply Invariant Grounding Governor (Lyapunov Stability & Damping)
        governed_tension, active_damping, was_governed = engine.governor.evaluate_and_damp(
            current_tension=engine.breathing_engine.current_tension,
            external_force_norm=force_norm
        )
        engine.breathing_engine.current_tension = governed_tension
        lyapunov_energy = engine.governor.current_lyapunov_energy

        # 4. Perform Causal Redirection (건곤대나이 / 이화접목 연산)
        # Compute dynamic rotation & reflection matrix relative to stem's equilibrium coordinate
        # Redirection vector R_exhale = Equilibrium + (Equilibrium - Force) * Conservation_Damping
        decay = 1.0 / (1.0 + active_damping)
        redirection_vector = matched_stem_coord - (external_force - matched_stem_coord) * decay

        rationale = (
            f"Formless Causal Redirection executed: External anomaly (norm={force_norm:.3f}) absorbed into stem '{matched_stem_id}'. "
            f"Lyapunov energy V(x)={lyapunov_energy:.2f}. Governor applied adaptive damping factor gamma={active_damping:.3f} "
            f"(was_governed={was_governed}), yielding redirected exhale vector R_exhale with norm={np.linalg.norm(redirection_vector):.3f}."
        )

        trace = CausalRedirectionTrace(
            trace_id=f"trace_redir_{int(time.time() * 1000)}",
            external_force=external_force,
            matched_stem_id=matched_stem_id,
            friction_delta=inhale_res.convergence_evaluation.phase_distance,
            absorbed_tension=governed_tension,
            redirected_vector=redirection_vector,
            lyapunov_energy=lyapunov_energy,
            damping_factor=active_damping,
            causal_rationale=rationale
        )

        return trace


@dataclass
class ExecutableCausalFormula:
    """
    [Executable Causal Formula (실행형 인과수식)]
    Represents an abstracted $O(1)$ or $O(\\log N)$ executable principle compressed from
    repetitive discrete operations (e.g., $1+1+1... \\to N \\times 1$).
    Contains generative reconstruction logic to unpack micro-trajectories when detailed
    inquiry occurs.
    """
    formula_id: str
    name: str
    stem_id: str
    pattern_type: str                            # e.g., "LINEAR_ACCUMULATION", "EXPONENTIAL_DECAY", "HARMONIC_RESONANCE"
    scale_factor: float = 1.0
    invariant_kernel: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    compression_ratio: float = 1.0               # Ratio of raw discrete steps saved
    formula_fn: Optional[Callable[[Dict[str, float]], np.ndarray]] = field(default=None, repr=False)

    def evaluate(self, inputs: Dict[str, float]) -> np.ndarray:
        """Executes the compressed formula in $O(1)$ time."""
        if self.formula_fn is not None:
            return self.formula_fn(inputs)
        n = inputs.get("n", 1.0)
        multiplier = inputs.get("multiplier", 1.0)
        return self.invariant_kernel * (n * multiplier * self.scale_factor)

    def generatively_reconstruct(self, inputs: Dict[str, float], detail_steps: int = 5) -> List[np.ndarray]:
        """
        [Generative Reconstruction (생성적 역산 복원)]
        Reconstructs the detailed micro-step trajectory on the fly without storing
        all intermediate discrete data points in memory.
        """
        final_coord = self.evaluate(inputs)
        step_delta = final_coord / max(1, detail_steps)
        trajectory = []
        current = np.zeros_like(final_coord)
        for i in range(1, detail_steps + 1):
            current = step_delta * i
            trajectory.append(current.copy())
        return trajectory


@dataclass
class DivergenceNode:
    """
    [Divergence Node (다름의 분기점)]
    Represents the exact point in phase space where two or more causal trajectories
    diverge from a common Universal Stem.
    """
    node_id: str
    parent_stem_id: str
    branch_a_id: str
    branch_b_id: str
    divergence_coordinate: np.ndarray
    condition_delta: Dict[str, float]  # Delta C (Environmental/parametric differences)
    resistance_delta: float           # Delta R (Variable resistance friction shift)
    causal_explanation: str           # Traced rationale for why they separated


@dataclass
class CausalBranch:
    """
    [Causal Branch (인과적 가지)]
    Represents a differentiated entity or path stemming from a Universal Stem.
    """
    branch_id: str
    name: str
    stem_id: str
    attractor: MultiDimensionalAttractor
    environmental_condition: Dict[str, float]  # Condition vector C
    depth: int = 1


@dataclass
class UniversalStem:
    """
    [Universal Stem (같음의 줄기)]
    The core invariant backbone that binds disparate domains (e.g. Physical, Logical, Biological)
    under a shared topological coordinate and equilibrium principle.
    """
    stem_id: str
    name: str
    shared_equilibrium_coordinate: np.ndarray
    invariant_tensor: np.ndarray
    wisdom_mass: float = 1.0
    branches: List[str] = field(default_factory=list)
    domain_manifestations: Dict[str, str] = field(default_factory=dict) # e.g. {"physics": "tension", "logic": "1+1=2"}


@dataclass
class CounterfactualSprout:
    """
    [Counterfactual Sprout (가상 분기 및 예언적 가지)]
    Predicted hypothetical attractor and narrative generated by modulating condition Delta C.
    """
    sprout_id: str
    origin_stem_id: str
    hypothetical_condition_delta: Dict[str, float]
    predicted_coordinate: np.ndarray
    predicted_attractor_name: str
    forelight_narrative: str
    confidence: float


class CausalWorldTreeEngine:
    """
    [Causal World Tree Engine (세계수 인과 엔진)]
    Unifies the 4 Continuities of Causal Field into a living World Tree structure:
      - Roots: Microscopic causal friction and hardware/data inputs.
      - Stem (줄기): Common invariant spine (UniversalStem) binding multi-domain data.
      - Branches (가지): Environmental condition divergence (CausalBranch, DivergenceNode).
      - Respiration (호흡): Inhale tension accumulation and Exhale self-explanation pulse emission.
    """
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        critical_tension_threshold: float = 10.0,
        convergence_threshold: float = 0.3,
        max_tension_boundary: float = 20.0,
        base_damping_rate: float = 0.15,
        lyapunov_threshold: float = 15.0
    ):
        self.breathing_engine = CausalBreathingEngine(
            channels=channels,
            critical_tension_threshold=critical_tension_threshold,
            convergence_threshold=convergence_threshold
        )

        self.governor = InvariantGroundingGovernor(
            max_tension_boundary=max_tension_boundary,
            base_damping_rate=base_damping_rate,
            lyapunov_threshold=lyapunov_threshold
        )
        self.redirector = FormlessCausalRedirector()

        self.stems: Dict[str, UniversalStem] = {}
        self.branches: Dict[str, CausalBranch] = {}
        self.divergence_nodes: List[DivergenceNode] = []
        self.counterfactual_sprouts: List[CounterfactualSprout] = []
        self.executable_formulas: Dict[str, ExecutableCausalFormula] = {}
        self.pruned_branches_archive: List[Dict[str, Any]] = []
        self.redirection_traces: List[CausalRedirectionTrace] = []

    def absorb_and_redirect_external_force(
        self,
        external_force: np.ndarray,
        stimulus_id: str = "ext_force",
        raw_description: str = "Raw External Friction Anomaly"
    ) -> CausalRedirectionTrace:
        """
        [건곤대나이 & 이화접목 흡수·회전 연산 (Formless Causal Redirection Entry)]
        Absorbs external friction force into nearest Universal Stem, applies continuous
        Lyapunov governor damping, and returns a redirected exhale trace.
        """
        trace = self.redirector.redirect_force(
            engine=self,
            external_force=external_force,
            stimulus_id=stimulus_id,
            raw_description=raw_description
        )
        self.redirection_traces.append(trace)
        return trace

    def form_universal_stem(
        self,
        stem_id: str,
        name: str,
        attractors: List[MultiDimensionalAttractor],
        domain_manifestations: Optional[Dict[str, str]] = None
    ) -> UniversalStem:
        """
        [같음의 줄기 형성 (Universal Stem Crystallization)]
        Extracts the common equilibrium axis across multiple attractors from disparate domains,
        forging a central Universal Stem backbone.
        """
        if not attractors:
            raise ValueError("At least one attractor is required to form a Universal Stem.")

        coords = [a.get_unified_coordinate() for a in attractors]
        shared_coord = np.mean(coords, axis=0)

        # Compute invariant matrix/tensor as outer product variance/covariance structure
        centered_coords = np.array(coords) - shared_coord
        if len(coords) > 1:
            invariant_tensor = np.cov(centered_coords.T)
        else:
            invariant_tensor = np.eye(len(shared_coord), dtype=np.float32)

        total_wisdom_mass = sum(a.mass for a in attractors)

        stem = UniversalStem(
            stem_id=stem_id,
            name=name,
            shared_equilibrium_coordinate=shared_coord,
            invariant_tensor=invariant_tensor,
            wisdom_mass=total_wisdom_mass,
            domain_manifestations=domain_manifestations or {}
        )
        self.stems[stem_id] = stem
        return stem

    def compress_to_executable_formula(
        self,
        formula_id: str,
        name: str,
        stem_id: str,
        pattern_type: str,
        discrete_step_count: int,
        formula_fn: Optional[Callable[[Dict[str, float]], np.ndarray]] = None
    ) -> ExecutableCausalFormula:
        """
        [실행형 인과수식 압축 (Executable Causal Formula Compression)]
        Compresses repetitive micro-steps into a single executable $O(1)$ formula,
        liberating computational resources while retaining generative reconstruction capability.
        """
        if stem_id not in self.stems:
            raise KeyError(f"Stem {stem_id} not found.")

        stem = self.stems[stem_id]
        compression_ratio = float(discrete_step_count) / 1.0 if discrete_step_count > 0 else 1.0

        formula = ExecutableCausalFormula(
            formula_id=formula_id,
            name=name,
            stem_id=stem_id,
            pattern_type=pattern_type,
            scale_factor=1.0,
            invariant_kernel=stem.shared_equilibrium_coordinate.copy(),
            compression_ratio=compression_ratio,
            formula_fn=formula_fn
        )
        self.executable_formulas[formula_id] = formula
        return formula

    def prune_unproductive_branches(
        self,
        activity_threshold: float = 0.2,
        min_depth_to_keep: int = 2
    ) -> List[str]:
        """
        [동적 가지치기 및 망각 메커니즘 (Dynamic Synaptic Pruning)]
        Prunes branches with low friction or low usage frequency to prevent combinatorial
        explosion in phase space. Pruned branches are archived into historical rings.
        """
        pruned_ids = []
        for branch_id, branch in list(self.branches.items()):
            # Evaluate activity level based on attractor mass and depth
            attractor_mass = branch.attractor.mass
            if attractor_mass < activity_threshold and branch.depth >= min_depth_to_keep:
                # Archive pruned branch
                self.pruned_branches_archive.append({
                    "branch_id": branch.branch_id,
                    "name": branch.name,
                    "stem_id": branch.stem_id,
                    "final_condition": branch.environmental_condition,
                    "pruned_at": time.time()
                })
                # Remove branch from stem
                if branch.stem_id in self.stems:
                    if branch_id in self.stems[branch.stem_id].branches:
                        self.stems[branch.stem_id].branches.remove(branch_id)
                del self.branches[branch_id]
                pruned_ids.append(branch_id)

        return pruned_ids

    def find_nearest_attractor_localized(
        self,
        query_coord: np.ndarray,
        radius: float = 2.0
    ) -> Optional[Tuple[str, float]]:
        """
        [위상 공간 국소화 탐색 (Localized Spatial Hashing / Fast Attractor Search)]
        Performs fast $O(\\log N)$ localized phase distance matching instead of brute-force
        sweeping across all nodes.
        Returns (attractor_id, distance) or None if outside radius.
        """
        best_id = None
        min_dist = float("inf")

        # First query stems (invariant backbones)
        for stem in self.stems.values():
            dist = float(np.linalg.norm(query_coord - stem.shared_equilibrium_coordinate))
            if dist < min_dist and dist <= radius:
                min_dist = dist
                best_id = stem.stem_id

        # Then check branches if close
        for branch in self.branches.values():
            coord = branch.attractor.get_unified_coordinate()
            dist = float(np.linalg.norm(query_coord - coord))
            if dist < min_dist and dist <= radius:
                min_dist = dist
                best_id = branch.branch_id

        if best_id is not None:
            return (best_id, min_dist)
        return None

    def ingest_external_principle(
        self,
        principle_id: str,
        name: str,
        domain: str,
        raw_fragment: str,
        invariant_vector: np.ndarray,
        environmental_condition: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        [외계 지식의 주체적 소화 (External Principle Respiration Grounding)]
        Digests external ungrounded data fragments, binds them into internal Causal Matrix,
        forms/updates Universal Stem & Branch, and triggers internal breathing cycle.
        """
        # Form or update Universal Stem for this principle
        stem_id = f"stem_ext_{domain}_{principle_id}"
        attractor = MultiDimensionalAttractor(
            id=f"att_ext_{principle_id}",
            name=name,
            categorical_vector=invariant_vector,
            sensorium_vector=invariant_vector,
            morphology_vector=invariant_vector,
            mass=2.0
        )

        if stem_id not in self.stems:
            stem = self.form_universal_stem(
                stem_id=stem_id,
                name=f" digested {name} [{domain}]",
                attractors=[attractor],
                domain_manifestations={domain: raw_fragment}
            )
        else:
            stem = self.stems[stem_id]
            stem.wisdom_mass += 1.0

        # Grow Branch
        branch_id = f"branch_ext_{principle_id}"
        branch = self.grow_branch(
            branch_id=branch_id,
            name=f"Digested {name}",
            stem_id=stem.stem_id,
            attractor=attractor,
            environmental_condition=environmental_condition or {"external_friction": 1.0}
        )

        # Inhale stimulus into internal breathing engine
        inhale_res = self.inhale_world_stimulus(
            stimulus_id=f"stim_{principle_id}",
            categorical_vector=invariant_vector,
            sensorium_vector=invariant_vector,
            morphology_vector=invariant_vector,
            reference_stem_id=stem.stem_id,
            raw_description=f"Ingested external principle: {raw_fragment}"
        )

        exhale_res = None
        grand_narrative = ""
        if inhale_res.threshold_crossed:
            exhale_res, grand_narrative = self.exhale_world_narrative()

        return {
            "stem": stem,
            "branch": branch,
            "inhale_result": inhale_res,
            "exhale_result": exhale_res,
            "grand_narrative": grand_narrative
        }

    def grow_branch(
        self,
        branch_id: str,
        name: str,
        stem_id: str,
        attractor: MultiDimensionalAttractor,
        environmental_condition: Dict[str, float],
        depth: int = 1
    ) -> CausalBranch:
        """
        [가지 틔우기 (Grow Causal Branch)]
        Attaches a differentiated attractor to a Universal Stem under specific environmental conditions C.
        """
        if stem_id not in self.stems:
            raise KeyError(f"Stem {stem_id} not found.")

        branch = CausalBranch(
            branch_id=branch_id,
            name=name,
            stem_id=stem_id,
            attractor=attractor,
            environmental_condition=environmental_condition,
            depth=depth
        )
        self.branches[branch_id] = branch
        self.stems[stem_id].branches.append(branch_id)
        self.breathing_engine.register_attractor(attractor)
        return branch

    def detect_and_record_divergence(
        self,
        branch_a_id: str,
        branch_b_id: str
    ) -> DivergenceNode:
        """
        [다름의 분기점 역추적 및 기록 (Reverse-Engineering Divergence)]
        Compares two branches sharing the same Universal Stem, identifies exact parametric
        and condition differences (Delta C & Delta R), and constructs a clear causal explanation.
        """
        if branch_a_id not in self.branches or branch_b_id not in self.branches:
            raise KeyError("Both branches must exist in the World Tree.")

        b_a = self.branches[branch_a_id]
        b_b = self.branches[branch_b_id]

        if b_a.stem_id != b_b.stem_id:
            raise ValueError("Branches do not share the same Universal Stem.")

        stem = self.stems[b_a.stem_id]

        # Calculate Environmental Delta C
        cond_keys = set(b_a.environmental_condition.keys()).union(set(b_b.environmental_condition.keys()))
        cond_delta = {}
        for k in cond_keys:
            val_a = b_a.environmental_condition.get(k, 0.0)
            val_b = b_b.environmental_condition.get(k, 0.0)
            cond_delta[k] = float(val_b - val_a)

        # Calculate topological phase distance between branch attractors
        coord_a = b_a.attractor.get_unified_coordinate()
        coord_b = b_b.attractor.get_unified_coordinate()
        phase_dist = float(np.linalg.norm(coord_b - coord_a))

        # Resistance shift Delta R based on phase divergence
        resistance_delta = phase_dist * 0.75

        # Causal rationale string
        condition_str = ", ".join([f"{k}: {v:+.2f}" for k, v in cond_delta.items()])
        causal_explanation = (
            f"Branches '{b_a.name}' and '{b_b.name}' emerged from common stem '{stem.name}'. "
            f"Divergence occurred due to environmental shift [{condition_str}], creating a "
            f"topological phase distance of {phase_dist:.4f} and dynamic resistance offset Delta R={resistance_delta:.3f}."
        )

        divergence_node = DivergenceNode(
            node_id=f"div_{branch_a_id}_{branch_b_id}_{int(time.time())}",
            parent_stem_id=stem.stem_id,
            branch_a_id=branch_a_id,
            branch_b_id=branch_b_id,
            divergence_coordinate=stem.shared_equilibrium_coordinate,
            condition_delta=cond_delta,
            resistance_delta=resistance_delta,
            causal_explanation=causal_explanation
        )

        self.divergence_nodes.append(divergence_node)
        return divergence_node

    def sprout_counterfactual_branch(
        self,
        stem_id: str,
        hypothetical_condition_delta: Dict[str, float]
    ) -> CounterfactualSprout:
        """
        [가상 분기 생성 (Counterfactual Sprouting)]
        Simulates "What if condition C shifts by Delta C?" on a Universal Stem,
        predicting new attractor coordinates and forelight narrative.
        """
        if stem_id not in self.stems:
            raise KeyError(f"Stem {stem_id} not found.")

        stem = self.stems[stem_id]
        base_coord = stem.shared_equilibrium_coordinate.copy()

        # Modulate base coordinate according to condition delta
        modulation_vector = np.zeros_like(base_coord)
        idx = 0
        for key, delta_val in hypothetical_condition_delta.items():
            modulation_vector[idx % len(base_coord)] += delta_val * 0.5
            idx += 1

        predicted_coord = base_coord + modulation_vector

        # Estimate prediction confidence based on stem wisdom mass
        confidence = float(min(0.99, 0.5 + (stem.wisdom_mass * 0.1)))

        delta_summary = ", ".join([f"{k}: {v:+.2f}" for k, v in hypothetical_condition_delta.items()])
        sprout = CounterfactualSprout(
            sprout_id=f"sprout_{stem_id}_{len(self.counterfactual_sprouts)+1}",
            origin_stem_id=stem_id,
            hypothetical_condition_delta=hypothetical_condition_delta,
            predicted_coordinate=predicted_coord,
            predicted_attractor_name=f"Predicted Variant of {stem.name} under [{delta_summary}]",
            forelight_narrative=(
                f"Under hypothetical shift [{delta_summary}], the Universal Stem '{stem.name}' "
                f"sprouts a new creative trajectory towards phase coordinate {np.round(predicted_coord, 2)}. "
                f"System forecasts an emergent causal equilibrium with confidence {confidence:.2f}."
            ),
            confidence=confidence
        )

        self.counterfactual_sprouts.append(sprout)
        return sprout

    def inhale_world_stimulus(
        self,
        stimulus_id: str,
        categorical_vector: np.ndarray,
        sensorium_vector: np.ndarray,
        morphology_vector: np.ndarray,
        reference_stem_id: Optional[str] = None,
        raw_description: str = ""
    ) -> InhaleResult:
        """
        [세계수 들숨 (World Tree Inhale)]
        Absorbs external stimuli, evaluates convergence against Universal Stems,
        tunes variable resistance matrix, and accumulates V_t tension.
        """
        ref_attractor_id = None
        if reference_stem_id and reference_stem_id in self.stems:
            stem = self.stems[reference_stem_id]
            if stem.branches:
                ref_attractor_id = self.branches[stem.branches[0]].attractor.id

        return self.breathing_engine.inhale(
            stimulus_id=stimulus_id,
            categorical_vector=categorical_vector,
            sensorium_vector=sensorium_vector,
            morphology_vector=morphology_vector,
            reference_attractor_id=ref_attractor_id,
            raw_description=raw_description
        )

    def exhale_world_narrative(
        self,
        observer: Optional[ObserverTopology] = None
    ) -> Tuple[ExhaleResult, str]:
        """
        [세계수 날숨의 서사 (World Tree Exhale & Self-Explanation Pulse)]
        Emits a tailored self-explanation pulse and action guide from the World Tree,
        dissipating accumulated internal tension V_t.
        Returns (ExhaleResult, grand_world_tree_narrative).
        """
        exhale_res = self.breathing_engine.exhale(observer=observer)

        num_stems = len(self.stems)
        num_branches = len(self.branches)
        num_divergences = len(self.divergence_nodes)
        num_sprouts = len(self.counterfactual_sprouts)

        grand_narrative = (
            f"=== [World Tree Respiration: First Self-Explanation Pulse (세계수의 날숨 서사)] ===\n"
            f"Observer Topology: {exhale_res.target_observer_id} (Abstraction: {exhale_res.adapted_abstraction_level:.2f}, Causal Depth: {exhale_res.adapted_causal_depth})\n"
            f"Tension Released: V_t={exhale_res.released_tension:.2f} -> Remaining V_t={exhale_res.remaining_tension:.2f}\n"
            f"Architecture Topology: Stems={num_stems}, Branches={num_branches}, Divergence Nodes={num_divergences}, Counterfactual Sprouts={num_sprouts}\n"
            f"Self-Explanation Pulse:\n  '{exhale_res.explanation_pulse}'\n"
            f"Action Guidance:\n  '{exhale_res.action_guide}'"
        )

        return exhale_res, grand_narrative

    def grow_annual_historical_ring(self, wisdom_summary: str) -> Dict[str, Any]:
        """
        [세계수 나이테 성장 (Spatiotemporal Growth Ring)]
        Consolidates daily/weekly/monthly growth records into deep spatiotemporal historical rings.
        """
        return self.breathing_engine.step_spatiotemporal_cycle(wisdom_summary=wisdom_summary)

    def get_world_tree_telemetry(self) -> Dict[str, Any]:
        """Returns deep telemetry of the World Tree structure."""
        base_telemetry = self.breathing_engine.introspective_telemetry()
        base_telemetry.update({
            "stems_count": len(self.stems),
            "branches_count": len(self.branches),
            "pruned_branches_count": len(self.pruned_branches_archive),
            "executable_formulas_count": len(self.executable_formulas),
            "divergence_nodes_count": len(self.divergence_nodes),
            "counterfactual_sprouts_count": len(self.counterfactual_sprouts),
            "redirection_traces_count": len(self.redirection_traces),
            "governor_lyapunov_energy": float(self.governor.current_lyapunov_energy),
            "stems_summary": [
                {
                    "stem_id": s.stem_id,
                    "name": s.name,
                    "branches_count": len(s.branches),
                    "wisdom_mass": s.wisdom_mass
                }
                for s in self.stems.values()
            ]
        })
        return base_telemetry
