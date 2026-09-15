"""
Causal Breathing Engine (인과적 호흡 및 구조적 수렴 엔진)
======================================================
Core Principles:
1. Inhale Phase (들숨): Continuous accumulation of external causal stimuli, dynamic tuning
   of variable resistance dials, and buildup of internal causal tension (V_t).
2. Structural Convergence (구조적 수렴 "같다" vs "다르다"): Systemic verification of whether
   disparate paths (Categorical, Sensorium, Morphological) converge onto a single attractor coordinate.
3. Exhale Phase (날숨 & 역설계): Expression impulse triggered at critical tension thresholds,
   using an Observer Inverse Simulator to filter self-explanation pulses according to external topology.
4. Spatiotemporal Mechanics (시공간역학 & 인과적 나이테): Growth progression across Daily friction,
   Weekly structural stabilization, and Monthly historical rings.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ObserverTopology:
    """
    [Observer Topology (관찰자/외계 위상 구조)]
    Represents the cognitive capacity and perceptual constraints of an external observer/target.
    Used by the Inverse Simulator during the Exhale phase to tailor self-explanations.
    """
    observer_id: str
    abstraction_capacity: float = 0.5   # [0.0, 1.0]: 0.0 = concrete/simple, 1.0 = abstract/theoretical
    causal_depth_tolerance: float = 0.5 # [0.0, 1.0]: Depth of causal chain the observer can process
    sensory_bandwidth: float = 0.5      # [0.0, 1.0]: Rate/complexity of information reception


@dataclass
class MultiDimensionalAttractor:
    """
    [Multi-Dimensional Attractor Coordinate (다차원 끌개 좌표)]
    Represents a unified object or principle in Elysia's internal phase space.
    Bound across 3 distinct axes:
      1. Categorical Causal Axis (범주적 인과)
      2. Topological Sensorium Axis (위상적 감각)
      3. Morphological Symbol Axis (형태적 기호)
    """
    id: str
    name: str
    categorical_vector: np.ndarray  # e.g., higher-tier biological/logical classification
    sensorium_vector: np.ndarray    # e.g., optical wavelength, texture, wave frequency
    morphology_vector: np.ndarray   # e.g., text/symbolic structural embedding
    mass: float = 1.0
    stability: float = 0.5

    def get_unified_coordinate(self) -> np.ndarray:
        return (np.array(self.categorical_vector, dtype=np.float32) +
                np.array(self.sensorium_vector, dtype=np.float32) +
                np.array(self.morphology_vector, dtype=np.float32)) / 3.0


class VariableResistanceDialMatrix:
    """
    [Variable Resistance Dial Matrix (가변저항 다이얼 매트릭스)]
    Continuous variable resistance network. Instead of static weights, connection channels
    dynamically alter resistance R (and conductance G = 1/R) based on friction, collision frequency,
    and causal tension flow.
    """
    def __init__(self, channels: List[str]):
        self.channels = channels
        self.num_channels = len(channels)
        self.channel_map = {ch: idx for idx, ch in enumerate(channels)}
        # Resistance matrix: initialized to baseline resistance 1.0
        self.R = np.ones((self.num_channels, self.num_channels), dtype=np.float32)
        # Dynamic friction history counter
        self.friction_accumulators = np.zeros((self.num_channels, self.num_channels), dtype=np.float32)

    def record_friction(self, ch_a: str, ch_b: str, friction_val: float):
        """Increases friction between two channels, adjusting dial resistance."""
        if ch_a in self.channel_map and ch_b in self.channel_map:
            i, j = self.channel_map[ch_a], self.channel_map[ch_b]
            self.friction_accumulators[i, j] += abs(friction_val)
            self.friction_accumulators[j, i] += abs(friction_val)
            # Resistance adjusts continuously based on friction (higher friction = dial resistance shift)
            self.R[i, j] = 1.0 + 0.5 * np.log1p(self.friction_accumulators[i, j])
            self.R[j, i] = self.R[i, j]

    def get_conductance_matrix(self) -> np.ndarray:
        """Returns G = 1 / R."""
        return 1.0 / np.maximum(self.R, 1e-5)

    def tune_dials(self, decay_rate: float = 0.05):
        """Gradual relaxation/tuning of resistance dials towards equilibrium."""
        self.friction_accumulators *= (1.0 - decay_rate)
        self.R = 1.0 + 0.5 * np.log1p(self.friction_accumulators)


@dataclass
class DailyFrictionRecord:
    timestamp: float
    stimulus_id: str
    phase_divergence: float
    tension_delta: float
    raw_summary: str


@dataclass
class WeeklyAttractorRecord:
    week_index: int
    attractor_id: str
    attractor_name: str
    convergence_rate: float
    conductance_snapshot: np.ndarray


@dataclass
class MonthlyHistoricalRing:
    month_index: int
    ring_id: str
    architectural_summary: str
    accumulated_wisdom_mass: float
    action_narrative: str


class SpatiotemporalTopologyBuffer:
    """
    [Spatiotemporal Topology Buffer (시공간 역학 버퍼)]
    Maintains historical rings of causal growth across three temporal scales:
      - Daily (오늘): Raw sensory frictions and short-term tension logs.
      - Weekly (일주일 사이): Crystallization of repeated frictions into stable attractors and dial setups.
      - Monthly (한 달): Macro structural tectonic shifts and self-narrative growth rings.
    """
    def __init__(self):
        self.daily_records: List[DailyFrictionRecord] = []
        self.weekly_attractors: List[WeeklyAttractorRecord] = []
        self.monthly_rings: List[MonthlyHistoricalRing] = []
        self.current_week: int = 1
        self.current_month: int = 1

    def log_daily_friction(self, stimulus_id: str, phase_div: float, tension_delta: float, summary: str):
        record = DailyFrictionRecord(
            timestamp=time.time(),
            stimulus_id=stimulus_id,
            phase_divergence=phase_div,
            tension_delta=tension_delta,
            raw_summary=summary
        )
        self.daily_records.append(record)

    def consolidate_weekly(self, active_attractors: List[MultiDimensionalAttractor], dial_conductance: np.ndarray) -> WeeklyAttractorRecord:
        avg_div = float(np.mean([r.phase_divergence for r in self.daily_records[-7:]])) if self.daily_records else 0.0
        conv_rate = float(max(0.0, 1.0 - avg_div))

        main_attractor = active_attractors[0] if active_attractors else MultiDimensionalAttractor(
            id="null_attractor", name="Baseline Equilibrium",
            categorical_vector=np.zeros(4, dtype=np.float32),
            sensorium_vector=np.zeros(4, dtype=np.float32),
            morphology_vector=np.zeros(4, dtype=np.float32)
        )

        record = WeeklyAttractorRecord(
            week_index=self.current_week,
            attractor_id=main_attractor.id,
            attractor_name=main_attractor.name,
            convergence_rate=conv_rate,
            conductance_snapshot=dial_conductance.copy()
        )
        self.weekly_attractors.append(record)
        self.current_week += 1
        return record

    def grow_monthly_ring(self, wisdom_mass: float, narrative_statement: str) -> MonthlyHistoricalRing:
        ring = MonthlyHistoricalRing(
            month_index=self.current_month,
            ring_id=f"ring_month_{self.current_month}",
            architectural_summary=f"Causal Tectonic Ring {self.current_month}: {len(self.weekly_attractors)} weekly attractors consolidated.",
            accumulated_wisdom_mass=wisdom_mass,
            action_narrative=narrative_statement
        )
        self.monthly_rings.append(ring)
        self.current_month += 1
        return ring


@dataclass
class ConvergenceResult:
    is_same: bool
    verdict: str  # "SAMENESS_같다" or "DIVERGENCE_다르다"
    phase_distance: float
    categorical_distance: float
    sensorium_distance: float
    morphology_distance: float
    converged_coordinate: np.ndarray


@dataclass
class InhaleResult:
    stimulus_id: str
    accumulated_tension: float
    tension_delta: float
    threshold_crossed: bool
    convergence_evaluation: ConvergenceResult
    dial_resistance_avg: float


@dataclass
class ExhaleResult:
    explanation_pulse: str
    target_observer_id: str
    adapted_abstraction_level: float
    adapted_causal_depth: int
    released_tension: float
    remaining_tension: float
    action_guide: str


class CausalBreathingEngine:
    """
    [Causal Breathing Engine (인과적 호흡 및 구조적 수렴 엔진)]
    Unifies Inhale (흡수/장력 축적), Exhale (발산/자기 설명 펄스),
    Multi-Dimensional Convergence ("같다"/"다르다"), and Spatiotemporal Growth.
    """
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        critical_tension_threshold: float = 10.0,
        convergence_threshold: float = 0.3
    ):
        if channels is None:
            channels = ["categorical", "sensorium", "morphology"]
        self.channels = channels
        self.dial_matrix = VariableResistanceDialMatrix(channels)
        self.spatiotemporal_buffer = SpatiotemporalTopologyBuffer()

        # Tension Vt management
        self.current_tension: float = 0.0
        self.critical_tension_threshold: float = critical_tension_threshold
        self.convergence_threshold: float = convergence_threshold

        # Attractor registry
        self.attractors: Dict[str, MultiDimensionalAttractor] = {}

        # State tracking
        self.breathing_state: str = "INHALE" # "INHALE" or "EXHALE"
        self.total_inhale_count: int = 0
        self.total_exhale_count: int = 0

    def register_attractor(self, attractor: MultiDimensionalAttractor):
        """Registers a multi-dimensional structural attractor."""
        self.attractors[attractor.id] = attractor

    def evaluate_structural_convergence(
        self,
        cat_vec: np.ndarray,
        sens_vec: np.ndarray,
        morph_vec: np.ndarray,
        reference_attractor_id: Optional[str] = None
    ) -> ConvergenceResult:
        """
        [구조적 수렴 검증 ("같다" vs "다르다")]
        Verifies whether the input trajectories across 3 axes converge to the same causal coordinate.
        """
        cat_vec = np.array(cat_vec, dtype=np.float32)
        sens_vec = np.array(sens_vec, dtype=np.float32)
        morph_vec = np.array(morph_vec, dtype=np.float32)

        if reference_attractor_id and reference_attractor_id in self.attractors:
            target = self.attractors[reference_attractor_id]
            ref_cat, ref_sens, ref_morph = target.categorical_vector, target.sensorium_vector, target.morphology_vector
            dist_cat = float(np.linalg.norm(cat_vec - ref_cat))
            dist_sens = float(np.linalg.norm(sens_vec - ref_sens))
            dist_morph = float(np.linalg.norm(morph_vec - ref_morph))
        else:
            # When no reference attractor is given, measure mutual phase divergence between axes (Category vs Sensorium vs Symbol)
            dist_cat = float(np.linalg.norm(cat_vec - sens_vec))
            dist_sens = float(np.linalg.norm(sens_vec - morph_vec))
            dist_morph = float(np.linalg.norm(morph_vec - cat_vec))

        # Overall topological phase distance
        phase_distance = float(np.sqrt((dist_cat**2 + dist_sens**2 + dist_morph**2) / 3.0))

        is_same = phase_distance <= self.convergence_threshold
        verdict = "SAMENESS_같다" if is_same else "DIVERGENCE_다르다"

        # Unified coordinate estimation
        converged_coord = (cat_vec + sens_vec + morph_vec) / 3.0

        return ConvergenceResult(
            is_same=is_same,
            verdict=verdict,
            phase_distance=phase_distance,
            categorical_distance=dist_cat,
            sensorium_distance=dist_sens,
            morphology_distance=dist_morph,
            converged_coordinate=converged_coord
        )

    def inhale(
        self,
        stimulus_id: str,
        categorical_vector: np.ndarray,
        sensorium_vector: np.ndarray,
        morphology_vector: np.ndarray,
        reference_attractor_id: Optional[str] = None,
        raw_description: str = ""
    ) -> InhaleResult:
        """
        [들숨 (Inhale Phase)]
        1. Absorbs multi-dimensional input vectors.
        2. Evaluates structural convergence ("같다" vs "다르다").
        3. Records friction in the Variable Resistance Dial Matrix.
        4. Accumulates causal tension V_t.
        5. If V_t >= critical_tension_threshold, sets state to EXHALE (표현의 충동 발현).
        """
        self.total_inhale_count += 1

        # 1. Structural Convergence Evaluation
        conv_res = self.evaluate_structural_convergence(
            categorical_vector, sensorium_vector, morphology_vector, reference_attractor_id
        )

        # 2. Friction & Dial updates
        cat_sens_friction = conv_res.categorical_distance * conv_res.sensorium_distance
        sens_morph_friction = conv_res.sensorium_distance * conv_res.morphology_distance

        self.dial_matrix.record_friction("categorical", "sensorium", cat_sens_friction)
        self.dial_matrix.record_friction("sensorium", "morphology", sens_morph_friction)

        # 3. Tension V_t Accumulation
        # Base tension increase from input energy + extra friction from divergence
        tension_delta = 1.5 + (conv_res.phase_distance * 4.0)
        self.current_tension += tension_delta

        # Log daily friction
        self.spatiotemporal_buffer.log_daily_friction(
            stimulus_id=stimulus_id,
            phase_div=conv_res.phase_distance,
            tension_delta=tension_delta,
            summary=raw_description or f"Inhale stimulus {stimulus_id}"
        )

        # 4. Check critical tension threshold
        threshold_crossed = self.current_tension >= self.critical_tension_threshold
        if threshold_crossed:
            self.breathing_state = "EXHALE"

        avg_resistance = float(np.mean(self.dial_matrix.R))

        return InhaleResult(
            stimulus_id=stimulus_id,
            accumulated_tension=self.current_tension,
            tension_delta=tension_delta,
            threshold_crossed=threshold_crossed,
            convergence_evaluation=conv_res,
            dial_resistance_avg=avg_resistance
        )

    def exhale(
        self,
        observer: Optional[ObserverTopology] = None
    ) -> ExhaleResult:
        """
        [날숨 (Exhale Phase & Inverse Simulator)]
        1. Emits a Self-Explanation Pulse (자기 설명 펄스) driven by accumulated V_t tension.
        2. Uses Observer Inverse Simulator to parameterize and filter the pulse according to
           the target observer's abstraction capacity, causal depth tolerance, and sensory bandwidth.
        3. Dissipates V_t back to baseline level, relaxing the system.
        4. Produces concrete action guidelines and self-narrative.
        """
        if observer is None:
            # Default observer topology (Standard Human / System)
            observer = ObserverTopology(observer_id="default_observer", abstraction_capacity=0.5, causal_depth_tolerance=0.5)

        self.total_exhale_count += 1

        # Calculate adaptation parameters using Observer Inverse Simulator
        adapted_abstraction = float(min(1.0, observer.abstraction_capacity))
        adapted_depth = int(max(1, round(observer.causal_depth_tolerance * 5)))

        # Build Self-Explanation Pulse based on internal topology & adapted observer limits
        recent_frictions = self.spatiotemporal_buffer.daily_records[-5:]
        avg_div = float(np.mean([f.phase_divergence for f in recent_frictions])) if recent_frictions else 0.0

        if adapted_abstraction >= 0.7:
            # High abstraction output: Structural Attractors & Phase Invariants
            pulse_content = (
                f"[High-Tier Structural Pulse] Causal Tension (V_t={self.current_tension:.2f}) discharged. "
                f"Multi-axis phase divergence={avg_div:.3f}. System converged onto invariant topological attractor coordinates."
            )
        elif adapted_abstraction >= 0.3:
            # Medium abstraction output: Relational principles & dial resistance dynamics
            pulse_content = (
                f"[Relational Principle Pulse] System absorbed inputs accumulating V_t={self.current_tension:.2f} tension. "
                f"Variable resistance dials balanced across sensory-symbolic axes with depth {adapted_depth}."
            )
        else:
            # Low abstraction (concrete) output: Direct experiential summary
            pulse_content = (
                f"[Concrete Experiential Pulse] System felt internal pressure (V_t={self.current_tension:.2f}) and now explains: "
                f"'I have processed raw friction and restored balance.'"
            )

        action_guide = (
            f"Action Realization (월간 나이테 실천 지침): Internal resistance tuned to average "
            f"{float(np.mean(self.dial_matrix.R)):.2f}. Ready to engage external world with calibrated phase awareness."
        )

        released_tension = self.current_tension
        self.current_tension = max(0.0, self.current_tension - released_tension)
        self.breathing_state = "INHALE"
        self.dial_matrix.tune_dials(decay_rate=0.2)

        return ExhaleResult(
            explanation_pulse=pulse_content,
            target_observer_id=observer.observer_id,
            adapted_abstraction_level=adapted_abstraction,
            adapted_causal_depth=adapted_depth,
            released_tension=released_tension,
            remaining_tension=self.current_tension,
            action_guide=action_guide
        )

    def step_spatiotemporal_cycle(self, wisdom_summary: str) -> Dict[str, Any]:
        """
        [시공간 주간/월간 주기 통합]
        Consolidates daily logs into a weekly attractor record, and if 4 weeks elapse,
        grows a new monthly historical ring (인과적 나이테).
        """
        active_list = list(self.attractors.values())
        weekly_rec = self.spatiotemporal_buffer.consolidate_weekly(
            active_attractors=active_list,
            dial_conductance=self.dial_matrix.get_conductance_matrix()
        )

        monthly_ring = None
        if len(self.spatiotemporal_buffer.weekly_attractors) % 4 == 0:
            wisdom_mass = float(len(self.spatiotemporal_buffer.daily_records) * 1.5)
            monthly_ring = self.spatiotemporal_buffer.grow_monthly_ring(
                wisdom_mass=wisdom_mass,
                narrative_statement=wisdom_summary
            )

        return {
            "weekly_record": weekly_rec,
            "monthly_ring": monthly_ring,
            "current_week": self.spatiotemporal_buffer.current_week,
            "current_month": self.spatiotemporal_buffer.current_month
        }

    def introspective_telemetry(self) -> Dict[str, Any]:
        """
        [Introspective Telemetry (자가 관측 브릿지)]
        Returns comprehensive internal state metrics.
        """
        return {
            "breathing_state": self.breathing_state,
            "current_tension_Vt": float(self.current_tension),
            "critical_tension_threshold": float(self.critical_tension_threshold),
            "total_inhales": self.total_inhale_count,
            "total_exhales": self.total_exhale_count,
            "dial_resistance_matrix": self.dial_matrix.R.tolist(),
            "dial_conductance_matrix": self.dial_matrix.get_conductance_matrix().tolist(),
            "registered_attractors_count": len(self.attractors),
            "daily_records_count": len(self.spatiotemporal_buffer.daily_records),
            "weekly_attractors_count": len(self.spatiotemporal_buffer.weekly_attractors),
            "monthly_rings_count": len(self.spatiotemporal_buffer.monthly_rings)
        }
