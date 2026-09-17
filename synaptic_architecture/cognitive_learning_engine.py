"""
Cognitive Learning Engine (인지형 학습 엔진)

Implementation based on Cognitive Learning Engine Design Principles:
1. Transition as 1st-class Atomic Unit (공리 1: 이행을 최소 단위로 기록)
2. Density Emergence & Hebbian Plasticity (공리 2: 반복 연결 강화, 미사용 약화)
3. Self-Modification Gate on Density Thresholds (공리 3: 자기수정의 문)
4. Multi-channel Continuous Symbol Grounding (공리 4: 기호접지)
5. Phase Transition Model (Ice, Water, Gas / 고체, 액체, 기체)
6. Dual Operational Modes: Forward Forecasting & Reverse Abductive Reasoning
7. Holonic Meta-Observation & Unitization (4-fold classification vocabulary + selection pressure filter)
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class CognitiveLearningConfig:
    """
    Tunable hyper-parameters for Cognitive Learning Engine.
    Exposed as configuration to allow dynamic re-evaluations and external modifications (Axiom 3).
    """
    # Axiom 2 & 3: Density & Self-Modification Thresholds
    INITIAL_PATH_WEIGHT: float = 1.0
    REINFORCE_RATE: float = 0.5
    DECAY_RATE: float = 0.05
    MIN_PATH_WEIGHT: float = 0.01
    REEVAL_THRESHOLD_MULTIPLIER: float = 3.0  # Alert when path density > REEVAL_THRESHOLD_MULTIPLIER * avg_density

    # Axiom 4: Co-occurrence & Grounding
    COOCCURRENCE_WINDOW_SEC: float = 1.0
    GROUNDING_THRESHOLD: float = 3.0
    GROUNDING_DECAY: float = 0.02

    # Phase Transition Thresholds (Cumulative Energy / Delta Accumulation)
    ICE_TO_WATER_ENERGY: float = 10.0
    WATER_TO_GAS_ENERGY: float = 50.0
    ENERGY_DAMPING_FACTOR: float = 0.95

    # Dual Mode Parameters
    FORECAST_HORIZON: int = 3
    MAX_COMBINATION_DEPTH: int = 4
    MAX_COMBINATION_BEAM_WIDTH: int = 10

    # Holonic Selection Pressure
    STABLE_UNIT_MIN_REPETITIONS: int = 3


@dataclass
class TransitionEvent:
    """
    Axiom 1: Atomic Transition Unit.
    Value snapshot is NOT saved isolated; always recorded as a triplet (prev_ref, current_val, interval_velocity).
    """
    prev_state_ref: Optional[str]
    current_val: Any
    interval_sec: float
    velocity: float  # (current_val - prev_val) / interval if scalar, or delta norm / interval
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PathEdge:
    """
    Represents an edge in the transition network.
    """
    source_node: str
    target_node: str
    weight: float
    last_accessed: float
    last_decay_time: float = field(default_factory=time.time)
    co_occurred_labels: Dict[str, float] = field(default_factory=dict)  # label -> co-occurrence strength


class PhaseMode:
    ICE = "ICE"       # Low degree of freedom: Strongly fixed pattern response
    WATER = "WATER"   # Medium degree of freedom: Maintain existing path, locally rearrange
    GAS = "GAS"       # High degree of freedom: High surprise / prediction failure -> wide re-exploration


class CognitiveLearningEngine:
    """
    Core Cognitive Learning Engine embodying the 4 Axioms and Phase Transition dynamics.
    """

    def __init__(self, config: Optional[CognitiveLearningConfig] = None):
        self.config = config or CognitiveLearningConfig()

        # Network topology: node_id -> {target_node_id -> PathEdge}
        self.network: Dict[str, Dict[str, PathEdge]] = {}

        # Last recorded state node
        self.current_state_node: Optional[str] = None
        self.last_event_time: Optional[float] = None
        self.last_scalar_value: Optional[float] = None

        # Cumulative energy / delta for phase transition
        self.accumulated_energy: float = 0.0
        self.current_phase: str = PhaseMode.ICE

        # Axiom 3 self-modification alert count
        self.self_modification_alerts: List[Dict[str, Any]] = []

        # Holonic vocabulary tracking (4-fold logic table co-occurrences)
        # Truth-table classification: (T-T, T-F, F-T, F-F)
        self.holonic_matrix: Dict[Tuple[bool, bool], int] = {
            (True, True): 0,
            (True, False): 0,
            (False, True): 0,
            (False, False): 0,
        }
        self.stable_units: Set[str] = set()

    def record_transition(
        self,
        new_val: float,
        timestamp: Optional[float] = None,
        external_labels: Optional[Dict[str, float]] = None
    ) -> TransitionEvent:
        """
        Record a new observation value adhering to Axiom 1.
        Calculates interval and velocity, creates/reinforces path edges (Axiom 2),
        updates phase transitions, and triggers Axiom 3/4 if thresholds met.
        """
        now = timestamp if timestamp is not None else time.time()

        if self.last_event_time is None or self.current_state_node is None:
            # First observation: bootstrap node
            prev_ref = None
            interval = 0.1
            velocity = 0.0
            current_node = f"S_{round(new_val, 2)}"
        else:
            prev_ref = self.current_state_node
            interval = max(0.001, now - self.last_event_time)
            delta = new_val - (self.last_scalar_value if self.last_scalar_value is not None else new_val)
            velocity = delta / interval
            current_node = f"S_{round(new_val, 2)}"

        event = TransitionEvent(
            prev_state_ref=prev_ref,
            current_val=new_val,
            interval_sec=interval,
            velocity=velocity,
            timestamp=now
        )

        # Axiom 2: Reinforce path if transition occurred
        if prev_ref is not None:
            self._reinforce_or_create_path(prev_ref, current_node, now)
            # Energy accumulation based on velocity / delta
            energy_delta = abs(velocity) * interval
            self._update_energy_and_phase(energy_delta)

        # Apply temporal decay to all paths in network
        self._decay_paths(now)

        # Axiom 4: Symbol Grounding co-occurrence check
        if external_labels and prev_ref is not None:
            self._apply_symbol_grounding(prev_ref, current_node, external_labels)

        # Axiom 3: Check density threshold for self-modification trigger
        self._check_axiom3_self_modification(prev_ref, current_node)

        # Update holonic classification matrix
        self._update_holonic_matrix(event)

        # Advance state
        self.current_state_node = current_node
        self.last_event_time = now
        self.last_scalar_value = new_val

        return event

    def _reinforce_or_create_path(self, source: str, target: str, now: float):
        if source not in self.network:
            self.network[source] = {}

        if target in self.network[source]:
            edge = self.network[source][target]
            edge.weight += self.config.REINFORCE_RATE
            edge.last_accessed = now
            edge.last_decay_time = now
        else:
            self.network[source][target] = PathEdge(
                source_node=source,
                target_node=target,
                weight=self.config.INITIAL_PATH_WEIGHT,
                last_accessed=now,
                last_decay_time=now
            )

    def _decay_paths(self, now: float):
        for source in list(self.network.keys()):
            for target in list(self.network[source].keys()):
                edge = self.network[source][target]
                time_passed = now - edge.last_decay_time
                if time_passed > 0.01:
                    edge.weight -= self.config.DECAY_RATE * time_passed
                    # Decay grounded labels
                    for label in list(edge.co_occurred_labels.keys()):
                        edge.co_occurred_labels[label] -= self.config.GROUNDING_DECAY * time_passed
                        if edge.co_occurred_labels[label] <= 0:
                            del edge.co_occurred_labels[label]
                    edge.last_decay_time = now

                if edge.weight < self.config.MIN_PATH_WEIGHT:
                    del self.network[source][target]

            if not self.network[source]:
                del self.network[source]

    def _update_energy_and_phase(self, energy_delta: float):
        self.accumulated_energy = (self.accumulated_energy * self.config.ENERGY_DAMPING_FACTOR) + energy_delta

        if self.accumulated_energy >= self.config.WATER_TO_GAS_ENERGY:
            self.current_phase = PhaseMode.GAS
        elif self.accumulated_energy >= self.config.ICE_TO_WATER_ENERGY:
            self.current_phase = PhaseMode.WATER
        else:
            self.current_phase = PhaseMode.ICE

    def _apply_symbol_grounding(self, source: str, target: str, labels: Dict[str, float]):
        if source in self.network and target in self.network[source]:
            edge = self.network[source][target]
            for label, intensity in labels.items():
                current_val = edge.co_occurred_labels.get(label, 0.0)
                edge.co_occurred_labels[label] = current_val + intensity

    def _get_average_network_density(self) -> float:
        total_weight = 0.0
        edge_count = 0
        for source, targets in self.network.items():
            for target, edge in targets.items():
                total_weight += edge.weight
                edge_count += 1
        return total_weight / edge_count if edge_count > 0 else 1.0

    def _check_axiom3_self_modification(self, source: Optional[str], target: str):
        if source is None or source not in self.network or target not in self.network[source]:
            return

        edge_weight = self.network[source][target].weight
        avg_density = self._get_average_network_density()

        if edge_weight > avg_density * self.config.REEVAL_THRESHOLD_MULTIPLIER:
            alert = {
                "source": source,
                "target": target,
                "edge_weight": edge_weight,
                "avg_density": avg_density,
                "timestamp": time.time(),
                "message": f"Axiom 3 Triggered: Path {source}->{target} weight ({edge_weight:.2f}) exceeds {self.config.REEVAL_THRESHOLD_MULTIPLIER}x average density ({avg_density:.2f}). Rules need self-review."
            }
            self.self_modification_alerts.append(alert)

    def _update_holonic_matrix(self, event: TransitionEvent):
        """
        Holonic 4-fold classification:
        Evaluates binary condition of (is_positive_velocity, is_above_average_energy).
        """
        cond_velocity = event.velocity > 0
        cond_energy = self.accumulated_energy > (self.config.ICE_TO_WATER_ENERGY / 2.0)
        pair = (cond_velocity, cond_energy)

        self.holonic_matrix[pair] += 1

        # Selection pressure: unitize if count >= STABLE_UNIT_MIN_REPETITIONS
        unit_key = f"UNIT_{pair[0]}_{pair[1]}"
        if self.holonic_matrix[pair] >= self.config.STABLE_UNIT_MIN_REPETITIONS:
            self.stable_units.add(unit_key)

    def predict_forward(self, start_node: Optional[str] = None, steps: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        5.1 Forward Mode (Forecast):
        Extends existing transition paths into the future. Returns list of (predicted_node, path_weight).
        """
        curr = start_node or self.current_state_node
        if not curr or curr not in self.network:
            return []

        horizon = steps or self.config.FORECAST_HORIZON
        trajectory = []

        for _ in range(horizon):
            if curr not in self.network or not self.network[curr]:
                break
            # Pick strongest weight path
            best_target, best_edge = max(self.network[curr].items(), key=lambda item: item[1].weight)
            trajectory.append((best_target, best_edge.weight))
            curr = best_target

        return trajectory

    def search_reverse_abduction(self, target_node: str) -> List[List[str]]:
        """
        5.2 Reverse Mode (Abduction / Design):
        Backtraces from target node to discover potential causal predecessor pathways leading to target_node.
        """
        paths = []

        def dfs(current: str, path: List[str], depth: int):
            paths.append(list(path))
            if depth >= self.config.MAX_COMBINATION_DEPTH:
                return

            # Find predecessor nodes `source` where `source -> current` exists
            for source, targets in self.network.items():
                if current in targets and source not in path:
                    dfs(source, [source] + path, depth + 1)

        # Search backward starting from direct predecessors of target_node
        for source, targets in self.network.items():
            if target_node in targets:
                dfs(source, [source, target_node], 1)

        # Sort paths by accumulated weight descending
        def path_score(p: List[str]) -> float:
            score = 0.0
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                if u in self.network and v in self.network[u]:
                    score += self.network[u][v].weight
            return score

        paths.sort(key=path_score, reverse=True)
        return paths[:self.config.MAX_COMBINATION_BEAM_WIDTH]
