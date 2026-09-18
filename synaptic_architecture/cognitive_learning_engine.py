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
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum


class InputType(Enum):
    SCALAR = "SCALAR"
    CATEGORICAL = "CATEGORICAL"
    VECTOR = "VECTOR"
    VOLUME = "VOLUME"
    UNKNOWN = "UNKNOWN"


class InputDispatcher:
    """
    Classifies raw incoming input data type and structure before routing
    to the appropriate domain encoder.
    Subject to Axiom 3: rule classification metrics are tracked so classifier
    rules themselves can be flagged for re-evaluation when anomaly limits are breached.
    """

    def __init__(self):
        self.classification_counts: Dict[InputType, int] = {
            InputType.SCALAR: 0,
            InputType.CATEGORICAL: 0,
            InputType.VECTOR: 0,
            InputType.VOLUME: 0,
            InputType.UNKNOWN: 0,
        }

    def classify(self, data: Any) -> InputType:
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            itype = InputType.SCALAR
        elif isinstance(data, str):
            itype = InputType.CATEGORICAL
        elif isinstance(data, dict):
            itype = InputType.VOLUME
        elif isinstance(data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in data):
                itype = InputType.VECTOR
            elif all(isinstance(x, dict) for x in data):
                itype = InputType.VOLUME
            else:
                itype = InputType.CATEGORICAL
        else:
            itype = InputType.UNKNOWN

        self.classification_counts[itype] += 1
        return itype


class ScalarEncoder:
    """Encodes scalar signals into normalized state representation and velocity."""
    @staticmethod
    def encode(val: float, last_val: Optional[float], interval: float) -> Tuple[str, float]:
        node_id = f"S_{round(val, 2)}"
        if last_val is None:
            velocity = 0.0
        else:
            velocity = (val - last_val) / interval
        return node_id, velocity


class CategoricalEncoder:
    """Encodes categorical/symbolic signals into normalized state representation and transition delta."""
    @staticmethod
    def encode(val: str, last_val: Optional[Any], interval: float) -> Tuple[str, float]:
        node_id = f"SYM_{val}"
        if last_val is None or str(last_val) == node_id or str(last_val) == str(val):
            velocity = 0.0
        else:
            # Shift in symbolic state represents a category transition step velocity
            velocity = 1.0 / interval
        return node_id, velocity


class VectorEncoder:
    """Encodes vector signals (lists/tuples of numeric values) into normalized state representation and velocity (L2 norm delta)."""
    @staticmethod
    def encode(val: Union[List[float], Tuple[float, ...]], last_val: Optional[Any], interval: float) -> Tuple[str, float]:
        rounded = [round(float(x), 2) for x in val]
        node_id = f"VEC_{rounded}"
        if last_val is None or not isinstance(last_val, (list, tuple)) or len(last_val) != len(val):
            velocity = 0.0
        else:
            # L2 norm delta velocity
            diff_sq = sum((float(a) - float(b)) ** 2 for a, b in zip(val, last_val))
            velocity = math.sqrt(diff_sq) / max(0.001, interval)
        return node_id, velocity


class VolumeEncoder:
    """
    Axiom 1.1: Encodes volume signals (multi-element structure frames / key-value snapshots)
    into normalized state representation and element-wise velocities.
    """
    @staticmethod
    def encode(
        val: Dict[str, Any],
        last_val: Optional[Dict[str, Any]],
        interval: float
    ) -> Tuple[str, Dict[str, float], float]:
        sorted_items = sorted(val.items())
        repr_parts = []
        for k, v in sorted_items:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                repr_parts.append(f"{k}:{round(float(v), 2)}")
            else:
                repr_parts.append(f"{k}:{v}")
        node_id = f"VOL_{{{', '.join(repr_parts)}}}"

        element_velocities: Dict[str, float] = {}
        sq_sum = 0.0
        numeric_count = 0

        if last_val is not None and isinstance(last_val, dict):
            for k, v in val.items():
                if k in last_val and isinstance(v, (int, float)) and isinstance(last_val[k], (int, float)):
                    v_diff = (float(v) - float(last_val[k])) / max(0.001, interval)
                    element_velocities[k] = v_diff
                    sq_sum += v_diff ** 2
                    numeric_count += 1
                elif k in last_val and str(v) != str(last_val[k]):
                    # Categorical shift within volume element
                    element_velocities[k] = 1.0 / max(0.001, interval)
                    sq_sum += element_velocities[k] ** 2
                    numeric_count += 1
                else:
                    element_velocities[k] = 0.0
        else:
            for k, v in val.items():
                element_velocities[k] = 0.0

        overall_velocity = math.sqrt(sq_sum) if numeric_count > 0 else 0.0
        return node_id, element_velocities, overall_velocity


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
    STABLE_UNIT_RELATIVE_RATIO: float = 0.35  # Require at least 35% relative frequency among all observed units

    # Axiom 3 Dispatcher Self-Modification Threshold
    DISPATCHER_UNKNOWN_LIMIT: int = 3  # Flag dispatcher rules for re-evaluation after N unknown/anomalous classifications

    # Axiom 5 & Section 10: Temporal Phase Alignment & Memory Tier Parameters
    PHASE_ALIGNMENT_TAU: float = 0.5
    PHASE_ALIGNMENT_THRESHOLD: float = 0.6
    PROMOTION_CACHE_TO_RAM_COUNT: int = 3
    PROMOTION_RAM_TO_SSD_WEIGHT: float = 5.0
    DEMOTION_DISCREPANCY_LIMIT: int = 3


@dataclass
class TransitionEvent:
    """
    Axiom 1 & 1.1: Atomic Transition Unit.
    Value snapshot is NOT saved isolated; always recorded as a triplet (prev_ref, current_val, interval_velocity).
    For volume structure frames, element_velocities captures component-wise rate of change.
    """
    prev_state_ref: Optional[str]
    current_val: Any
    interval_sec: float
    velocity: float  # (current_val - prev_val) / interval if scalar, or delta norm / interval
    element_velocities: Optional[Dict[str, float]] = None
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


@dataclass
class CompositeNode:
    """
    Axiom 4.1, 4.3 & 7.4: Independent 3rd node (Composite / Exclusive Node)
    generated upon binding / co-occurrence / bootstrap discrepancy.
    Holds references to component sources A and B, but maintains its own
    multiplicative binding strength (Axiom 4.2) and phase alignment.
    """
    id: str
    source_a: str
    source_b: str
    binding_strength: float  # Multiplicative: density_a * density_b * phase_alignment (Axiom 4.2)
    phase_alignment: float
    created_at: float
    last_activated: float
    activation_count: int = 1
    labels: Dict[str, float] = field(default_factory=dict)
    is_exclusive: bool = False  # True for Divergence/Exclusive nodes (Axiom 4.3 T-F, F-T)


class MemoryTier(Enum):
    """
    Section 10: Logical memory tier mapping.
    """
    TRANSIENT_CACHE = "TRANSIENT_CACHE"  # Axiom 5 (real-time phase alignment / neural oscillations)
    WORKING_MEMORY = "WORKING_MEMORY"    # Axiom 2 (path density / Hebbian network)
    PERSISTENT_SSD = "PERSISTENT_SSD"    # Axiom 1 seed priors & consolidated invariants


@dataclass
class TransientBindingUnit:
    """
    Axiom 5: Transient Binding Unit formed via real-time temporal phase alignment
    (neural oscillatory synchrony). Lives in Transient Cache memory tier.
    """
    id: str
    transition_a: str
    transition_b: str
    phase_alignment: float  # [0.0, 1.0] based on interval/rhythm similarity
    created_at: float
    last_synced: float
    sync_count: int = 1
    tier: MemoryTier = MemoryTier.TRANSIENT_CACHE


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
        self.last_interval_sec: Optional[float] = None
        self.last_transition_id: Optional[str] = None
        self.last_scalar_value: Optional[float] = None
        self.last_vector_value: Optional[List[float]] = None
        self.last_volume_value: Optional[Dict[str, Any]] = None

        # Section 10 Memory Hierarchy Tiers
        self.transient_cache: Dict[str, TransientBindingUnit] = {}
        self.persistent_seeds: Dict[str, Dict[str, Any]] = {}

        # Cumulative energy / delta for phase transition
        self.accumulated_energy: float = 0.0
        self.current_phase: str = PhaseMode.ICE

        # Input Dispatcher instance (tracks classification counts for Axiom 3 rule self-review)
        self.dispatcher = InputDispatcher()

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

        # Axiom 4.1 & 7.4: Composite nodes registry
        self.composite_nodes: Dict[str, CompositeNode] = {}

    def record_transition(
        self,
        new_val: float,
        timestamp: Optional[float] = None,
        external_labels: Optional[Dict[str, float]] = None
    ) -> TransitionEvent:
        """
        Record a new observation value adhering to Axiom 1.
        Delegates to dispatch_and_record for normalized multi-modal routing.
        """
        event, _ = self.dispatch_and_record(new_val, timestamp=timestamp, external_labels=external_labels)
        return event

    def dispatch_and_record(
        self,
        raw_data: Any,
        timestamp: Optional[float] = None,
        external_labels: Optional[Dict[str, float]] = None
    ) -> Tuple[TransitionEvent, InputType]:
        """
        Dispatches raw input through the InputDispatcher to determine data type,
        encodes it via appropriate modal encoder, and records normalized transition.
        Subject to Axiom 3 self-modification trigger on dispatcher classification anomalies.
        """
        now = timestamp if timestamp is not None else time.time()
        input_type = self.dispatcher.classify(raw_data)

        interval = 0.1
        if self.last_event_time is not None:
            interval = max(0.001, now - self.last_event_time)

        elem_vels = None
        if input_type == InputType.SCALAR:
            val = float(raw_data)
            current_node, velocity = ScalarEncoder.encode(val, self.last_scalar_value, interval)
            self.last_scalar_value = val
        elif input_type == InputType.CATEGORICAL:
            val = str(raw_data)
            current_node, velocity = CategoricalEncoder.encode(val, self.current_state_node, interval)
        elif input_type == InputType.VECTOR:
            vec_val = [float(x) for x in raw_data]
            current_node, velocity = VectorEncoder.encode(vec_val, self.last_vector_value, interval)
            self.last_vector_value = vec_val
        elif input_type == InputType.VOLUME:
            vol_val = dict(raw_data)
            current_node, elem_vels, velocity = VolumeEncoder.encode(vol_val, self.last_volume_value, interval)
            self.last_volume_value = vol_val
        else:
            # Fallback for unknown / unclassifiable custom objects
            val = str(raw_data)
            current_node = f"GEN_{val}"
            velocity = 0.0

        prev_ref = self.current_state_node
        event = TransitionEvent(
            prev_state_ref=prev_ref,
            current_val=raw_data,
            interval_sec=interval,
            velocity=velocity,
            element_velocities=elem_vels,
            timestamp=now,
            metadata={"input_type": input_type.value}
        )

        # Reinforce & decay
        if prev_ref is not None:
            self._reinforce_or_create_path(prev_ref, current_node, now)
            energy_delta = abs(velocity) * interval
            self._update_energy_and_phase(energy_delta)

        self._decay_paths(now)

        if external_labels and prev_ref is not None:
            self._apply_symbol_grounding(prev_ref, current_node, external_labels)

        self._check_axiom3_self_modification(prev_ref, current_node)
        self._check_axiom3_dispatcher_modification(input_type)
        self._update_holonic_matrix(event)

        self.current_state_node = current_node
        self.last_event_time = now

        # Axiom 5 & Section 10: Temporal Phase Alignment & Memory Tier Management
        self._update_phase_alignment_and_cache(event, current_node)
        self._evaluate_memory_tier_promotions_and_demotions(now)

        self.last_interval_sec = interval
        if prev_ref is not None:
            self.last_transition_id = f"{prev_ref}->{current_node}"
        if isinstance(raw_data, (int, float)):
            self.last_scalar_value = float(raw_data)

        return event, input_type

    def _update_phase_alignment_and_cache(self, event: TransitionEvent, current_node: str):
        """
        Axiom 5: Measures real-time temporal phase alignment (rhythm similarity)
        between current transition and preceding transition.
        Forms transient binding units in the Transient Cache memory tier if aligned.
        """
        if event.prev_state_ref is None or self.last_transition_id is None or self.last_interval_sec is None:
            return

        current_transition_id = f"{event.prev_state_ref}->{current_node}"
        interval_diff = abs(event.interval_sec - self.last_interval_sec)
        phase_alignment = math.exp(-interval_diff / max(0.001, self.config.PHASE_ALIGNMENT_TAU))

        if phase_alignment >= self.config.PHASE_ALIGNMENT_THRESHOLD:
            tb_id = f"TB_{self.last_transition_id}_{current_transition_id}"
            now = event.timestamp
            if tb_id in self.transient_cache:
                unit = self.transient_cache[tb_id]
                unit.phase_alignment = phase_alignment
                unit.last_synced = now
                unit.sync_count += 1
            else:
                self.transient_cache[tb_id] = TransientBindingUnit(
                    id=tb_id,
                    transition_a=self.last_transition_id,
                    transition_b=current_transition_id,
                    phase_alignment=phase_alignment,
                    created_at=now,
                    last_synced=now,
                    sync_count=1
                )

    def _evaluate_memory_tier_promotions_and_demotions(self, now: float):
        """
        Section 10.3: Logical memory tier promotion and demotion.
        1. Transient Cache -> Working Memory (RAM) promotion upon repeated strong phase alignment.
        2. Working Memory (RAM) -> Persistent SSD promotion upon high density stability.
        """
        # 1. Transient Cache -> Working Memory promotion
        for tb_id, unit in list(self.transient_cache.items()):
            if unit.sync_count >= self.config.PROMOTION_CACHE_TO_RAM_COUNT:
                # Promote to Working Memory: create/reinforce composite node
                self.create_or_update_composite_node(
                    source_a=unit.transition_a,
                    source_b=unit.transition_b,
                    density_a=1.0 * unit.sync_count,
                    density_b=1.0 * unit.sync_count,
                    phase_alignment=unit.phase_alignment,
                    is_exclusive=False,
                    timestamp=now
                )
                unit.tier = MemoryTier.WORKING_MEMORY

        # 2. Working Memory -> Persistent SSD promotion
        for source, targets in self.network.items():
            for target, edge in targets.items():
                if edge.weight >= self.config.PROMOTION_RAM_TO_SSD_WEIGHT:
                    seed_key = f"SEED_PATH_{source}->{target}"
                    if seed_key not in self.persistent_seeds:
                        self.persistent_seeds[seed_key] = {
                            "type": "PATH_SEED",
                            "source": source,
                            "target": target,
                            "weight": edge.weight,
                            "promoted_at": now,
                            "discrepancy_count": 0,
                            "tier": MemoryTier.PERSISTENT_SSD.value
                        }

        for cid, cnode in self.composite_nodes.items():
            if cnode.binding_strength >= self.config.PROMOTION_RAM_TO_SSD_WEIGHT:
                seed_key = f"SEED_COMP_{cid}"
                if seed_key not in self.persistent_seeds:
                    self.persistent_seeds[seed_key] = {
                        "type": "COMPOSITE_SEED",
                        "composite_id": cid,
                        "binding_strength": cnode.binding_strength,
                        "promoted_at": now,
                        "discrepancy_count": 0,
                        "tier": MemoryTier.PERSISTENT_SSD.value
                    }

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

    def create_or_update_composite_node(
        self,
        source_a: str,
        source_b: str,
        density_a: float,
        density_b: float,
        phase_alignment: float = 1.0,
        is_exclusive: bool = False,
        timestamp: Optional[float] = None
    ) -> CompositeNode:
        """
        Axiom 4.1, 4.2, 4.3: Creates or updates an independent 3rd composite node.
        Multiplicative binding strength = density_a * density_b * phase_alignment.
        """
        now = timestamp if timestamp is not None else time.time()
        node_prefix = "EXCL" if is_exclusive else "COMP"
        cid = f"{node_prefix}_{source_a}_{source_b}"

        # Multiplicative calculation (Axiom 4.2)
        binding_strength = max(0.0, density_a) * max(0.0, density_b) * max(0.0, phase_alignment)

        if cid in self.composite_nodes:
            cnode = self.composite_nodes[cid]
            cnode.binding_strength = binding_strength
            cnode.phase_alignment = phase_alignment
            cnode.last_activated = now
            cnode.activation_count += 1
        else:
            cnode = CompositeNode(
                id=cid,
                source_a=source_a,
                source_b=source_b,
                binding_strength=binding_strength,
                phase_alignment=phase_alignment,
                created_at=now,
                last_activated=now,
                activation_count=1,
                is_exclusive=is_exclusive
            )
            self.composite_nodes[cid] = cnode

        return cnode

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

        # Decay composite nodes
        for cid, cnode in list(self.composite_nodes.items()):
            time_passed = now - cnode.last_activated
            if time_passed > 0.1:
                cnode.binding_strength -= self.config.GROUNDING_DECAY * time_passed
                if cnode.binding_strength <= self.config.MIN_PATH_WEIGHT:
                    del self.composite_nodes[cid]

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
            path_id = f"{source}->{target}"
            now = self.last_event_time or time.time()

            for label, intensity in labels.items():
                current_val = edge.co_occurred_labels.get(label, 0.0)
                new_intensity = current_val + intensity
                edge.co_occurred_labels[label] = new_intensity

                # Axiom 4.1 & 4.2: Immediately generate or update Composite Node for grounding
                self.create_or_update_composite_node(
                    source_a=path_id,
                    source_b=f"LABEL_{label}",
                    density_a=edge.weight,
                    density_b=new_intensity,
                    phase_alignment=1.0,  # Co-occurring within window
                    is_exclusive=False,
                    timestamp=now
                )

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

    def _check_axiom3_dispatcher_modification(self, classified_type: InputType):
        """
        Axiom 3: Check if dispatcher classification rules are repeatedly producing
        unknown or anomalous classifications, signaling that the classifier rules themselves
        require self-review / restructuring.
        """
        unknown_count = self.dispatcher.classification_counts[InputType.UNKNOWN]
        if classified_type == InputType.UNKNOWN and unknown_count >= self.config.DISPATCHER_UNKNOWN_LIMIT:
            alert = {
                "type": "DISPATCHER_RULE_REEVALUATION",
                "unknown_count": unknown_count,
                "timestamp": time.time(),
                "message": f"Axiom 3 Triggered (Dispatcher): Unclassified/Anomalous inputs count ({unknown_count}) reached limit ({self.config.DISPATCHER_UNKNOWN_LIMIT}). Dispatcher classification rules require self-review & restructuring."
            }
            # Only record alert once per limit threshold breach
            if not any(a.get("type") == "DISPATCHER_RULE_REEVALUATION" and a.get("unknown_count") == unknown_count for a in self.self_modification_alerts):
                self.self_modification_alerts.append(alert)

    def _update_holonic_matrix(self, event: TransitionEvent):
        """
        Holonic 4-fold classification (6.3 & 4.3):
        Evaluates binary condition of (is_positive_velocity, is_above_average_energy).
        Selection pressure: Unitize ONLY if count >= STABLE_UNIT_MIN_REPETITIONS AND
        relative frequency ratio >= STABLE_UNIT_RELATIVE_RATIO.

        Axiom 4.3: Branching into Convergence vs Divergence:
        - Convergence (T-T, F-F): Promoted to Composite Node (is_exclusive=False)
        - Divergence (T-F, F-T): Promoted to Exclusive Node (is_exclusive=True)
        """
        cond_velocity = event.velocity > 0
        cond_energy = self.accumulated_energy > (self.config.ICE_TO_WATER_ENERGY / 2.0)
        pair = (cond_velocity, cond_energy)

        self.holonic_matrix[pair] += 1
        total_observations = sum(self.holonic_matrix.values())
        now = event.timestamp

        # Evaluate all 4 pairs with relative selection pressure
        self.stable_units.clear()
        for p, count in self.holonic_matrix.items():
            rel_ratio = count / total_observations if total_observations > 0 else 0.0
            if count >= self.config.STABLE_UNIT_MIN_REPETITIONS and rel_ratio >= self.config.STABLE_UNIT_RELATIVE_RATIO:
                unit_key = f"UNIT_{p[0]}_{p[1]}"
                self.stable_units.add(unit_key)

                # Axiom 4.3: Convergence vs Divergence Branching
                is_convergence = (p[0] == p[1])  # T-T or F-F
                self.create_or_update_composite_node(
                    source_a=f"COND_VEL_{p[0]}",
                    source_b=f"COND_ENG_{p[1]}",
                    density_a=float(count),
                    density_b=rel_ratio * 10.0,
                    phase_alignment=1.0,
                    is_exclusive=not is_convergence,
                    timestamp=now
                )

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

    def _get_node_density(self, node: str) -> float:
        total = 0.0
        count = 0
        if node in self.network:
            for _, edge in self.network[node].items():
                total += edge.weight
                count += 1
        return total / count if count > 0 else 1.0

    def load_seed_priors(self, seed_data: Dict[str, Any]):
        """
        Section 7.1 & 7.3: Loads imported seed priors ('Parent' role initial boundary cuts).
        """
        for seed_id, seed_content in seed_data.items():
            self.persistent_seeds[f"SEED_PRIOR_{seed_id}"] = {
                "type": "EXTERNAL_SEED_PRIOR",
                "content": seed_content,
                "discrepancy_count": 0,
                "tier": MemoryTier.PERSISTENT_SSD.value
            }

    def process_dual_source_observation(
        self,
        raw_data: Any,
        seed_prior_id: str,
        expected_next_val: Any,
        timestamp: Optional[float] = None,
        external_labels: Optional[Dict[str, float]] = None
    ) -> Tuple[TransitionEvent, Optional[CompositeNode]]:
        """
        Section 7.3 & 7.4 & Roadmap Step 5: Process dual-source signals (Seed Priors vs Direct Delta Experience).
        If direct experience mismatches seed prior expectations:
        Creates/updates an independent Observation Node (CompositeNode) tracking the discrepancy delta (7.4).
        If discrepancy repeats consistently in same direction, triggers seed re-alignment.
        """
        event, _ = self.dispatch_and_record(raw_data, timestamp=timestamp, external_labels=external_labels)
        now = event.timestamp

        discrepancy_delta = 0.0
        if isinstance(raw_data, (int, float)) and isinstance(expected_next_val, (int, float)):
            discrepancy_delta = abs(float(raw_data) - float(expected_next_val))
        elif str(raw_data) != str(expected_next_val):
            discrepancy_delta = 1.0

        obs_node = None
        if discrepancy_delta > 0.1:
            seed_key = f"SEED_PRIOR_{seed_prior_id}"
            if seed_key in self.persistent_seeds:
                self.persistent_seeds[seed_key]["discrepancy_count"] += 1

            # 7.4: Create/update Observation Node
            obs_id = f"OBS_DISCREPANCY_{seed_prior_id}_{event.current_val}"
            obs_node = self.create_or_update_composite_node(
                source_a=seed_key,
                source_b=f"EXP_{event.current_val}",
                density_a=float(self.persistent_seeds.get(seed_key, {}).get("discrepancy_count", 1)),
                density_b=discrepancy_delta,
                phase_alignment=1.0,
                is_exclusive=False,
                timestamp=now
            )

            # Re-alignment trigger when discrepancy persists
            if obs_node.activation_count >= self.config.DEMOTION_DISCREPANCY_LIMIT:
                alert = {
                    "type": "SEED_REALIGNMENT_TRIGGERED",
                    "seed_id": seed_prior_id,
                    "discrepancy_count": obs_node.activation_count,
                    "new_experienced_value": raw_data,
                    "timestamp": now,
                    "message": f"Section 7.4 Re-alignment: Seed Prior '{seed_prior_id}' updated to match consistent experience delta ({raw_data})."
                }
                if not any(a.get("type") == "SEED_REALIGNMENT_TRIGGERED" and a.get("seed_id") == seed_prior_id and a.get("discrepancy_count") == obs_node.activation_count for a in self.self_modification_alerts):
                    self.self_modification_alerts.append(alert)
                if seed_key in self.persistent_seeds:
                    self.persistent_seeds[seed_key]["realigned_value"] = raw_data
                    self.persistent_seeds[seed_key]["realigned_at"] = now

        return event, obs_node

    def search_reverse_abduction(self, target_node: str) -> Dict[str, Any]:
        """
        5.2 Reverse Mode (Abduction / Design & Roadmap Step 4):
        1. Backtraces existing historical paths from target_node.
        2. Synthesizes novel recombined pathways by connecting previously unlinked nodes
           that share common grounded labels (shared principles/semantic grounding).
        3. Generates unseen combinatorial design hypotheses: attempts new unlinked edges
           combining separate component fragments to satisfy the goal.
        """
        direct_paths = []

        def dfs(current: str, path: List[str], depth: int):
            direct_paths.append(list(path))
            if depth >= self.config.MAX_COMBINATION_DEPTH:
                return

            for source, targets in self.network.items():
                if current in targets and source not in path:
                    dfs(source, [source] + path, depth + 1)

        for source, targets in self.network.items():
            if target_node in targets:
                dfs(source, [source, target_node], 1)

        def path_score(p: List[str]) -> float:
            score = 0.0
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                if u in self.network and v in self.network[u]:
                    score += self.network[u][v].weight
            return score

        direct_paths.sort(key=path_score, reverse=True)
        top_direct = direct_paths[:self.config.MAX_COMBINATION_BEAM_WIDTH]

        # Novel Recombinations: find nodes sharing co-occurred grounded labels
        target_predecessor_labels: Set[str] = set()
        for source, targets in self.network.items():
            if target_node in targets:
                edge = targets[target_node]
                target_predecessor_labels.update(edge.co_occurred_labels.keys())

        novel_recombinations = []
        if target_predecessor_labels:
            for s1, targets1 in self.network.items():
                for t1, edge1 in targets1.items():
                    if t1 == target_node:
                        continue  # skip already direct transitions
                    shared = set(edge1.co_occurred_labels.keys()).intersection(target_predecessor_labels)
                    if shared:
                        novel_recombinations.append({
                            "novel_pathway": [s1, t1, f"[Bridge via shared principle: {list(shared)}]", target_node],
                            "shared_grounded_principles": list(shared),
                            "origin_edge": f"{s1}->{t1}"
                        })

        # Combinatorial Design Synthesis (Roadmap Step 4: Unseen edge combinations)
        combinatorial_hypotheses = []
        all_nodes = list(self.network.keys())
        for n1 in all_nodes:
            if n1 == target_node:
                continue
            for n2 in all_nodes:
                if n2 == n1 or (n1 in self.network and n2 in self.network[n1]):
                    continue
                if n2 == target_node or (n2 in self.network and target_node in self.network[n2]):
                    n1_labels = set()
                    if n1 in self.network:
                        for _, edge in self.network[n1].items():
                            n1_labels.update(edge.co_occurred_labels.keys())
                    n2_labels = set()
                    if n2 in self.network and target_node in self.network[n2]:
                        n2_labels.update(self.network[n2][target_node].co_occurred_labels.keys())

                    shared_principles = list(n1_labels.intersection(n2_labels))
                    design_score = (len(shared_principles) + 1.0) * (self._get_node_density(n1) + self._get_node_density(n2))
                    combinatorial_hypotheses.append({
                        "hypothetical_edge": f"{n1} --[UNSEEN COMBINATION]--> {n2}",
                        "source": n1,
                        "bridge_target": n2,
                        "goal_target": target_node,
                        "shared_principles": shared_principles,
                        "estimated_design_score": design_score,
                        "full_synthetic_path": [n1, f"[Hypothetical edge -> {n2}]", target_node]
                    })

        combinatorial_hypotheses.sort(key=lambda x: x["estimated_design_score"], reverse=True)

        return {
            "historical_retrace_paths": top_direct,
            "novel_recombined_pathways": novel_recombinations,
            "combinatorial_design_hypotheses": combinatorial_hypotheses[:self.config.MAX_COMBINATION_BEAM_WIDTH]
        }
