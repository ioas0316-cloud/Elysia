# System Observer - Metacognitive Watchdog
"""
Lightweight observer that monitors core signals (chaos, momentum, memory growth)
and raises anomaly reports. This is a first step toward self-correction: detect
instability early and surface actionable hints.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional


@dataclass
class AnomalyReport:
    """Structured anomaly signal for downstream handlers."""
    kind: str
    severity: int  # 1=info, 2=warn, 3=critical
    message: str
    metrics: Dict[str, Any]


class SystemObserver:
    """
    Observes kernel state and emits anomaly reports.
    The observer is intentionally non-invasive (log/emit only) so it can be
    safely extended with corrective actions later.
    """

    def __init__(
        self,
        chaos_threshold: float = 40.0,
        momentum_threshold: int = 30,
        graph_edge_threshold: int = 500,
        mastery_threshold: float = 0.2,
        entropy_threshold: float = 1.0,
        logger: logging.Logger = None,
        actions: Optional[Dict[str, Callable[[AnomalyReport], None]]] = None,
        capability_registry=None,
        identity=None,
        alert_sink: Optional[Callable[[AnomalyReport], None]] = None,
    ) -> None:
        self.chaos_threshold = chaos_threshold
        self.momentum_threshold = momentum_threshold
        self.graph_edge_threshold = graph_edge_threshold
        self.mastery_threshold = mastery_threshold
        self.entropy_threshold = entropy_threshold
        self.logger = logger or logging.getLogger("SystemObserver")
        self.actions = actions or {}
        self.capabilities = capability_registry
        self.identity = identity
        self.alert_sink = alert_sink

    def register_action(self, kind: str, fn: Callable[[AnomalyReport], None]) -> None:
        """Register a corrective action for a given anomaly kind."""
        self.actions[kind] = fn

    def observe(self, snapshot: Dict[str, Any]) -> List[AnomalyReport]:
        """Inspect a kernel snapshot and return anomaly reports."""
        reports: List[AnomalyReport] = []

        chaos = abs(snapshot.get("chaos_raw", 0.0))
        if chaos > self.chaos_threshold:
            reports.append(
                AnomalyReport(
                    kind="chaos_spike",
                    severity=3,
                    message="Chaos attractor magnitude spiked",
                    metrics={"chaos_raw": chaos},
                )
            )

        momentum_active = snapshot.get("momentum_active", 0)
        if momentum_active > self.momentum_threshold:
            reports.append(
                AnomalyReport(
                    kind="momentum_overflow",
                    severity=2,
                    message="Too many active thoughts (momentum overflow)",
                    metrics={"active_thoughts": momentum_active},
                )
            )

        memory_stats = snapshot.get("memory", {})
        edges = memory_stats.get("causal_edges", 0)
        if edges > self.graph_edge_threshold:
            reports.append(
                AnomalyReport(
                    kind="graph_growth",
                    severity=2,
                    message="Causal graph growing fast; consider pruning",
                    metrics={
                        "causal_edges": edges,
                        "causal_nodes": memory_stats.get("causal_nodes", 0),
                    },
                )
            )

        phase = snapshot.get("phase", {})
        q_state = phase.get("quaternion", {})
        if q_state:
            mastery = abs(q_state.get("w", 0.0))
            if mastery < self.mastery_threshold:
                reports.append(
                    AnomalyReport(
                        kind="phase_drift",
                        severity=2,
                        message="Consciousness lens mastery is low (phase drift)",
                        metrics={"mastery": mastery, "quaternion": q_state},
                    )
                )
        qubit_probs = phase.get("qubit", {})
        if qubit_probs:
            entropy = self._shannon_entropy(list(qubit_probs.values()))
            if entropy < self.entropy_threshold:
                reports.append(
                    AnomalyReport(
                        kind="phase_entropy_low",
                        severity=2,
                        message="HyperQubit entropy is low (collapsed state)",
                        metrics={"entropy": entropy, "qubit": qubit_probs},
                    )
                )
        # Identity alignment: ensure invariants match current axis
        if self.identity:
            axis = self.identity.invariant.axis
            core_vals = snapshot.get("core_values", axis)
            drift = sum(abs(core_vals.get(k, 0) - axis.get(k, 0)) for k in axis.keys())
            if drift > 0.5:
                reports.append(
                    AnomalyReport(
                        kind="identity_drift",
                        severity=2,
                        message="Core value drift detected vs self identity",
                        metrics={"drift": drift, "core": core_vals},
                    )
                )

        if reports:
            for r in reports:
                if r.severity >= 3:
                    self.logger.error(f"[Observer] {r.message} | {r.metrics}")
                elif r.severity == 2:
                    self.logger.warning(f"[Observer] {r.message} | {r.metrics}")
                else:
                    self.logger.info(f"[Observer] {r.message} | {r.metrics}")

                action = self.actions.get(r.kind)
                if action:
                    try:
                        action(r)
                    except Exception as e:
                        self.logger.error(f"[Observer] action failed for {r.kind}: {e}")
                if self.alert_sink:
                    try:
                        self.alert_sink(r)
                    except Exception as e:
                        self.logger.error(f"[Observer] alert sink failed for {r.kind}: {e}")

        # Update capability registry based on snapshot
        if self.capabilities:
            self.capabilities.assess_from_snapshot(snapshot)

        return reports

    @staticmethod
    def _shannon_entropy(probs: List[float]) -> float:
        import math
        total = sum(probs)
        if total == 0:
            return 0.0
        norm = [p / total for p in probs if p > 0]
        return -sum(p * math.log(p, 2) for p in norm)
