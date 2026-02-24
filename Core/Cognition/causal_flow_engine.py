"""
Causal Flow Engine: The Living Process
======================================
Core.Cognition.causal_flow_engine

"Cause does not push Effect; Cause resonates, and Effect emerges."

This engine orchestrates the Holographic Causal Cycle:
1. Ignition (Input -> Intent Wave)
2. Resonance (Wave -> Manifold Interference)
3. Collapse (Interference -> Action/Result)
"""

from Core.System.holographic_memory import HolographicMemory
from Core.Monad.rotor_trajectory import RotorTrajectory
from Core.System.rotor import DoubleHelixEngine, RotorConfig
import numpy as np
from dataclasses import dataclass

from Core.Cognition.causal_admissibility_gate import (
    CausalAdmissibilityGate,
    CausalSignature,
    TransitionRecord,
)
from threading import Event, Thread
from queue import Queue, Empty
from typing import Optional
import time

@dataclass
class MerkabaParams:
    """
    [SPACETIME CONTROL]
    Parameters that modulate the Causal Flow.
    """
    rotor_rpm: float = 1.0      # Time Dilation (Simulation Speed)
    focus_depth: float = 1.0    # Spatial Compression (Attention Radius)
    axis_tilt: float = 0.0      # Perspective Shift (Phase Modulation)

class CausalFlowEngine:
    def __init__(self, memory: HolographicMemory):
        self.memory = memory
        self.current_state = "IDLE"
        self.merkaba = MerkabaParams()
        self.trajectory = RotorTrajectory()
        
        # [NEW] Double Helix Rotor Engine
        cfg = RotorConfig(rpm=120.0, idle_rpm=60.0)
        self.double_helix = DoubleHelixEngine("CausalFlow", cfg)

        self.gate = CausalAdmissibilityGate()
        self._quarantine_queue: Queue = Queue()
        self._recovery_thread: Optional[Thread] = None
        self._recovery_stop = Event()
        self._recovery_policy = {
            "missing_cause": "await_cause_enrichment",
            "phase_incoherent": "phase_realign",
            "energy_over_budget": "throttle_and_retry",
            "will_misaligned": "realign_intent",
            "trinary_unstable": "rebalance_trinary",
        }

    def adjust_merkaba(self, rpm: float = None, focus: float = None, tilt: float = None):
        """
        [CONTROL] Dynamic adjustment of the spacetime engine.
        """
        if rpm is not None: self.merkaba.rotor_rpm = rpm
        if focus is not None: self.merkaba.focus_depth = focus
        if tilt is not None: self.merkaba.axis_tilt = tilt

    def ignite(self, intent_seed: str, intensity: float = 1.0) -> dict:
        """
        [STEP 1] Ignition: Converts raw intent into a Wave Pulse.
        Intensity is modulated by the Merkaba's Focus.
        """
        self.current_state = "IGNITED"

        # [CONTROL] Focus compresses the wave, increasing local intensity
        modulated_intensity = intensity * self.merkaba.focus_depth

        # [NEW] Wake the Double Helix
        self.double_helix.modulate(intensity)

        return {
            "seed": intent_seed,
            "intensity": modulated_intensity,
            "phase": "RISING",
            "axis_tilt": self.merkaba.axis_tilt
        }

    def flow(self, ignition_packet: dict) -> dict:
        """
        [STEP 2] Resonance: The Wave travels through the Memory Manifold.
        """
        self.current_state = "RESONATING"
        seed = ignition_packet["seed"]
        intensity = ignition_packet["intensity"]
        tilt = ignition_packet.get("axis_tilt", 0.0)

        # [CONTROL] Rotor RPM determines how "far" the resonance spreads (Simulation)
        # High RPM = Fast Search (Shallow), Low RPM = Deep Resonance
        # Here we simulate it by modulating the threshold or 'noise' floor.

        # Check Resonance with existing memories
        # We query the memory manifold.
        # Note: In a real holographic system, we don't query *specific* keys.
        # We shine the light and see what *image* forms.
        # Here, we simulate that by checking resonance amplitude.

        # For prototype: Does the memory recognize this seed?
        # We treat the seed itself as a query.

        # [SOUL LAYER] Double Helix spins to find resonance
        # The Interference Snapshot represents the 'Structural Mirroring'
        self.double_helix.update(0.1) # Simulate a physics step (dt=0.1)
        interference = self.double_helix.get_interference_snapshot()
        
        # Resonance is modulated by structural interference
        (concept, base_amplitude, phase_shift) = self.memory.resonate(seed)
        
        # [NEW] The 'Lightning Path' logic: Interference amplifies or dampens resonance
        # Interference energy (cosine of phase diff) acts as a structural gate.
        amplitude = base_amplitude * (0.5 + 0.5 * interference)

        # [TRAJECTORY] Record the path
        self.trajectory.record(
            angle=self.double_helix.afferent.current_angle,
            resonance=amplitude,
            state="FLOWING"
        )

        # Determine Flow State based on Resonance
        # Semantic lock-in prevents rotor interference from downgrading clearly learned concepts.
        semantic_harmony = concept == seed and base_amplitude > 0.85

        flow_type = "UNKNOWN"
        if semantic_harmony or amplitude > 0.8:
            flow_type = "HARMONY" # Known, strong memory
        elif amplitude > 0.3:
            flow_type = "ECHO"    # Faint memory
        else:
            flow_type = "DISSONANCE" # New or conflicting

        return {
            "seed": seed,
            "flow_type": flow_type,
            "amplitude": amplitude,
            "phase_shift": phase_shift,
            "interference": interference
        }


    def evaluate_transition(
        self,
        *,
        from_state: str,
        to_state: str,
        cause_id: str,
        intent_vector: list[float],
        result_vector: list[float],
        phase_delta: float,
        energy_cost: float,
        trinary_state: dict,
        resonance_score: float = 0.0,
    ):
        """Evaluate transition admissibility via the Causal Admissibility Gate."""
        signature = CausalSignature(
            cause_id=cause_id,
            intent_vector=intent_vector,
            result_vector=result_vector,
            phase_delta=phase_delta,
            energy_cost=energy_cost,
            trinary_state=trinary_state,
        )
        return self.gate.evaluate(
            from_state=from_state,
            to_state=to_state,
            signature=signature,
            resonance_score=resonance_score,
        )


    def _default_trinary_state(self, flow_type: str, amplitude: float) -> dict:
        """Infer a conservative trinary distribution from resonance flow."""
        if flow_type == "HARMONY":
            return {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        if flow_type == "ECHO":
            return {"negative": 0.2, "neutral": 0.5, "positive": 0.3}
        if flow_type == "DISSONANCE":
            # Keep neutral floor to preserve observer slot under conflict.
            return {"negative": min(0.7, 0.4 + (1.0 - amplitude) * 0.3), "neutral": 0.2, "positive": 0.1}
        return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}

    def collapse_guarded(
        self,
        resonance_packet: dict,
        *,
        cause_id: str,
        intent_vector: list[float],
        result_vector: list[float],
        phase_delta: float,
        energy_cost: float,
        from_state: str = "RESONATING",
        to_state: str = "COLLAPSED",
    ) -> str:
        """Collapse only when transition is admissible; otherwise quarantine."""
        flow_type = resonance_packet.get("flow_type", "UNKNOWN")
        amplitude = float(resonance_packet.get("amplitude", 0.0))
        trinary_state = self._default_trinary_state(flow_type, amplitude)

        record = self.evaluate_transition(
            from_state=from_state,
            to_state=to_state,
            cause_id=cause_id,
            intent_vector=intent_vector,
            result_vector=result_vector,
            phase_delta=phase_delta,
            energy_cost=energy_cost,
            trinary_state=trinary_state,
            resonance_score=amplitude,
        )

        if not record.admissible:
            reasons = ",".join(record.rejection_reasons)
            seed = resonance_packet.get("seed", "unknown")
            return f"[QUARANTINE] Transition blocked for '{seed}' ({reasons})"

        return self.collapse(resonance_packet)

    def start_quarantine_recovery_loop(self, interval_sec: float = 0.1) -> None:
        """Start a lightweight worker that drains gate quarantine without blocking runtime."""
        if self._recovery_thread and self._recovery_thread.is_alive():
            return

        self._recovery_stop.clear()

        def _worker():
            while not self._recovery_stop.is_set():
                self._drain_quarantine_once()
                time.sleep(interval_sec)

        self._recovery_thread = Thread(target=_worker, daemon=True)
        self._recovery_thread.start()

    def stop_quarantine_recovery_loop(self, timeout: float = 1.0) -> None:
        """Stop the quarantine recovery worker."""
        self._recovery_stop.set()
        if self._recovery_thread:
            self._recovery_thread.join(timeout=timeout)

    def _drain_quarantine_once(self) -> int:
        """Move newly blocked transitions into a worker queue for external recovery handlers."""
        drained = self.gate.drain_quarantine()
        for record in drained:
            self._quarantine_queue.put(record)
        return len(drained)

    def pull_quarantined(self, max_items: int = 32) -> list:
        """Non-blocking fetch for recovery supervisors."""
        pulled = []
        for _ in range(max_items):
            try:
                pulled.append(self._quarantine_queue.get_nowait())
            except Empty:
                break
        return pulled

    def _recovery_action_for(self, record: TransitionRecord) -> str:
        """Derive primary recovery action from rejection reasons."""
        for reason in record.rejection_reasons:
            if reason in self._recovery_policy:
                return self._recovery_policy[reason]
        return "manual_review"

    def _retry_transition(self, record: TransitionRecord, *, phase_delta: float, energy_cost: float, result_vector: list[float], trinary_state: dict) -> bool:
        """Re-evaluate a quarantined transition with recovery-adjusted parameters."""
        retry_record = self.evaluate_transition(
            from_state=record.from_state,
            to_state=record.to_state,
            cause_id=record.signature.cause_id or "",
            intent_vector=list(record.signature.intent_vector),
            result_vector=result_vector,
            phase_delta=phase_delta,
            energy_cost=energy_cost,
            trinary_state=trinary_state,
            resonance_score=record.resonance_score,
        )
        return retry_record.admissible

    def _execute_recovery_action(self, record: TransitionRecord, action: str, retry_budget: int = 2) -> tuple[str, bool]:
        """Execute deterministic recovery action and report status."""
        signature = record.signature

        if action == "await_cause_enrichment":
            return ("deferred", False)

        if action == "manual_review":
            return ("manual_review", False)

        phase_delta = float(signature.phase_delta)
        energy_cost = float(signature.energy_cost)
        result_vector = list(signature.result_vector)
        trinary_state = dict(signature.trinary_state)

        for _ in range(max(1, retry_budget)):
            if action == "phase_realign":
                phase_delta = 0.0
            elif action == "throttle_and_retry":
                energy_cost = energy_cost * 0.5
            elif action == "realign_intent":
                result_vector = list(signature.intent_vector)
            elif action == "rebalance_trinary":
                neutral_floor = self.gate.thresholds.min_neutral_ratio
                trinary_state = {
                    "negative": max(0.0, float(trinary_state.get("negative", 0.0))),
                    "neutral": max(neutral_floor, float(trinary_state.get("neutral", 0.0))),
                    "positive": max(0.0, float(trinary_state.get("positive", 0.0))),
                }
                total = trinary_state["negative"] + trinary_state["neutral"] + trinary_state["positive"]
                if total > 0.0:
                    trinary_state = {k: v / total for k, v in trinary_state.items()}

            if self._retry_transition(
                record,
                phase_delta=phase_delta,
                energy_cost=energy_cost,
                result_vector=result_vector,
                trinary_state=trinary_state,
            ):
                return ("recovered", True)

        return ("retry_failed", False)

    def process_quarantine_batch(self, max_items: int = 32, *, auto_execute: bool = False, retry_budget: int = 2) -> list[dict]:
        """Drain + classify quarantined transitions for supervised recovery."""
        self._drain_quarantine_once()
        records = self.pull_quarantined(max_items=max_items)

        decisions = []
        for rec in records:
            action = self._recovery_action_for(rec)
            decision = {
                "from_state": rec.from_state,
                "to_state": rec.to_state,
                "reasons": list(rec.rejection_reasons),
                "action": action,
                "resonance_score": rec.resonance_score,
                "cause_id": rec.signature.cause_id or "",
            }
            if auto_execute:
                status, recovered = self._execute_recovery_action(rec, action, retry_budget=retry_budget)
                decision["execution_status"] = status
                decision["recovered"] = recovered
            decisions.append(decision)
        return decisions

    def collapse(self, resonance_packet: dict) -> str:
        """
        [STEP 3] Collapse: The Wave Function becomes Reality (Result).
        [SPIRIT LAYER] The Monad judges the outcome based on the Soul's Trajectory.
        """
        self.current_state = "COLLAPSED"
        flow_type = resonance_packet["flow_type"]
        amplitude = resonance_packet["amplitude"]
        seed = resonance_packet["seed"]

        # [TRAJECTORY ANALYSIS]
        narrative = self.trajectory.get_narrative()

        # Decision Logic based on Energy State (Not strict rules)

        result_str = ""
        if flow_type == "HARMONY":
            result_str = f"[MANIFEST] Validated Truth: '{seed}' (Energy: {amplitude:.2f})"

        elif flow_type == "ECHO":
            result_str = f"[AMPLIFY] Weak Signal Detected: '{seed}'. Requires more focus."

        elif flow_type == "DISSONANCE":
            # High energy but low resonance -> Creative Friction
            result_str = f"[GENESIS] New Pattern Detected: '{seed}'. Creating new memory path."

        else:
            result_str = "[VOID] Signal dissipated."

        return f"{result_str} | Path: {narrative}"
