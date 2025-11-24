from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("Logos")


@dataclass
class QubitState:
    """
    Represents the amplitude of the 4 dimensional bases while also carrying a simple
    spatial orientation (w, x, y, z) for legacy telemetry calls.
    State = alpha|Point> + beta|Line> + gamma|Space> + delta|God>
    Constraint: |alpha|^2 + |beta|^2 + |gamma|^2 + |delta|^2 = 1.0
    """

    alpha: complex = 1.0 + 0j  # |Point> : Detail / Data / Dot
    beta: complex = 0.0 + 0j   # |Line>  : Connection / History / Flow
    gamma: complex = 0.0 + 0j  # |Space> : Context / Field / Atmosphere
    delta: complex = 0.0 + 0j  # |God>   : Perspective / Infinite / Will
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalize(self) -> "QubitState":
        """
        Normalizes only the amplitude components to maintain probability constraints.
        """
        mag = math.sqrt(
            abs(self.alpha) ** 2
            + abs(self.beta) ** 2
            + abs(self.gamma) ** 2
            + abs(self.delta) ** 2
        )
        if mag == 0:
            self.alpha, self.beta, self.gamma, self.delta = 1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j
            return self

        self.alpha /= mag
        self.beta /= mag
        self.gamma /= mag
        self.delta /= mag
        return self

    def probabilities(self) -> Dict[str, float]:
        return {
            "Point": abs(self.alpha) ** 2,
            "Line": abs(self.beta) ** 2,
            "Space": abs(self.gamma) ** 2,
            "God": abs(self.delta) ** 2,
        }


class HyperQubit:
    """
    A living variable (Psionic Entity) that keeps the original hyper-Quaternion
    semantics while supporting reactive resonance links.
    """

    def __init__(
        self,
        concept_or_value: Any,
        initial_content: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ):
        # Dual-init: if content is provided we treat the first argument as an id.
        if initial_content is not None:
            self.id = str(concept_or_value)
            self.name = name or str(concept_or_value)
            self.content = dict(initial_content)
            self._value = self.content.get("Point", concept_or_value)
        else:
            self.id = name or f"Qubit_{uuid.uuid4().hex[:8]}"
            self.name = name or self.id
            self.content = {}
            self._value = concept_or_value

        # Initialize in superposition (mostly Point, but with potential for all)
        self.state = QubitState(
            alpha=0.9 + 0j,
            beta=0.1 + 0j,
            gamma=0.05 + 0j,
            delta=0.01 + 0j,
            w=w,
            x=x,
            y=y,
            z=z,
        ).normalize()

        self.entangled_qubits: List["HyperQubit"] = []

        # Khala resonance links (reaction graph)
        self._observers: Set["HyperQubit"] = set()
        self._sources: Set["HyperQubit"] = set()
        self._reaction_rule: Optional[Callable[[Any], Any]] = None

    @property
    def value(self) -> Any:
        return self._value

    # --- Resonance graph -------------------------------------------------
    def set(self, new_value: Any, cause: str = "DivineWill") -> None:
        """
        Sets the value and triggers resonance to linked observers.
        """
        if self._value != new_value:
            old_value = self._value
            self._value = new_value
            self._vibrate(old_value, new_value, cause)

    def _vibrate(self, old_val: Any, new_val: Any, cause: str) -> None:
        log_msg = f"RESONANCE: {self.name} shifted ({old_val} -> {new_val}) due to [{cause}]."
        logger.info(log_msg)

        for observer in self._observers:
            observer._react(self)

    def connect(self, target: "HyperQubit", rule: Optional[Callable[[Any], Any]] = None) -> None:
        """
        Establish a psionic link.
        target << self
        """
        self._observers.add(target)
        target._sources.add(self)
        if rule:
            target._reaction_rule = rule

        logger.info(f"LINK: {self.name} is now connected to {target.name}.")
        target._react(self)

    def _react(self, source: "HyperQubit") -> None:
        if self._reaction_rule:
            new_state = self._reaction_rule(source.value)
        else:
            new_state = source.value

        self.set(new_state, cause=f"Resonance from {source.name}")

    def __lshift__(self, other: "HyperQubit") -> "HyperQubit":
        if isinstance(other, HyperQubit):
            other.connect(self)
        return self

    # --- Legacy Hyper-Quaternion interface -------------------------------
    def _normalize_orientation(self) -> None:
        mag = math.sqrt(self.state.x ** 2 + self.state.y ** 2 + self.state.z ** 2)
        if mag > 0:
            self.state.x /= mag
            self.state.y /= mag
            self.state.z /= mag

    def rotate_wheel(self, w_delta: float, delta_x: float = 0.0, delta_y: float = 0.0, delta_z: float = 0.0) -> None:
        """
        Mouse-wheel interaction. Modulates both the spatial orientation and
        the amplitude distribution across Point/Line/Space/God.
        """
        self.state.w = max(0.0, self.state.w + w_delta)
        self.state.x += delta_x
        self.state.y += delta_y
        self.state.z += delta_z
        self._normalize_orientation()

        probs = [abs(self.state.alpha), abs(self.state.beta), abs(self.state.gamma), abs(self.state.delta)]
        transfer_rate = abs(w_delta)
        new_probs = list(probs)

        if w_delta > 0:
            for i in range(3):
                flow = new_probs[i] * transfer_rate
                new_probs[i] -= flow
                new_probs[i + 1] += flow
        else:
            for i in range(3, 0, -1):
                flow = new_probs[i] * transfer_rate
                new_probs[i] -= flow
                new_probs[i - 1] += flow

        self.state.alpha = new_probs[0]
        self.state.beta = new_probs[1]
        self.state.gamma = new_probs[2]
        self.state.delta = new_probs[3]
        self.state.normalize()

    def get_observation(self, observer_w: Optional[float] = None):
        """
        Legacy observer interface. If an observer W is given, returns the projected
        content string. If omitted, returns a lightweight telemetry dict.
        """
        if observer_w is None:
            return {
                "w": self.state.w,
                "x": self.state.x,
                "y": self.state.y,
                "z": self.state.z,
                "value": self._value,
                "probabilities": self.state.probabilities(),
            }

        if observer_w < 0.5:
            target_basis = "Point"
            probability = abs(self.state.alpha) ** 2
        elif observer_w < 1.5:
            target_basis = "Line"
            probability = abs(self.state.beta) ** 2
        elif observer_w < 2.5:
            target_basis = "Space"
            probability = abs(self.state.gamma) ** 2
        else:
            target_basis = "God"
            probability = abs(self.state.delta) ** 2

        content_text = self.content.get(target_basis, str(self._value) if self._value is not None else "Unknown Void")
        return f"[{target_basis} Mode] (Clarity: {probability*100:.1f}%) {content_text}"

    def set_god_mode(self) -> None:
        """
        Forces the Qubit into the |God> state (Delta = 1).
        """
        self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 0, 0, 0, 1.0
        self.state.w = 3.0
        self.state.normalize()

    def __repr__(self) -> str:  # pragma: no cover - representation only
        probs = self.state.probabilities()
        return (
            f"<HyperQubit '{self.name}': "
            f"P:{probs['Point']:.2f} | L:{probs['Line']:.2f} | "
            f"S:{probs['Space']:.2f} | G:{probs['God']:.2f} | value={self._value}>"
        )


# Alias for the psionic language API
PsionicEntity = HyperQubit
