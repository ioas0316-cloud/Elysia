"""
HyperQubit - Psionic entity carrying both value and hyper-quaternion-like state.
Ported from Legacy/Project_Elysia with minor cleanups for Core integration.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("HyperQubit")


@dataclass
class QubitState:
    """
    Amplitudes over 4 bases (Point/Line/Space/God) plus a simple orientation (wxyz).
    Constraint: |alpha|^2 + |beta|^2 + |gamma|^2 + |delta|^2 = 1.
    """

    alpha: complex = 1.0 + 0j  # Point : Detail / Data
    beta: complex = 0.0 + 0j   # Line  : Connection / History
    gamma: complex = 0.0 + 0j  # Space : Context / Field
    delta: complex = 0.0 + 0j  # God   : Perspective / Will
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalize(self) -> "QubitState":
        """Normalize amplitudes to preserve probability mass."""
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
    A living variable (Psionic Entity) with resonance links.
    """

    def __init__(
        self,
        concept_or_value: Any = None,
        initial_content: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
        value: Any = None,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
    ):
        concept_value = concept_or_value if value is None else value
        if initial_content is not None:
            self.id = str(concept_or_value)
            self.name = name or str(concept_or_value)
            self.content = dict(initial_content)
            self._value = self.content.get("Point", concept_value)
        else:
            self.id = name or f"Qubit_{uuid.uuid4().hex[:8]}"
            self.name = name or self.id
            self.content = {}
            self._value = concept_value

        self.state = QubitState(
            alpha=0.4 + 0j,
            beta=0.3 + 0j,
            gamma=0.2 + 0j,
            delta=0.1 + 0j,
            w=w,
            x=x,
            y=y,
            z=z,
        ).normalize()

        self.entangled_qubits: List["HyperQubit"] = []
        self._observers: Set["HyperQubit"] = set()
        self._sources: Set["HyperQubit"] = set()
        self._reaction_rule: Optional[Callable[[Any], Any]] = None

    @property
    def value(self) -> Any:
        return self._value

    # --- Resonance graph -------------------------------------------------
    def set(self, new_value: Any, cause: str = "DivineWill") -> None:
        """Set value and propagate resonance to observers."""
        if self._value != new_value:
            old_value = self._value
            self._value = new_value
            self._vibrate(old_value, new_value, cause)

    def _vibrate(self, old_val: Any, new_val: Any, cause: str) -> None:
        logger.info(f"RESONANCE: {self.name} shifted ({old_val} -> {new_val}) due to [{cause}].")
        for observer in self._observers:
            observer._react(self)

    def connect(self, target: "HyperQubit", rule: Optional[Callable[[Any], Any]] = None) -> None:
        """Establish a psionic link: target << self."""
        self._observers.add(target)
        target._sources.add(self)
        if rule:
            target._reaction_rule = rule
        logger.info(f"LINK: {self.name} is now connected to {target.name}.")
        target._react(self)

    def _react(self, source: "HyperQubit") -> None:
        new_state = self._reaction_rule(source.value) if self._reaction_rule else source.value
        self.set(new_state, cause=f"Resonance from {source.name}")

    def __lshift__(self, other: "HyperQubit") -> "HyperQubit":
        if isinstance(other, HyperQubit):
            other.connect(self)
        return self

    # --- Hyper-Quaternion-like orientation -------------------------------
    def _normalize_orientation(self) -> None:
        mag = math.sqrt(self.state.x ** 2 + self.state.y ** 2 + self.state.z ** 2)
        if mag > 0:
            self.state.x /= mag
            self.state.y /= mag
            self.state.z /= mag

    def rotate_wheel(self, w_delta: float, delta_x: float = 0.0, delta_y: float = 0.0, delta_z: float = 0.0) -> None:
        """Adjust orientation and redistribute amplitude between bases."""
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

        self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = new_probs
        self.state.normalize()

    def get_observation(self, observer_w: Optional[float] = None):
        """Legacy observer interface; returns telemetry dict or projected content."""
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
        """Force state to pure |God>."""
        self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 0, 0, 0, 1.0
        self.state.w = 3.0
        self.state.normalize()

    def collapse(self, mode: str = "max", reason: Optional[str] = None) -> str:
        """
        Collapse to a single basis.
        mode='max' picks highest probability; 'random' samples.
        """
        probs = self.state.probabilities()
        bases = ["Point", "Line", "Space", "God"]
        weights = [probs["Point"], probs["Line"], probs["Space"], probs["God"]]
        choice = random.choices(bases, weights=weights, k=1)[0] if mode == "random" else max(bases, key=lambda b: probs[b])

        if choice == "Point":
            self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 1.0, 0.0, 0.0, 0.0
        elif choice == "Line":
            self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 0.0, 1.0, 0.0, 0.0
        elif choice == "Space":
            self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 0.0, 0.0, 1.0, 0.0
        else:
            self.state.alpha, self.state.beta, self.state.gamma, self.state.delta = 0.0, 0.0, 0.0, 1.0
        self.state.normalize()
        self._value = self.content.get(choice, self._value)
        return choice

    def __repr__(self) -> str:  # pragma: no cover - representation only
        probs = self.state.probabilities()
        return (
            f"<HyperQubit '{self.name}': "
            f"P:{probs['Point']:.2f} | L:{probs['Line']:.2f} | "
            f"S:{probs['Space']:.2f} | G:{probs['God']:.2f} | value={self._value}>"
        )


# Alias for psionic API
PsionicEntity = HyperQubit
