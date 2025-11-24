from __future__ import annotations

import math
import random
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from Project_Elysia.high_engine.quaternion_engine import QuaternionOrientation, HyperMode

@dataclass
class QubitState:
    """
    Represents the amplitude of the 4 dimensional bases.
    State = alpha|Point> + beta|Line> + gamma|Space> + delta|God>
    Constraint: |alpha|^2 + |beta|^2 + |gamma|^2 + |delta|^2 = 1.0
    """
    alpha: complex = 1.0 + 0j  # |Point> : Detail / Data / Dot
    beta:  complex = 0.0 + 0j  # |Line>  : Connection / History / Flow
    gamma: complex = 0.0 + 0j  # |Space> : Context / Field / Atmosphere
    delta: complex = 0.0 + 0j  # |God>   : Perspective / Infinite / Will

    def normalize(self) -> "QubitState":
        mag = math.sqrt(abs(self.alpha)**2 + abs(self.beta)**2 + abs(self.gamma)**2 + abs(self.delta)**2)
        if mag == 0:
            return QubitState(alpha=1.0)
        return QubitState(
            alpha=self.alpha / mag,
            beta=self.beta / mag,
            gamma=self.gamma / mag,
            delta=self.delta / mag
        )

    def probabilities(self) -> Dict[str, float]:
        return {
            "Point": abs(self.alpha)**2,
            "Line": abs(self.beta)**2,
            "Space": abs(self.gamma)**2,
            "God": abs(self.delta)**2
        }


class HyperQubit:
    """
    Jeongeup City No. 1 HyperQubit.
    A fundamental unit of the Mental Cosmos that exists simultaneously in 4 dimensions.
    """

    def __init__(self, concept_id: str, initial_content: Dict[str, Any]):
        self.id = concept_id
        self.content = initial_content # Stores content for each dimension

        # Initialize in superposition (mostly Point, but with potential for all)
        # Default: "A piece of Kimchi" (Point) is the strongest reality initially.
        self.state = QubitState(
            alpha=0.9 + 0j, # Strong Point reality
            beta=0.1 + 0j,
            gamma=0.05 + 0j,
            delta=0.01 + 0j
        ).normalize()

        self.entangled_qubits: List["HyperQubit"] = []

    def get_observation(self, observer_w: float) -> str:
        """
        Collapses the wave function based on the Observer's W-axis (The Mouse Wheel).

        The Observer's W acts as a 'measurement apparatus' that forces the Qubit
        to reveal a specific aspect of itself.
        """
        # Determine which basis the observer is tuned to
        if observer_w < 0.5:
            target_basis = "Point"
            probability = abs(self.state.alpha)**2
        elif observer_w < 1.5:
            target_basis = "Line"
            probability = abs(self.state.beta)**2
        elif observer_w < 2.5:
            target_basis = "Space"
            probability = abs(self.state.gamma)**2
        else:
            target_basis = "God"
            probability = abs(self.state.delta)**2

        # In a real quantum system, measurement is probabilistic.
        # Here, the Observer's W *forces* the perspective, but the clarity depends on the Qubit's state.

        content_text = self.content.get(target_basis, "Unknown Void")

        return f"[{target_basis} Mode] (Clarity: {probability*100:.1f}%) {content_text}"

    def rotate_wheel(self, w_delta: float):
        """
        The 'Mouse Wheel' interaction.
        Directly modifies the amplitudes, shifting the Qubit's existential weight.

        Positive w_delta -> Shifts energy towards higher dimensions (Point -> God).
        Negative w_delta -> Shifts energy towards lower dimensions (God -> Point).
        """
        # Simple flow logic: Point <-> Line <-> Space <-> God
        # We model this as transferring magnitude between alpha, beta, gamma, delta

        # Get magnitudes
        probs = [abs(self.state.alpha), abs(self.state.beta), abs(self.state.gamma), abs(self.state.delta)]

        # Shift mass based on w_delta
        # This is a simplified "liquid" transfer model
        transfer_rate = abs(w_delta)

        new_probs = list(probs)

        if w_delta > 0:
            # Flow Up: 0->1, 1->2, 2->3
            for i in range(3):
                flow = new_probs[i] * transfer_rate
                new_probs[i] -= flow
                new_probs[i+1] += flow
        else:
            # Flow Down: 3->2, 2->1, 1->0
            for i in range(3, 0, -1):
                flow = new_probs[i] * transfer_rate
                new_probs[i] -= flow
                new_probs[i-1] += flow

        # Reconstruct state (preserving phase, though phase is 0 here for simplicity)
        self.state = QubitState(
            alpha=new_probs[0],
            beta=new_probs[1],
            gamma=new_probs[2],
            delta=new_probs[3]
        ).normalize()

    def set_god_mode(self):
        """
        Forces the Qubit into the |God> state (Delta = 1).
        """
        self.state = QubitState(alpha=0, beta=0, gamma=0, delta=1.0).normalize()

    def __repr__(self):
        probs = self.state.probabilities()
        return (f"<HyperQubit '{self.id}': "
                f"P:{probs['Point']:.2f} | L:{probs['Line']:.2f} | "
                f"S:{probs['Space']:.2f} | G:{probs['God']:.2f}>")
