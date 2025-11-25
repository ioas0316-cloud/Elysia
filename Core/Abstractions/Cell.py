from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# TODO: (SOUL FUSION) Replace these placeholders with actual Core Abstractions
# from .Tensor import SoulTensor
SoulTensor = None


@dataclass(slots=True)
class Cell:
    """
    Represents a single, living conceptual cell in Elysia's world.

    This is a lightweight, data-oriented abstraction (flyweight). Most mutable
    'state' (like HP, position, age) is managed in parallel NumPy arrays
    by the `World` simulation for extreme performance. This class holds the
    immutable 'identity' and core properties of a cell.

    Each cell also carries a tiny perceptron-like "brain" that can learn
    simple preferences from experience:

        y = w · x + b

    where `x` are environment features, `w` are learned weights, and `b`
    is the innate bias/temperament of the cell.
    """

    id: str
    dna: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)
    element_type: str = "unknown"

    # The 'Soul' is a complex object. For performance, a cell may not have its
    # soul 'materialized' at all times. It can be loaded on-demand by the World.
    soul: Optional[Any] = None  # Placeholder for SelfFractalCell
    tensor: Optional[SoulTensor] = None

    # Tiny Perceptron Brain
    perceptron_weights: Dict[str, float] = field(default_factory=dict)
    perceptron_bias: float = 0.0
    perceptron_learning_rate: float = 0.1

    def __post_init__(self):
        """
        Called after the dataclass is initialized.
        We can derive properties here.
        """
        # Ensure element_type is correctly set from properties if available.
        self.element_type = self.properties.get("element_type", "unknown")

        # Optional: seed temperament/bias from properties if provided.
        if "perceptron_bias" in self.properties:
            try:
                self.perceptron_bias = float(self.properties["perceptron_bias"])
            except (TypeError, ValueError):
                pass

        # Optional initial weights from properties, if present.
        initial_w = self.properties.get("perceptron_weights")
        if isinstance(initial_w, dict):
            # Only accept numeric values
            clean_weights: Dict[str, float] = {}
            for k, v in initial_w.items():
                try:
                    clean_weights[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if clean_weights:
                self.perceptron_weights = clean_weights

    def __repr__(self) -> str:
        # The 'status' is now external, so we represent the cell by its identity.
        return f"<Cell: {self.id}, Element: {self.element_type}>"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the cell's identity to a dictionary."""
        return {
            "id": self.id,
            "dna": self.dna,
            "properties": self.properties,
            "element_type": self.element_type,
            "perceptron_weights": self.perceptron_weights,
            "perceptron_bias": self.perceptron_bias,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cell":
        """Deserializes a cell from a dictionary."""
        props = data.get("properties", {}) or {}
        # If weights/bias were persisted at top-level keys, merge them back into properties.
        if "perceptron_weights" in data and "perceptron_weights" not in props:
            props["perceptron_weights"] = data["perceptron_weights"]
        if "perceptron_bias" in data and "perceptron_bias" not in props:
            props["perceptron_bias"] = data["perceptron_bias"]

        return cls(
            id=data.get("id", "unknown"),
            dna=data.get("dna", {}),
            properties=props,
        )

    # === Tiny Perceptron API ===

    def perceptron_output(self, features: Dict[str, float]) -> float:
        """
        Compute y = w · x + b for the given feature dictionary.

        Features are identified by string keys (e.g., "value_mass", "threat").
        Missing weights default to 0.0.
        """
        total = self.perceptron_bias
        for name, value in features.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            w = self.perceptron_weights.get(name, 0.0)
            total += w * v
        return total

    def perceptron_learn(self, features: Dict[str, float], target: float) -> float:
        """
        Single-step gradient descent update toward the target value.

        Args:
            features: Input feature dictionary.
            target: Desired output (e.g., +1.0 for "good outcome", -1.0 for "bad").

        Returns:
            The prediction error after the update (target - y_before).
        """
        try:
            t = float(target)
        except (TypeError, ValueError):
            return 0.0

        # Current prediction
        y = self.perceptron_output(features)
        error = t - y
        lr = float(self.perceptron_learning_rate)

        # Update weights
        for name, value in features.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            old_w = self.perceptron_weights.get(name, 0.0)
            self.perceptron_weights[name] = old_w + lr * error * v

        # Update bias
        self.perceptron_bias += lr * error

        return error
