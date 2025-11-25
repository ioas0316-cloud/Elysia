from typing import Callable, Dict, Tuple

import numpy as np


class ScalarField2D:
    """Lightweight wrapper for a 2D scalar field with optional cross-scale gating.

    The field array is provided lazily via `array_getter` to always read the latest
    world-owned buffer. Gradient is computed by a provided finite-difference
    function to avoid duplicating logic and to respect world-specific boundaries.
    """

    def __init__(
        self,
        name: str,
        scale: str,
        array_getter: Callable[[], np.ndarray],
        grad_func: Callable[[np.ndarray, float, float], Tuple[float, float]],
    ) -> None:
        self.name = name
        self.scale = scale  # 'micro' | 'meso' | 'macro'
        self._array_getter = array_getter
        self._grad_func = grad_func
        # Cross-scale gates (applied multiplicatively). Default to 1.0 (no change).
        self.gates: Dict[str, float] = {"macro": 1.0, "meso": 1.0, "micro": 1.0}

    def set_gate(self, scale: str, value: float) -> None:
        self.gates[scale] = float(np.clip(value, 0.0, 1.0))

    def _gate_product(self) -> float:
        g = self.gates
        return float(g.get("macro", 1.0) * g.get("meso", 1.0) * g.get("micro", 1.0))

    def array(self) -> np.ndarray:
        return self._array_getter()

    def sample(self, fx: float, fy: float) -> float:
        arr = self.array()
        # Nearest-neighbor sample (fast, consistent with finite-diff indexing)
        x = int(np.clip(fx, 0, arr.shape[1] - 1))
        y = int(np.clip(fy, 0, arr.shape[0] - 1))
        return float(arr[y, x]) * self._gate_product()

    def grad(self, fx: float, fy: float) -> Tuple[float, float]:
        arr = self.array()
        gx, gy = self._grad_func(arr, fx, fy)
        gprod = self._gate_product()
        return gx * gprod, gy * gprod


class VectorField2D:
    """Lightweight wrapper for a 2D vector field."""

    def __init__(
        self,
        name: str,
        scale: str,
        array_getter: Callable[[], np.ndarray],
    ) -> None:
        self.name = name
        self.scale = scale
        self._array_getter = array_getter
        self.gates: Dict[str, float] = {"macro": 1.0, "meso": 1.0, "micro": 1.0}

    def array(self) -> np.ndarray:
        return self._array_getter()

    def sample(self, fx: float, fy: float) -> Tuple[float, float]:
        arr = self.array()
        x = int(np.clip(fx, 0, arr.shape[1] - 1))
        y = int(np.clip(fy, 0, arr.shape[0] - 1))
        vx, vy = arr[y, x]
        return float(vx), float(vy)


class FieldRegistry:
    """Registry for scalar/vector fields with scale-aware gating.

    Minimal skeleton to unify existing fields (threat, hydration) under a common
    sampling/gradient interface without changing default behavior.
    """

    def __init__(self) -> None:
        self._scalar_fields: Dict[str, ScalarField2D] = {}
        self._vector_fields: Dict[str, VectorField2D] = {}

    def register_scalar(
        self,
        name: str,
        scale: str,
        array_getter: Callable[[], np.ndarray],
        grad_func: Callable[[np.ndarray, float, float], Tuple[float, float]],
    ) -> None:
        self._scalar_fields[name] = ScalarField2D(name, scale, array_getter, grad_func)

    def register_vector(
        self,
        name: str,
        scale: str,
        array_getter: Callable[[], np.ndarray],
    ) -> None:
        self._vector_fields[name] = VectorField2D(name, scale, array_getter)

    def set_gate(self, name: str, scale: str, value: float) -> None:
        fld = self._scalar_fields.get(name)
        if fld:
            fld.set_gate(scale, value)

    def sample(self, name: str, fx: float, fy: float) -> float:
        fld = self._scalar_fields.get(name)
        if not fld:
            raise KeyError(f"Field not found: {name}")
        return fld.sample(fx, fy)

    def grad(self, name: str, fx: float, fy: float) -> Tuple[float, float]:
        fld = self._scalar_fields.get(name)
        if not fld:
            raise KeyError(f"Field not found: {name}")
        return fld.grad(fx, fy)

    def vector(self, name: str, fx: float, fy: float) -> Tuple[float, float]:
        fld = self._vector_fields.get(name)
        if not fld:
            raise KeyError(f"Vector field not found: {name}")
        return fld.sample(fx, fy)
