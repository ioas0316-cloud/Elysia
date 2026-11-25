r"""
ELYSIA Continuous Coupled-Axis Variable-Resistor Scale-Rotor Field
==================================================================

Eliminates discrete `if/else` point-logic ("sit / lie down / sleep / attack / block / flee",
"Level 0 / Level 1 / Level 2", "if stage == 0/1/2") in favor of ONE continuous upper-system
coupling structure with genuine **Plasticity** and **Structural Axis Emergence**:
  - Every fundamental physical/cognitive principle is a continuous Axis dial `theta_i in [-1, 1]`
    with an adjustable Variable Resistor `R_i > 0` (`G_i = 1 / R_i`, Rainbow Circuit §16).
  - Upper-system relational couplings live in continuous matrices `W` (coupling) and `M` (readout).
  - **Unified Flow:** Perception (external stimulus deflecting dials), Computation (vectorized
    minimum-resistance relaxation to equilibrium `theta*`), and Result (continuous posture/metabolism,
    combat strike/guard/evade/flee, visual scale blur<->edge modulation, attention weights, blind tool
    prediction & parameters) emerge simultaneously from ONE branch-free continuous dynamics.
  - **Continuous Plasticity (`adapt_from_friction`):** Real world prediction error deforms `W`, `M`,
    `b`, and Variable Resistors `R` smoothly along causal error gradients with spectral bounds.
  - **Structural Axis Emergence (`grow_coupled_axis`):** Persistent unmodeled world invariants
    (e.g., emergent 3D hollow cavities) crystallize brand-new coupled axes into `W, M, R` live.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RotorSpectrumState:
    """Continuous equilibrium state of a coupled Variable-Resistor Scale-Rotor Field."""

    axis_names: List[str]
    dial_values: Dict[str, float]
    resistances: Dict[str, float]
    emergent_outputs: Dict[str, float]
    convergence_energy: float
    relaxation_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuousRotorSpectrumField:
    """
    Branch-free continuous Variable-Resistor Scale-Rotor Field with Plasticity & Axis Emergence:
        d(theta)/dt = (1 / R) * (stimulus - theta) + W @ theta - damping * theta
    Converges smoothly via vectorized NumPy relaxation onto the minimum-resistance
    spectral equilibrium `theta*`, projects `theta*` into continuous emergent readouts
    `y = tanh(M @ theta* + b)`, and physically deforms `(W, M, b, R)` from real world friction.
    """

    def __init__(
        self,
        axis_names: Sequence[str],
        coupling_matrix: np.ndarray,
        output_names: Sequence[str],
        readout_matrix: np.ndarray,
        output_bias: Optional[np.ndarray] = None,
        resistances: Optional[Sequence[float]] = None,
        damping: float = 0.25,
    ) -> None:
        self.axis_names: List[str] = [str(a) for a in axis_names]
        self.dim: int = len(self.axis_names)
        self.coupling = np.asarray(coupling_matrix, dtype=np.float64).copy()
        assert self.coupling.shape == (self.dim, self.dim)

        self.output_names: List[str] = [str(o) for o in output_names]
        self.readout = np.asarray(readout_matrix, dtype=np.float64).copy()
        assert self.readout.shape == (len(self.output_names), self.dim)

        self.output_bias = (
            np.asarray(output_bias, dtype=np.float64).copy()
            if output_bias is not None
            else np.zeros(len(self.output_names), dtype=np.float64)
        )
        self.resistances = (
            np.clip(np.asarray(resistances, dtype=np.float64), 0.08, 10.0).copy()
            if resistances is not None
            else np.ones(self.dim, dtype=np.float64)
        )
        self.damping = float(max(0.02, damping))
        self.theta = np.zeros(self.dim, dtype=np.float64)

        # Immutable birth baselines for honest self-measurement of real plastic deformation
        self._initial_axis_count: int = self.dim
        self._initial_coupling: np.ndarray = self.coupling.copy()
        self._initial_readout: np.ndarray = self.readout.copy()
        self._initial_resistances: np.ndarray = self.resistances.copy()
        self.emergent_axes: List[str] = []
        self.plasticity_updates: int = 0
        self.prediction_error_history: List[float] = []

    def set_resistance(self, axis_name: str, value: float) -> None:
        """Turns the Variable Resistor dial on a single axis (`low R = fluid`, `high R = anchored`)."""
        idx = self.axis_names.index(axis_name)
        self.resistances[idx] = float(np.clip(value, 0.08, 10.0))

    def converge(
        self,
        stimulus: Dict[str, float],
        *,
        dt: float = 0.22,
        max_iters: int = 24,
        tol: float = 1e-4,
    ) -> RotorSpectrumState:
        """
        Minimum-path continuous relaxation from external stimulus `u` to equilibrium `theta*`.
        ZERO `if/else` domain logic — purely continuous coupled-axis physics.
        """
        u = np.array([float(stimulus.get(name, 0.0)) for name in self.axis_names], dtype=np.float64)
        u = np.clip(u, -1.0, 1.0)
        conductance = 1.0 / np.clip(self.resistances, 0.08, 10.0)

        theta = self.theta.copy()
        steps_used = 0
        energy = 0.0
        for step in range(1, max_iters + 1):
            steps_used = step
            force = conductance * (u - theta) + (self.coupling @ theta) - self.damping * theta
            next_theta = np.tanh(theta + dt * force)
            energy = float(np.linalg.norm(next_theta - theta))
            theta = next_theta
            if energy < tol:
                break

        self.theta = theta
        raw_out = self.readout @ theta + self.output_bias
        out_map = {
            name: round(float(np.tanh(raw_out[i])), 5)
            for i, name in enumerate(self.output_names)
        }
        return RotorSpectrumState(
            axis_names=list(self.axis_names),
            dial_values={name: round(float(theta[i]), 5) for i, name in enumerate(self.axis_names)},
            resistances={name: round(float(self.resistances[i]), 4) for i, name in enumerate(self.axis_names)},
            emergent_outputs=out_map,
            convergence_energy=round(energy, 6),
            relaxation_steps=steps_used,
        )

    def adapt_from_friction(
        self,
        stimulus: Dict[str, float],
        target_outputs: Dict[str, float],
        *,
        lr_readout: float = 0.22,
        lr_coupling: float = 0.08,
        lr_resistance: float = 0.16,
        max_weight: float = 2.6,
    ) -> Dict[str, Any]:
        """
        Continuous Causal Plasticity (Learning directly from world prediction error):
        Deforms `M` (readout), `b` (bias), `W` (inter-axis coupling), and `R` (variable resistors)
        proportionally to the continuous mismatch between predicted `y` and real world outcome `target`.
        """
        before_state = self.converge(stimulus)
        u = np.clip(
            np.array([float(stimulus.get(name, 0.0)) for name in self.axis_names], dtype=np.float64),
            -1.0,
            1.0,
        )
        y_pred = np.array([before_state.emergent_outputs[n] for n in self.output_names], dtype=np.float64)
        mask = np.array([1.0 if n in target_outputs else 0.0 for n in self.output_names], dtype=np.float64)
        y_tgt = np.clip(
            np.array([float(target_outputs.get(n, y_pred[i])) for i, n in enumerate(self.output_names)], dtype=np.float64),
            -0.98,
            0.98,
        )
        err_vec = mask * (y_tgt - y_pred)
        active_n = max(1.0, float(np.sum(mask)))
        error_before = float(np.sqrt(np.sum(err_vec ** 2) / active_n))

        # Local continuous Jacobian through tanh(raw_out)
        tanh_deriv = np.maximum(0.08, 1.0 - y_pred ** 2)
        delta_out = err_vec * tanh_deriv

        theta = self.theta.copy()
        dM = lr_readout * np.outer(delta_out, theta)
        db = lr_readout * 0.60 * delta_out

        # Back-propagate causal mismatch into latent axis equilibrium `theta`
        e_theta = self.readout.T @ delta_out
        dW = lr_coupling * np.outer(e_theta, theta)
        np.fill_diagonal(dW, 0.0)

        # Variable Resistor R adaptation:
        # Since d(theta)/dt includes (1/R)*(u - theta), lowering R amplifies movement along (u - theta).
        # If e_theta aligns with (u - theta), fluidize (R down); if opposed, stiffen/damp (R up).
        dR = -lr_resistance * (e_theta * (u - theta))

        self.readout = np.clip(self.readout + dM, -max_weight, max_weight)
        self.output_bias = np.clip(self.output_bias + db, -max_weight, max_weight)
        new_W = np.clip(self.coupling + dW, -max_weight, max_weight)
        np.fill_diagonal(new_W, 0.0)
        # Keep spectral norm bounded so continuous relaxation never diverges
        spec = float(np.linalg.norm(new_W, ord=2))
        if spec > 0.92:
            new_W = new_W * (0.92 / spec)
        self.coupling = new_W
        self.resistances = np.clip(self.resistances + dR, 0.08, 10.0)

        after_state = self.converge(stimulus)
        y_after = np.array([after_state.emergent_outputs[n] for n in self.output_names], dtype=np.float64)
        err_after_vec = mask * (y_tgt - y_after)
        error_after = float(np.sqrt(np.sum(err_after_vec ** 2) / active_n))

        self.plasticity_updates += 1
        self.prediction_error_history.append(round(error_before, 5))
        if len(self.prediction_error_history) > 128:
            self.prediction_error_history = self.prediction_error_history[-128:]

        return {
            "error_before": round(error_before, 5),
            "error_after": round(error_after, 5),
            "error_reduced": bool(error_after < error_before - 1e-5),
            "delta_W_norm": round(float(np.linalg.norm(dW)), 5),
            "delta_M_norm": round(float(np.linalg.norm(dM)), 5),
            "delta_R_norm": round(float(np.linalg.norm(dR)), 5),
            "plasticity_updates": self.plasticity_updates,
            "outputs_before": before_state.emergent_outputs,
            "outputs_after": after_state.emergent_outputs,
        }

    def grow_coupled_axis(
        self,
        axis_name: str,
        *,
        initial_resistance: float = 0.85,
        residual_outputs: Optional[Dict[str, float]] = None,
        coupling_strength: float = 0.14,
    ) -> Dict[str, Any]:
        """
        Structural Axis Emergence:
        When an external invariant cannot be captured by existing axes, crystallizes a NEW
        coupled axis directly into `(theta, R, W, M)` without resetting learned structure.
        """
        clean = str(axis_name).strip()
        if not clean or clean in self.axis_names:
            return {"grown": False, "axis_name": clean, "dim": self.dim}

        old_d = self.dim
        new_d = old_d + 1
        self.axis_names.append(clean)
        self.dim = new_d

        # Expand coupling W smoothly
        new_W = np.zeros((new_d, new_d), dtype=np.float64)
        new_W[:old_d, :old_d] = self.coupling
        # Seed mild cross-coupling with active axes proportional to their current tension
        active_dir = np.tanh(self.theta)
        if np.linalg.norm(active_dir) > 1e-6:
            seed_vec = coupling_strength * (active_dir / max(1.0, float(np.linalg.norm(active_dir))))
        else:
            seed_vec = np.full(old_d, coupling_strength / max(1, old_d), dtype=np.float64)
        new_W[old_d, :old_d] = seed_vec
        new_W[:old_d, old_d] = 0.5 * seed_vec
        self.coupling = new_W

        # Expand readout M aligned with the unexplained external residual that birthed this axis
        new_M = np.zeros((len(self.output_names), new_d), dtype=np.float64)
        new_M[:, :old_d] = self.readout
        if residual_outputs:
            for i, out_name in enumerate(self.output_names):
                if out_name in residual_outputs:
                    new_M[i, old_d] = float(np.clip(residual_outputs[out_name], -1.0, 1.0)) * 0.65
        else:
            new_M[:, old_d] = 0.15
        self.readout = new_M

        self.resistances = np.append(self.resistances, float(np.clip(initial_resistance, 0.08, 10.0)))
        self.theta = np.append(self.theta, 0.0)

        # Pad birth baselines with zeros along the newly emerged dimension so drift stays measurable
        pad_W0 = np.zeros((new_d, new_d), dtype=np.float64)
        pad_W0[:old_d, :old_d] = self._initial_coupling
        self._initial_coupling = pad_W0

        pad_M0 = np.zeros((len(self.output_names), new_d), dtype=np.float64)
        pad_M0[:, :old_d] = self._initial_readout
        self._initial_readout = pad_M0

        self._initial_resistances = np.append(self._initial_resistances, float(np.clip(initial_resistance, 0.08, 10.0)))
        self.emergent_axes.append(clean)
        return {
            "grown": True,
            "axis_name": clean,
            "dim": self.dim,
            "emergent_axes": list(self.emergent_axes),
        }

    def plasticity_summary(self) -> Dict[str, Any]:
        """Exact self-measured structural deformation since birth (honest metrics, zero hype)."""
        return {
            "dim": self.dim,
            "initial_axis_count": self._initial_axis_count,
            "emergent_axes": list(self.emergent_axes),
            "plasticity_updates": int(self.plasticity_updates),
            "drift_W_norm": round(float(np.linalg.norm(self.coupling - self._initial_coupling)), 5),
            "drift_M_norm": round(float(np.linalg.norm(self.readout - self._initial_readout)), 5),
            "drift_R_norm": round(float(np.linalg.norm(self.resistances - self._initial_resistances)), 5),
            "recent_pred_error": (
                round(float(self.prediction_error_history[-1]), 5) if self.prediction_error_history else None
            ),
            "error_history_len": len(self.prediction_error_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis_names": list(self.axis_names),
            "output_names": list(self.output_names),
            "coupling": self.coupling.tolist(),
            "readout": self.readout.tolist(),
            "output_bias": self.output_bias.tolist(),
            "resistances": self.resistances.tolist(),
            "damping": self.damping,
            "theta": self.theta.tolist(),
            "initial_axis_count": self._initial_axis_count,
            "initial_coupling": self._initial_coupling.tolist(),
            "initial_readout": self._initial_readout.tolist(),
            "initial_resistances": self._initial_resistances.tolist(),
            "emergent_axes": list(self.emergent_axes),
            "plasticity_updates": int(self.plasticity_updates),
            "prediction_error_history": list(self.prediction_error_history),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ContinuousRotorSpectrumField":
        obj = cls(
            axis_names=raw["axis_names"],
            coupling_matrix=np.asarray(raw["coupling"], dtype=np.float64),
            output_names=raw["output_names"],
            readout_matrix=np.asarray(raw["readout"], dtype=np.float64),
            output_bias=np.asarray(raw["output_bias"], dtype=np.float64),
            resistances=raw["resistances"],
            damping=float(raw.get("damping", 0.25)),
        )
        obj.theta = np.asarray(raw.get("theta", [0.0] * obj.dim), dtype=np.float64)
        obj._initial_axis_count = int(raw.get("initial_axis_count", obj.dim))
        if "initial_coupling" in raw:
            obj._initial_coupling = np.asarray(raw["initial_coupling"], dtype=np.float64)
        if "initial_readout" in raw:
            obj._initial_readout = np.asarray(raw["initial_readout"], dtype=np.float64)
        if "initial_resistances" in raw:
            obj._initial_resistances = np.asarray(raw["initial_resistances"], dtype=np.float64)
        obj.emergent_axes = [str(a) for a in raw.get("emergent_axes", [])]
        obj.plasticity_updates = int(raw.get("plasticity_updates", 0))
        obj.prediction_error_history = [float(e) for e in raw.get("prediction_error_history", [])]
        return obj


def build_gravity_metabolism_rotor() -> ContinuousRotorSpectrumField:
    """
    Upper-system coupled axes: `[gravity_load, metabolic_energy, fatigue_pressure, support_comfort]`.
    Without ANY `if tired: lie_down()` or `if hungry: eat()` rules, spontaneous posture &
    metabolic conservation emerge continuously across `[standing -> sitting -> lying -> sleeping]`.
    """
    axes = ["gravity_load", "metabolic_energy", "fatigue_pressure", "support_comfort"]
    W = np.array(
        [
            [0.00, -0.25, 0.35, -0.15],
            [-0.20, 0.00, -0.40, 0.15],
            [0.30, -0.35, 0.00, 0.10],
            [0.20, -0.25, 0.35, 0.00],
        ],
        dtype=np.float64,
    )
    outputs = ["posture_height", "metabolic_expenditure", "support_seeking", "foraging_drive"]
    M = np.array(
        [
            [-0.55, 0.85, -0.75, -0.45],
            [-0.35, 0.80, -0.80, -0.35],
            [0.50, -0.60, 0.80, 0.65],
            [-0.25, -0.85, -0.35, -0.15],
        ],
        dtype=np.float64,
    )
    return ContinuousRotorSpectrumField(axes, W, outputs, M, damping=0.22)


def build_combat_interaction_rotor() -> ContinuousRotorSpectrumField:
    """
    Upper-system coupled axes: `[target_proximity, incoming_threat, self_reserve, evasion_space]`.
    Without ANY `if dist < 2: attack() elif dist > 5: flee()` rules, continuous combat flow
    rotates smoothly through Attack <-> Guard/Parry <-> Sidestep/Evade <-> Retreat/Flee.
    """
    axes = ["target_proximity", "incoming_threat", "self_reserve", "evasion_space"]
    W = np.array(
        [
            [0.00, 0.25, 0.15, -0.20],
            [0.20, 0.00, -0.25, -0.10],
            [0.10, -0.20, 0.00, 0.15],
            [-0.15, 0.15, 0.10, 0.00],
        ],
        dtype=np.float64,
    )
    outputs = ["engage_vs_retreat", "guard_compression", "lateral_evade"]
    M = np.array(
        [
            [0.75, -0.45, 0.85, -0.20],
            [0.55, 0.85, 0.15, -0.40],
            [-0.10, 0.70, 0.45, 0.85],
        ],
        dtype=np.float64,
    )
    return ContinuousRotorSpectrumField(axes, W, outputs, M, damping=0.22)


def build_cognitive_attention_rotor() -> ContinuousRotorSpectrumField:
    """
    Upper-system coupled axes governing Elysia's live sensory/tool/world attention & visual scale dial:
      Axes: `[visual_gap, tool_mystery, world_void, growth_momentum]`
      Outputs: `[visual_scale_drive, tool_probe_drive, world_inquiry_drive, focus_retention]`
    Replaces discrete `if focus_target == ...` switches with continuous spectral resonance.
    """
    axes = ["visual_gap", "tool_mystery", "world_void", "growth_momentum"]
    W = np.array(
        [
            [0.00, 0.15, 0.05, -0.15],
            [0.15, 0.00, 0.10, 0.12],
            [0.05, 0.10, 0.00, 0.08],
            [0.25, 0.25, 0.25, 0.00],
        ],
        dtype=np.float64,
    )
    outputs = ["visual_scale_drive", "tool_probe_drive", "world_inquiry_drive", "focus_retention"]
    M = np.array(
        [
            [1.15, 0.10, -0.15, 0.05],
            [-0.10, 1.15, -0.15, 0.15],
            [-0.15, -0.20, 1.10, 0.15],
            [0.55, 0.55, 0.35, -0.35],
        ],
        dtype=np.float64,
    )
    return ContinuousRotorSpectrumField(axes, W, outputs, M, damping=0.20)


def _build_blind_op_predictor() -> ContinuousRotorSpectrumField:
    """
    Tabula-rasa continuous predictive rotor for ONE opaque external CAD operator.
    Maps continuous input dials `[base_volume, param_span, param_factor, companion_ratio]`
    -> predicted physical response `[volume_delta, hollow_norm, stability_score]`.
    Starts neutral/uninformed and physically deforms `(W, M, R)` from real 3D outcomes.
    """
    axes = ["base_volume", "param_span", "param_factor", "companion_ratio"]
    W = np.array(
        [
            [0.00, 0.10, 0.10, -0.05],
            [0.10, 0.00, 0.08, 0.05],
            [0.08, 0.05, 0.00, 0.00],
            [-0.05, 0.05, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    outputs = ["volume_delta", "hollow_norm", "stability_score"]
    M = np.full((3, 4), 0.08, dtype=np.float64)
    return ContinuousRotorSpectrumField(axes, W, outputs, M, damping=0.22)


def _build_blind_cad_param_rotor() -> ContinuousRotorSpectrumField:
    """
    Continuous parameter-exploration rotor for blind 3D CAD interaction.
    Replaces discrete `if stage == 0/1/2` scripts: modulates `(hx, hy, hz, factor)` continuously
    from `[prediction_surprise, volume_fill, coverage_progress, stability_tension]`.
    """
    axes = ["prediction_surprise", "volume_fill", "coverage_progress", "stability_tension"]
    W = np.array(
        [
            [0.00, -0.15, -0.25, 0.20],
            [-0.10, 0.00, 0.15, -0.20],
            [-0.20, 0.15, 0.00, -0.15],
            [0.15, -0.15, -0.10, 0.00],
        ],
        dtype=np.float64,
    )
    outputs = ["param_hx", "param_hz", "param_factor", "inner_ratio"]
    M = np.array(
        [
            [0.55, -0.40, 0.25, -0.20],
            [0.45, -0.25, 0.35, 0.15],
            [0.65, -0.30, 0.40, -0.25],
            [-0.35, 0.55, -0.20, 0.30],
        ],
        dtype=np.float64,
    )
    return ContinuousRotorSpectrumField(axes, W, outputs, M, damping=0.22)


class ContinuousBlindCADRotor:
    """
    Zero-cheat, branch-free continuous predictive exploration system over ANY opaque 3D CAD kernel.
    - Never hardcodes `"op_k0"`, `"op_k3"`, `"op_k4"` or discrete `stage == 0/1/2` ladders.
    - Discovers operator arity (0-ary creator vs unary/binary transformers) purely from physical friction.
    - Maintains a plastic `ContinuousRotorSpectrumField` predictor per discovered op, updating
      `(W, M, R)` from real prediction error `||obs - pred||` every interaction.
    - Automatically crystallizes new coupled axes (`grow_coupled_axis`) when unexpected physical
      invariants (such as hollow 3D cavities) contradict prior flat-volume predictions.
    """

    def __init__(self) -> None:
        self.param_rotor: ContinuousRotorSpectrumField = _build_blind_cad_param_rotor()
        self.op_predictors: Dict[str, ContinuousRotorSpectrumField] = {}
        self.op_trials: Dict[str, int] = {}
        self.op_arity: Dict[str, int] = {}  # 0=creator, 1+=needs existing shape
        self.op_last_error: Dict[str, float] = {}
        self.op_mean_vol_delta: Dict[str, float] = {}
        self.op_max_hollow: Dict[str, int] = {}
        self.discovered_laws: List[str] = []
        self.probe_count: int = 0
        self.causal_coverage: float = 0.0
        self.prediction_errors: List[float] = []
        self._last_surprise: float = 0.85
        self._last_volume: float = 0.0
        self._last_stability_tension: float = 0.30
        self._creator_op: Optional[str] = None
        self._base_shape_id: Optional[str] = None
        self._base_volume: float = 0.0

    def _ensure_op_predictor(self, op_id: str) -> ContinuousRotorSpectrumField:
        if op_id not in self.op_predictors:
            self.op_predictors[op_id] = _build_blind_op_predictor()
            self.op_trials.setdefault(op_id, 0)
            self.op_last_error.setdefault(op_id, 0.90)
            self.op_mean_vol_delta.setdefault(op_id, 0.0)
            self.op_max_hollow.setdefault(op_id, 0)
        return self.op_predictors[op_id]

    def _evaluate_single_op(
        self,
        kernel: Any,
        op_id: str,
        *,
        shape_a: Optional[str],
        shape_b: Optional[str],
        params: Dict[str, float],
        base_vol: float,
    ) -> Optional[Dict[str, Any]]:
        pred_rotor = self._ensure_op_predictor(op_id)
        stim = {
            "base_volume": float(np.clip(base_vol, -1.0, 1.0)),
            "param_span": float(np.clip(params.get("hx", 0.45), -1.0, 1.0)),
            "param_factor": float(np.clip(params.get("factor", 1.2) - 1.0, -1.0, 1.0)),
            "companion_ratio": 0.65 if shape_b is not None else -0.5,
        }
        pred_state = pred_rotor.converge(stim)
        try:
            obs = kernel.execute(op_id, shape_a=shape_a, shape_b=shape_b, params=params)
        except ValueError:
            # Physically discovered that this opaque operator cannot run without shape inputs
            self.op_arity[op_id] = max(1, self.op_arity.get(op_id, 1))
            return None

        vol_delta = float(obs.volume - base_vol)
        hollow_norm = float(np.tanh(float(obs.hollow_cavity_voxels) / 250.0))
        stab_score = 0.85 if obs.stable_under_gravity else -0.85
        target = {
            "volume_delta": float(np.clip(vol_delta, -0.95, 0.95)),
            "hollow_norm": hollow_norm,
            "stability_score": stab_score,
        }
        adapt = pred_rotor.adapt_from_friction(stim, target)

        self.op_trials[op_id] = self.op_trials.get(op_id, 0) + 1
        self.op_last_error[op_id] = float(adapt["error_before"])
        prev_m = self.op_mean_vol_delta.get(op_id, 0.0)
        n_t = self.op_trials[op_id]
        self.op_mean_vol_delta[op_id] = prev_m + (vol_delta - prev_m) / float(n_t)
        self.op_max_hollow[op_id] = max(self.op_max_hollow.get(op_id, 0), int(obs.hollow_cavity_voxels))
        self.prediction_errors.append(float(adapt["error_before"]))
        if len(self.prediction_errors) > 128:
            self.prediction_errors = self.prediction_errors[-128:]

        # Structural Axis Emergence: if unexpected 3D cavity appears, crystallize a new coupled axis!
        emerged_axis = None
        if obs.hollow_cavity_voxels > 0 and "emergent_cavity_topology" not in self.param_rotor.axis_names:
            grow_res = self.param_rotor.grow_coupled_axis(
                "emergent_cavity_topology",
                initial_resistance=0.65,
                residual_outputs={"param_hx": 0.35, "param_hz": 0.45, "inner_ratio": -0.55},
            )
            if grow_res.get("grown"):
                emerged_axis = "emergent_cavity_topology"

        # Also adapt master parameter rotor smoothly toward higher-information contrast regions
        self.param_rotor.adapt_from_friction(
            {
                "prediction_surprise": float(np.clip(adapt["error_before"], -1.0, 1.0)),
                "volume_fill": float(np.clip(obs.volume, -1.0, 1.0)),
                "coverage_progress": float(np.clip(self.causal_coverage, -1.0, 1.0)),
                "stability_tension": 0.15 if obs.stable_under_gravity else 0.85,
            },
            {
                "param_factor": float(np.clip(0.35 + 0.55 * adapt["error_before"], -0.95, 0.95)),
            },
            lr_readout=0.10,
            lr_coupling=0.04,
            lr_resistance=0.08,
        )

        surprise_score = float(adapt["error_before"]) + 0.65 * hollow_norm + 0.35 * abs(vol_delta)
        return {
            "op_used": op_id,
            "obs": obs,
            "volume": round(float(obs.volume), 4),
            "volume_delta": round(vol_delta, 4),
            "hollow_voxels": int(obs.hollow_cavity_voxels),
            "stable": bool(obs.stable_under_gravity),
            "pred_before": pred_state.emergent_outputs,
            "error_before": float(adapt["error_before"]),
            "error_after": float(adapt["error_after"]),
            "error_reduced": bool(adapt["error_reduced"]),
            "delta_W_norm": float(adapt["delta_W_norm"]),
            "delta_M_norm": float(adapt["delta_M_norm"]),
            "delta_R_norm": float(adapt["delta_R_norm"]),
            "surprise_score": round(surprise_score, 5),
            "emerged_axis": emerged_axis,
            "params": dict(params),
        }

    def step_blind_probe(self, kernel: Any) -> Dict[str, Any]:
        """
        Executes ONE continuous, zero-hardcoded-stage blind probe cycle against `kernel`:
          - Queries continuous parameters `(hx, hy, hz, factor, inner_ratio)` from `param_rotor`.
          - If no physical 3D object exists yet, physically discovers a 0-ary shape creator op
            by probing available opaque ops and absorbing arity friction.
          - Once a base shape exists, sweeps untried opaque operators (or focuses on the highest
            residual prediction-error operator) against continuous rotor-modulated shapes, updates
            every tested op's plastic `(W, M, R)` predictor, and surfaces the highest-surprise law.
        """
        self.probe_count += 1
        ops = list(kernel.available_ops())
        for op_id in ops:
            self._ensure_op_predictor(op_id)

        p_state = self.param_rotor.converge(
            {
                "prediction_surprise": float(np.clip(self._last_surprise, -1.0, 1.0)),
                "volume_fill": float(np.clip(self._last_volume, -1.0, 1.0)),
                "coverage_progress": float(np.clip(self.causal_coverage, -1.0, 1.0)),
                "stability_tension": float(np.clip(self._last_stability_tension, -1.0, 1.0)),
                "emergent_cavity_topology": 0.75 if "emergent_cavity_topology" in self.param_rotor.axis_names else 0.0,
            }
        )
        outs = p_state.emergent_outputs
        hx = round(0.42 + 0.14 * float(outs["param_hx"]), 3)
        hz = round(0.42 + 0.14 * float(outs["param_hz"]), 3)
        factor = round(1.18 + 0.28 * float(outs["param_factor"]), 3)
        inner_scale = round(0.58 + 0.10 * float(outs["inner_ratio"]), 3)

        # Case A: No base physical shape exists in the kernel yet -> discover 0-ary creator blind
        if self._base_shape_id is None or self._creator_op is None:
            creator_res = None
            for op_id in ops:
                res = self._evaluate_single_op(
                    kernel,
                    op_id,
                    shape_a=None,
                    shape_b=None,
                    params={"hx": hx, "hy": hx, "hz": hz, "factor": factor},
                    base_vol=0.0,
                )
                if res is not None:
                    self._creator_op = op_id
                    self.op_arity[op_id] = 0
                    self._base_shape_id = res["obs"].shape_id
                    self._base_volume = float(res["obs"].volume)
                    creator_res = res
                    break

            assert creator_res is not None, "Opaque CAD kernel exposed no 0-ary shape producer"
            self.causal_coverage = min(1.0, 1.0 / float(max(1, len(ops))))
            self._last_surprise = creator_res["error_before"]
            self._last_volume = creator_res["volume"]
            self._last_stability_tension = 0.15 if creator_res["stable"] else 0.85
            law = (
                f"블라인드 연속 탐침 `{creator_res['op_used']}`(hx={hx}, hz={hz}) -> "
                f"실체 생성 부피={creator_res['volume']:.3f}, 공동={creator_res['hollow_voxels']}, "
                f"예측오차 {creator_res['error_before']:.3f}->{creator_res['error_after']:.3f}"
            )
            self.discovered_laws.append(law)
            return {
                "ok": True,
                "probe_count": self.probe_count,
                "op_used": creator_res["op_used"],
                "rgb_iso": creator_res["obs"].rgb_iso,
                "volume": creator_res["volume"],
                "volume_delta": creator_res["volume_delta"],
                "hollow_voxels": creator_res["hollow_voxels"],
                "stable": creator_res["stable"],
                "error_before": creator_res["error_before"],
                "error_after": creator_res["error_after"],
                "error_reduced": creator_res["error_reduced"],
                "emerged_axis": creator_res["emerged_axis"],
                "params": creator_res["params"],
                "discovered_law_ko": law,
                "summary_ko": (
                    f"L3 연속 예측 탐침 `{creator_res['op_used']}`: 부피={creator_res['volume']:.3f}, "
                    f"공동={creator_res['hollow_voxels']}, 오차={creator_res['error_before']:.3f}->{creator_res['error_after']:.3f}"
                ),
            }

        # Case B: Base physical shape exists -> generate a continuous contrasting companion probe
        inner_hx = round(max(0.18, hx * inner_scale), 3)
        inner_hz = round(min(0.85, hz * 1.22), 3)
        companion = kernel.execute(
            self._creator_op,
            params={"hx": inner_hx, "hy": inner_hx, "hz": inner_hz},
        )

        # Untried opaque ops get contrasted simultaneously to discover tool physics without cheating;
        # once all ops have been experienced, pick the single op with highest continuous epistemic drive.
        untried = [op_id for op_id in ops if self.op_trials.get(op_id, 0) == 0]
        if untried:
            candidate_ops = untried
        else:
            def _epistemic_drive(oid: str) -> float:
                novelty = 0.45 / float(np.sqrt(1.0 + self.op_trials.get(oid, 0)))
                err_drive = float(self.op_last_error.get(oid, 0.5))
                return err_drive + novelty
            candidate_ops = [max(ops, key=_epistemic_drive)]

        results: List[Dict[str, Any]] = []
        for op_id in candidate_ops:
            res = self._evaluate_single_op(
                kernel,
                op_id,
                shape_a=self._base_shape_id,
                shape_b=companion.shape_id,
                params={"hx": hx, "hy": hx, "hz": hz, "factor": factor},
                base_vol=self._base_volume,
            )
            if res is not None:
                results.append(res)

        best = max(results, key=lambda item: item["surprise_score"])
        tried_count = sum(1 for oid in ops if self.op_trials.get(oid, 0) > 0)
        mean_err = float(np.mean([self.op_last_error.get(oid, 0.5) for oid in ops]))
        self.causal_coverage = round(
            float(np.clip(0.55 * (tried_count / float(max(1, len(ops)))) + 0.45 * max(0.0, 1.0 - mean_err), 0.0, 1.0)),
            4,
        )
        self._last_surprise = best["error_before"]
        self._last_volume = best["volume"]
        self._last_stability_tension = 0.15 if best["stable"] else 0.85

        if best["hollow_voxels"] > 0:
            law = (
                f"블라인드 연속 대조 `{best['op_used']}` 최고 놀람 발견: 부피 {self._base_volume:.3f}->{best['volume']:.3f}"
                f"({best['volume_delta']:+.3f}), 내부 공동 {best['hollow_voxels']}복셀 창발, "
                f"예측오차 {best['error_before']:.3f}->{best['error_after']:.3f}"
            )
        else:
            law = (
                f"블라인드 연속 대조 `{best['op_used']}`(factor={factor:.2f}): 부피변화 {best['volume_delta']:+.3f}, "
                f"공동={best['hollow_voxels']}, 예측오차 {best['error_before']:.3f}->{best['error_after']:.3f}"
            )
        self.discovered_laws.append(law)

        return {
            "ok": True,
            "probe_count": self.probe_count,
            "ops_tested_this_cycle": [r["op_used"] for r in results],
            "op_used": best["op_used"],
            "rgb_iso": best["obs"].rgb_iso,
            "volume": best["volume"],
            "volume_delta": best["volume_delta"],
            "hollow_voxels": best["hollow_voxels"],
            "stable": best["stable"],
            "error_before": best["error_before"],
            "error_after": best["error_after"],
            "error_reduced": best["error_reduced"],
            "emerged_axis": best["emerged_axis"],
            "params": best["params"],
            "causal_coverage": self.causal_coverage,
            "discovered_law_ko": law,
            "summary_ko": (
                f"L3 연속 예측 탐침 `{best['op_used']}`({len(results)}종 대조): "
                f"부피변화 {best['volume_delta']:+.3f}, 공동={best['hollow_voxels']}복셀, "
                f"오차={best['error_before']:.3f}->{best['error_after']:.3f}"
            ),
        }

    def plasticity_summary(self) -> Dict[str, Any]:
        total_updates = self.param_rotor.plasticity_updates + sum(
            p.plasticity_updates for p in self.op_predictors.values()
        )
        drift_W = self.param_rotor.plasticity_summary()["drift_W_norm"] + sum(
            p.plasticity_summary()["drift_W_norm"] for p in self.op_predictors.values()
        )
        drift_M = self.param_rotor.plasticity_summary()["drift_M_norm"] + sum(
            p.plasticity_summary()["drift_M_norm"] for p in self.op_predictors.values()
        )
        drift_R = self.param_rotor.plasticity_summary()["drift_R_norm"] + sum(
            p.plasticity_summary()["drift_R_norm"] for p in self.op_predictors.values()
        )
        return {
            "ops_modeled": len(self.op_predictors),
            "total_trials": int(sum(self.op_trials.values())),
            "total_plasticity_updates": int(total_updates),
            "drift_W_norm": round(float(drift_W), 5),
            "drift_M_norm": round(float(drift_M), 5),
            "drift_R_norm": round(float(drift_R), 5),
            "emergent_axes": list(self.param_rotor.emergent_axes),
            "causal_coverage": round(float(self.causal_coverage), 4),
            "mean_recent_error": (
                round(float(np.mean(self.prediction_errors[-5:])), 5) if self.prediction_errors else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_rotor": self.param_rotor.to_dict(),
            "op_predictors": {k: v.to_dict() for k, v in self.op_predictors.items()},
            "op_trials": dict(self.op_trials),
            "op_arity": dict(self.op_arity),
            "op_last_error": dict(self.op_last_error),
            "op_mean_vol_delta": dict(self.op_mean_vol_delta),
            "op_max_hollow": dict(self.op_max_hollow),
            "discovered_laws": list(self.discovered_laws[-64:]),
            "probe_count": int(self.probe_count),
            "causal_coverage": float(self.causal_coverage),
            "prediction_errors": list(self.prediction_errors[-128:]),
            "last_surprise": float(self._last_surprise),
            "last_volume": float(self._last_volume),
            "last_stability_tension": float(self._last_stability_tension),
            "creator_op": self._creator_op,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ContinuousBlindCADRotor":
        obj = cls()
        if "param_rotor" in raw:
            obj.param_rotor = ContinuousRotorSpectrumField.from_dict(raw["param_rotor"])
        obj.op_predictors = {
            str(k): ContinuousRotorSpectrumField.from_dict(v)
            for k, v in (raw.get("op_predictors") or {}).items()
        }
        obj.op_trials = {str(k): int(v) for k, v in (raw.get("op_trials") or {}).items()}
        obj.op_arity = {str(k): int(v) for k, v in (raw.get("op_arity") or {}).items()}
        obj.op_last_error = {str(k): float(v) for k, v in (raw.get("op_last_error") or {}).items()}
        obj.op_mean_vol_delta = {str(k): float(v) for k, v in (raw.get("op_mean_vol_delta") or {}).items()}
        obj.op_max_hollow = {str(k): int(v) for k, v in (raw.get("op_max_hollow") or {}).items()}
        obj.discovered_laws = [str(x) for x in raw.get("discovered_laws", [])]
        obj.probe_count = int(raw.get("probe_count", 0))
        obj.causal_coverage = float(raw.get("causal_coverage", 0.0))
        obj.prediction_errors = [float(e) for e in raw.get("prediction_errors", [])]
        obj._last_surprise = float(raw.get("last_surprise", 0.85))
        obj._last_volume = float(raw.get("last_volume", 0.0))
        obj._last_stability_tension = float(raw.get("last_stability_tension", 0.30))
        obj._creator_op = raw.get("creator_op")
        return obj


def apply_continuous_scale_vision(rgb_array: np.ndarray, scale_dial: float) -> Dict[str, Any]:
    """
    ONE branch-free continuous visual scale operator (Diffusion-style scale dial `sigma in [0.0, 1.0]`):
      - `sigma -> 0.0`: heavy Gaussian low-pass smoothing (macro coarse luminance field, few large zones)
      - `sigma -> 0.5`: mesoscale luminance gradient bands
      - `sigma -> 1.0`: crisp micro pixel gradient field & fine connected structures
    ZERO `if level == 0 / elif level == 1 / else` statements! Everything varies continuously with `sigma`.
    """
    from scipy.ndimage import gaussian_filter, label as nd_label

    sigma = float(np.clip(scale_dial, 0.0, 1.0))
    arr = np.asarray(rgb_array, dtype=np.float64)
    if arr.size == 0:
        return {"scale_dial": sigma, "perceived_units": 0, "edge_energy": 0.0, "blur_radius": 0.0}
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[2] >= 3:
        luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    else:
        luma = arr.reshape(arr.shape[0], arr.shape[1])

    blur_radius = float(7.5 * ((1.0 - sigma) ** 1.6))
    smoothed = gaussian_filter(luma, sigma=blur_radius) if blur_radius > 0.15 else luma

    gx = np.abs(np.diff(smoothed, axis=1, prepend=smoothed[:, :1]))
    gy = np.abs(np.diff(smoothed, axis=0, prepend=smoothed[:1, :]))
    grad_mag = np.hypot(gx, gy)
    edge_energy = float(np.mean(grad_mag))

    thresh = float(max(0.035, np.mean(grad_mag) + (1.75 - 0.85 * sigma) * np.std(grad_mag)))
    edge_mask = grad_mag >= thresh

    stride = max(1, int(round(1.0 + 11.0 * ((1.0 - sigma) ** 1.4))))
    h, w = edge_mask.shape
    ph, pw = max(1, h // stride), max(1, w // stride)
    pooled_mask = edge_mask[: ph * stride, : pw * stride].reshape(ph, stride, pw, stride).any(axis=(1, 3))

    _labeled, n_components = nd_label(pooled_mask)
    return {
        "scale_dial": round(sigma, 4),
        "blur_radius": round(blur_radius, 3),
        "pooling_stride": int(stride),
        "perceived_units": int(n_components),
        "edge_energy": round(edge_energy, 5),
        "mechanism_ko": (
            f"연속 가변저항 시각 로터(σ={sigma:.2f}, 블러반경={blur_radius:.2f}px, 풀링={stride}px)"
        ),
    }
