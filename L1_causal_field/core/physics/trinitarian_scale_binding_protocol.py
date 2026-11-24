"""
ELYSIA Trinitarian Scale-Binding Protocol & Coupled Equality-Manifold Engine (Phase 8)
===================================================================================

Implements the recursive Trinitarian Scale-Binding Protocol bridging all strata
of causal existence as a living, self-observing hierarchy:

    Lower Subsystem (Values / Formula Equality)
        <-> Bidirectional Equality-Bond Membrane (Mechanism & Conservation)
        <-> Upper Self-Observing Rotor (Fences, Intent, Sensitivity & Explanation)

Key Principles (c:\\Elysia\\AGENTS.md):
1. No dead numbers: Every quantity lives inside a network of bidirectional
   equality (`LHS == RHS`) and boundary-fence (`LHS <= RHS`, `LHS >= RHS`) bonds.
   Turning any Variable-Resistor Dial propagates physically across all connected
   tiers via Principle of Least Action (Levenberg-Marquardt variational
   relaxation on the Action functional A(x) = 1/2 ||r(x)||^2).
2. Scale-consistent Trinitarian Discernment:
   - Layer 0 (`VALUE`): Variable-resistor dials, state quantities, fixed boundaries.
   - Layer 1 (`FORMULA`): Local lower-subsystem equality rules.
   - Layer 2 (`MECHANISM`): Temporal/metabolic/narrative process continuity.
   - Layer 3 (`UPPER_ROTOR`): Whole-organism / narrative safety fences & intent.
   Crucially, an action can be 100% locally valid at Layer 1 ("cancer cell
   mitosis", "contextless stat exploit", "circuit overheating") while violating
   Layer 2/3 fences. The Upper Rotor observes this divergence, explains *why*,
   computes exact causal sensitivities via the Implicit Function Theorem
   (S = -J_free^+ J_pinned), prescribes counterfactual dial adjustments, and
   can downward-regulate the dials into harmonious convergence.
3. 4 Continuities & 4-Axis Self-Boundary:
   - Tracks Internal Origin (laws/rotors), Relational External Influence
     (dial turns / tampers), Boundary Ignorance (underdetermined nullspace DoF
     from Jacobian SVD), and Upper Regulation / Criteria Alignment.
   - Projects live quantities, bonds, and broken-bond beam tearing directly into
     Elysia's `CausalField` substrate with consistent 4D voxel tensors.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.topology.causal_scale_rotor import (
    CausalScaleRotorField,
    RotorNode,
    ScaleRotor,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


class ScaleTier(IntEnum):
    """Hierarchical strata unified by the Trinitarian Scale-Binding Protocol."""

    VALUE = 0        # Atomic quantities & Variable-Resistor Dials
    FORMULA = 1      # Lower-subsystem local equality laws (LHS == RHS)
    MECHANISM = 2    # Temporal, metabolic, or narrative process continuity
    UPPER_ROTOR = 3  # Organism/narrative fences, intent, & self-explanation


class PinMode(str, Enum):
    """Causal binding mode of a quantity in the coupled manifold."""

    FREE = "free"                 # Flows freely along equality bonds to minimize Action
    DIAL = "dial"                 # Variable-resistor dial held by agent/environment
    FIXED_BOUNDARY = "fixed"      # Structural parameter or environmental boundary
    FORCED_TAMPER = "tamper"      # Forced injection ignoring causal bonds (cheat/glitch)


class DiscernmentStatus(str, Enum):
    """4-way Upper Rotor causal verdict across lower, middle, and upper strata."""

    HARMONIOUS_CONVERGENCE = "harmonious_convergence"
    INTENT_DIVERGENCE_LAWFUL = "intent_divergence_lawful"
    LOCAL_VALID_UPPER_VIOLATION = "local_valid_upper_violation"
    STRUCTURAL_NON_CAUSAL_RUPTURE = "structural_non_causal_rupture"


@dataclass
class CoupledQuantity:
    """A scalar physical, biological, or narrative quantity living on a ScaleTier."""

    name: str
    value: float
    tier: ScaleTier = ScaleTier.VALUE
    unit: str = ""
    role: str = "state"  # "dial" | "cause" | "process" | "outcome" | "fence_metric" | "parameter"
    pin_mode: PinMode = PinMode.FREE
    min_bound: Optional[float] = None
    max_bound: Optional[float] = None
    phase_rad: float = 0.0
    previous_value: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        self.value = float(self.value)
        if self.previous_value == 0.0:
            self.previous_value = self.value

    def copy(self) -> "CoupledQuantity":
        return CoupledQuantity(
            name=self.name,
            value=self.value,
            tier=self.tier,
            unit=self.unit,
            role=self.role,
            pin_mode=self.pin_mode,
            min_bound=self.min_bound,
            max_bound=self.max_bound,
            phase_rad=self.phase_rad,
            previous_value=self.previous_value,
            description=self.description,
        )

    def clamp_value(self, raw_val: float) -> float:
        val = float(raw_val)
        if self.min_bound is not None:
            val = max(self.min_bound, val)
        if self.max_bound is not None:
            val = min(self.max_bound, val)
        return val

    def update_value(self, new_val: float, respect_bounds: bool = True) -> float:
        old_val = float(self.value)
        updated = self.clamp_value(new_val) if respect_bounds else float(new_val)
        self.previous_value = old_val
        self.value = updated
        d_phase = math.atan2(updated - old_val, max(1.0, abs(old_val)))
        self.phase_rad = (self.phase_rad + d_phase) % (2.0 * math.pi)
        return updated


@dataclass
class EqualityBond:
    """
    Bidirectional causal constraint coupling two or more quantities:
    - `equality`: lhs(x) == rhs(x)  -> residual = lhs - rhs
    - `fence_le`: lhs(x) <= rhs(x)  -> residual = max(0, lhs - rhs)
    - `fence_ge`: lhs(x) >= rhs(x)  -> residual = max(0, rhs - lhs)
    """

    name: str
    tier: ScaleTier
    variables: Tuple[str, ...]
    lhs_fn: Callable[[Dict[str, float]], float]
    rhs_fn: Callable[[Dict[str, float]], float]
    lhs_expr: str
    rhs_expr: str
    kind: str = "equality"  # "equality" | "fence_le" | "fence_ge"
    weight: float = 1.0
    description: str = ""

    def evaluate(self, values: Dict[str, float]) -> Tuple[float, float, float]:
        lhs_val = float(self.lhs_fn(values))
        rhs_val = float(self.rhs_fn(values))
        if self.kind == "equality":
            residual = lhs_val - rhs_val
        elif self.kind == "fence_le":
            residual = max(0.0, lhs_val - rhs_val)
        elif self.kind == "fence_ge":
            residual = max(0.0, rhs_val - lhs_val)
        else:
            raise ValueError(f"Unknown EqualityBond kind '{self.kind}' on bond '{self.name}'")
        return lhs_val, rhs_val, residual

    def is_satisfied(self, values: Dict[str, float], tol: float = 1e-6) -> bool:
        _, _, residual = self.evaluate(values)
        return abs(residual) <= tol

    def relation_symbol(self) -> str:
        if self.kind == "equality":
            return "=="
        if self.kind == "fence_le":
            return "<="
        return ">="


@dataclass
class IntentTarget:
    """Teleological intent or goal observed by the Upper Rotor."""

    name: str
    variable: str
    target_value: float
    tolerance: float = 1e-3
    relation: str = "equal"  # "equal" | "at_most" | "at_least"
    description: str = ""

    def evaluate(self, values: Dict[str, float]) -> Tuple[float, float, bool]:
        actual = float(values[self.variable])
        target = float(self.target_value)
        if self.relation == "equal":
            gap = actual - target
            satisfied = abs(gap) <= self.tolerance
        elif self.relation == "at_most":
            gap = max(0.0, actual - target)
            satisfied = gap <= self.tolerance
        elif self.relation == "at_least":
            gap = max(0.0, target - actual)
            satisfied = gap <= self.tolerance
        else:
            raise ValueError(f"Unknown IntentTarget relation '{self.relation}'")
        return actual, float(gap), bool(satisfied)


@dataclass
class RelaxationReport:
    """Telemetry returned by the Principle-of-Least-Action manifold solver."""

    converged: bool
    iterations: int
    initial_action: float
    final_action: float
    max_residual: float
    free_variables: List[str]
    nullspace_dof: int


@dataclass
class UpperRotorVerdict:
    """Full self-observational diagnosis produced by the Upper Self-Rotor."""

    status: DiscernmentStatus
    lower_subsystem_verdict: bool
    mechanism_layer_verdict: bool
    upper_fence_verdict: bool
    intent_satisfied: bool
    total_action: float
    tier_actions: Dict[str, float]
    broken_bonds: List[Dict[str, Any]]
    intent_gaps: List[Dict[str, Any]]
    causal_chains: List[List[str]]
    sensitivities: Dict[str, Dict[str, float]]
    prescribed_adjustments: Dict[str, float]
    nullspace_dof: int
    chromatic_balance: Dict[str, float]
    four_axis_attribution: Dict[str, Any]
    explanation: str


class TrinitarianScaleRotorSystem:
    """
    Unified multi-scale causal manifold implementing:
        Lower Formula Subsystem <-> Mechanism Membrane <-> Upper Self-Rotor.
    """

    def __init__(self, name: str = "trinitarian_system") -> None:
        self.name = name
        self.quantities: Dict[str, CoupledQuantity] = {}
        self.bonds: Dict[str, EqualityBond] = {}
        self.intents: Dict[str, IntentTarget] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_changed_dials: List[str] = []
        self._last_nullspace_dof: int = 0
        self.scale_rotors = self._build_scale_rotors()

    @staticmethod
    def _build_scale_rotors(
        rates: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
        phases: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    ) -> CausalScaleRotorField:
        nodes = [
            RotorNode(
                node_id="tier_0_value",
                rotor=ScaleRotor(angular_rate=float(rates[0]), phase=float(phases[0]), radius=1.0, scale=0.25),
                parent_id=None,
            ),
            RotorNode(
                node_id="tier_1_formula",
                rotor=ScaleRotor(angular_rate=float(rates[1]), phase=float(phases[1]), radius=0.8, scale=0.50),
                parent_id="tier_0_value",
            ),
            RotorNode(
                node_id="tier_2_mechanism",
                rotor=ScaleRotor(angular_rate=float(rates[2]), phase=float(phases[2]), radius=0.6, scale=0.75),
                parent_id="tier_1_formula",
            ),
            RotorNode(
                node_id="tier_3_upper_rotor",
                rotor=ScaleRotor(angular_rate=float(rates[3]), phase=float(phases[3]), radius=0.4, scale=1.00),
                parent_id="tier_2_mechanism",
            ),
        ]
        return CausalScaleRotorField(nodes=nodes)

    def _refresh_scale_rotors(self) -> None:
        """Couples current tier residuals & quantity velocities into nested ScaleRotors."""
        vals = self.values_dict()
        rates: List[float] = []
        phases: List[float] = []
        for tier in ScaleTier:
            tier_qs = [q for q in self.quantities.values() if q.tier == tier]
            tier_bs = [b for b in self.bonds.values() if b.tier == tier]
            mean_vel = (
                float(np.mean([abs(q.value - q.previous_value) for q in tier_qs]))
                if tier_qs
                else 0.0
            )
            mean_res = (
                float(np.mean([abs(b.evaluate(vals)[2]) for b in tier_bs]))
                if tier_bs
                else 0.0
            )
            mean_ph = (
                float(np.mean([q.phase_rad for q in tier_qs]))
                if tier_qs
                else (mean_res % (2.0 * math.pi))
            )
            rates.append(max(0.05, 0.25 * (float(tier) + 1.0) + mean_vel))
            phases.append((mean_ph + 0.1 * mean_res) % (2.0 * math.pi))

        self.scale_rotors = self._build_scale_rotors(rates=rates, phases=phases)
        self.scale_rotors.advance(float(len(self.history)))

    def add_quantity(self, quantity: CoupledQuantity) -> None:
        self.quantities[quantity.name] = quantity

    def add_bond(self, bond: EqualityBond) -> None:
        for var in bond.variables:
            if var not in self.quantities:
                raise KeyError(f"Bond '{bond.name}' references unknown quantity '{var}'")
        self.bonds[bond.name] = bond

    def add_intent(self, intent: IntentTarget) -> None:
        if intent.variable not in self.quantities:
            raise KeyError(f"Intent '{intent.name}' references unknown quantity '{intent.variable}'")
        self.intents[intent.name] = intent

    def values_dict(self) -> Dict[str, float]:
        return {k: float(q.value) for k, q in self.quantities.items()}

    def clone(self) -> "TrinitarianScaleRotorSystem":
        """Creates an isolated counterfactual sandbox copy sharing immutable bonds."""
        copy_sys = TrinitarianScaleRotorSystem(name=f"{self.name}_sandbox")
        for name, q in self.quantities.items():
            copy_sys.quantities[name] = q.copy()
        copy_sys.bonds = dict(self.bonds)
        copy_sys.intents = dict(self.intents)
        copy_sys.last_changed_dials = list(self.last_changed_dials)
        copy_sys._last_nullspace_dof = self._last_nullspace_dof
        return copy_sys

    def _collect_bonds(
        self,
        active_tiers: Optional[Sequence[ScaleTier]] = None,
        extra_bonds: Optional[Sequence[EqualityBond]] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[EqualityBond, float]]:
        tier_set = set(active_tiers) if active_tiers is not None else None
        selected: List[Tuple[EqualityBond, float]] = []
        for bond in self.bonds.values():
            if tier_set is not None and bond.tier not in tier_set:
                continue
            w = weight_overrides.get(bond.name, bond.weight) if weight_overrides else bond.weight
            selected.append((bond, float(w)))
        if extra_bonds:
            for bond in extra_bonds:
                w = weight_overrides.get(bond.name, bond.weight) if weight_overrides else bond.weight
                selected.append((bond, float(w)))
        return selected

    def _eval_vector_and_action(
        self,
        vals: Dict[str, float],
        bond_weights: Sequence[Tuple[EqualityBond, float]],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        weighted = np.empty(len(bond_weights), dtype=np.float64)
        unweighted = np.empty(len(bond_weights), dtype=np.float64)
        for idx, (bond, w) in enumerate(bond_weights):
            _, _, res = bond.evaluate(vals)
            unweighted[idx] = res
            weighted[idx] = math.sqrt(max(1e-12, w)) * res
        action = 0.5 * float(np.dot(weighted, weighted))
        return weighted, unweighted, action

    def _jacobian_subset(
        self,
        vals: Dict[str, float],
        var_names: Sequence[str],
        bond_weights: Sequence[Tuple[EqualityBond, float]],
    ) -> np.ndarray:
        """Central finite-difference Jacobian d(weighted_residual) / d(var)."""
        n_b = len(bond_weights)
        n_v = len(var_names)
        jac = np.zeros((n_b, n_v), dtype=np.float64)
        if n_b == 0 or n_v == 0:
            return jac

        scratch = dict(vals)
        for j, name in enumerate(var_names):
            x0 = vals[name]
            h = 1e-6 * max(1.0, abs(x0))
            scratch[name] = x0 + h
            r_plus, _, _ = self._eval_vector_and_action(scratch, bond_weights)
            scratch[name] = x0 - h
            r_minus, _, _ = self._eval_vector_and_action(scratch, bond_weights)
            scratch[name] = x0
            jac[:, j] = (r_plus - r_minus) / (2.0 * h)
        return jac

    def _clamp_free_vector(self, free_names: Sequence[str], vec: np.ndarray) -> np.ndarray:
        clamped = vec.copy()
        for idx, name in enumerate(free_names):
            clamped[idx] = self.quantities[name].clamp_value(float(clamped[idx]))
        return clamped

    def relax_least_action(
        self,
        active_tiers: Optional[Sequence[ScaleTier]] = None,
        free_override: Optional[Sequence[str]] = None,
        extra_bonds: Optional[Sequence[EqualityBond]] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
        max_iter: int = 120,
        tol: float = 1e-10,
    ) -> RelaxationReport:
        """
        Evolves FREE quantities along coupled bonds via the Principle of Least
        Action (dA = 0) using damped Gauss-Newton (Levenberg-Marquardt) with
        nullspace rank tracking for epistemic humility.
        """
        bond_weights = self._collect_bonds(
            active_tiers=active_tiers,
            extra_bonds=extra_bonds,
            weight_overrides=weight_overrides,
        )
        if free_override is not None:
            free_names = [n for n in free_override if n in self.quantities]
        else:
            free_names = [
                n for n, q in self.quantities.items() if q.pin_mode == PinMode.FREE
            ]

        vals = self.values_dict()
        r_vec, raw_vec, init_action = self._eval_vector_and_action(vals, bond_weights)

        if len(free_names) == 0 or len(bond_weights) == 0:
            max_res = float(np.max(np.abs(raw_vec))) if len(raw_vec) else 0.0
            self._last_nullspace_dof = 0
            return RelaxationReport(
                converged=max_res <= tol,
                iterations=0,
                initial_action=init_action,
                final_action=init_action,
                max_residual=max_res,
                free_variables=free_names,
                nullspace_dof=0,
            )

        x = np.array([vals[n] for n in free_names], dtype=np.float64)
        curr_action = init_action
        lam = 1e-4
        iterations = 0
        nullspace_dof = 0

        for it in range(max_iter):
            iterations = it + 1
            for idx, n in enumerate(free_names):
                vals[n] = float(x[idx])
            r_vec, raw_vec, curr_action = self._eval_vector_and_action(vals, bond_weights)
            max_raw = float(np.max(np.abs(raw_vec)))
            if max_raw <= tol:
                break

            jac = self._jacobian_subset(vals, free_names, bond_weights)
            grad = jac.T @ r_vec

            # Saddle-point escape guard when gradient vanishes at non-zero action
            if float(np.linalg.norm(grad)) < 1e-12 and curr_action > (tol * tol):
                nudge = np.array([0.01 * max(1.0, abs(v)) for v in x], dtype=np.float64)
                x_nudge = self._clamp_free_vector(free_names, x + nudge)
                vals_nudge = dict(vals)
                for idx, n in enumerate(free_names):
                    vals_nudge[n] = float(x_nudge[idx])
                r_nudge, _, a_nudge = self._eval_vector_and_action(vals_nudge, bond_weights)
                if a_nudge < curr_action:
                    x = x_nudge
                    curr_action = a_nudge
                    continue
                jac = self._jacobian_subset(vals_nudge, free_names, bond_weights)
                grad = jac.T @ r_nudge

            hess = jac.T @ jac
            diag_scale = np.maximum(1.0, np.diag(hess))

            accepted = False
            for _ in range(16):
                a_mat = hess + lam * np.diag(diag_scale)
                try:
                    dx = -np.linalg.solve(a_mat, grad)
                except np.linalg.LinAlgError:
                    dx, _, _, _ = np.linalg.lstsq(a_mat, -grad, rcond=1e-12)

                # Trust-region cap prevents extreme single-step blowups across poles
                step_scale = 1.0
                for _bt in range(6):
                    x_trial = self._clamp_free_vector(free_names, x + step_scale * dx)
                    trial_vals = dict(vals)
                    for idx, n in enumerate(free_names):
                        trial_vals[n] = float(x_trial[idx])
                    _, _, trial_action = self._eval_vector_and_action(trial_vals, bond_weights)
                    if trial_action < curr_action:
                        x = x_trial
                        curr_action = trial_action
                        lam = max(1e-12, lam * 0.3)
                        accepted = True
                        break
                    step_scale *= 0.5

                if accepted:
                    break
                lam = min(1e10, lam * 5.0)
                if lam >= 1e10:
                    break

            if not accepted and lam >= 1e10:
                break

        for idx, n in enumerate(free_names):
            vals[n] = float(x[idx])
        r_final, raw_final, final_action = self._eval_vector_and_action(vals, bond_weights)
        max_final_res = float(np.max(np.abs(raw_final))) if len(raw_final) else 0.0

        final_jac = self._jacobian_subset(vals, free_names, bond_weights)
        if final_jac.size > 0:
            s_vals = np.linalg.svd(final_jac, compute_uv=False)
            thresh = 1e-8 * max(1.0, float(s_vals[0]) if len(s_vals) else 1.0)
            rank = int(np.sum(s_vals > thresh))
            nullspace_dof = max(0, len(free_names) - rank)

        for n in free_names:
            self.quantities[n].update_value(vals[n], respect_bounds=True)

        self._last_nullspace_dof = nullspace_dof
        self.history.append(
            {
                "event": "relax_least_action",
                "dials": list(self.last_changed_dials),
                "final_action": final_action,
                "max_residual": max_final_res,
                "nullspace_dof": nullspace_dof,
            }
        )
        self._refresh_scale_rotors()

        return RelaxationReport(
            converged=max_final_res <= tol,
            iterations=iterations,
            initial_action=init_action,
            final_action=final_action,
            max_residual=max_final_res,
            free_variables=free_names,
            nullspace_dof=nullspace_dof,
        )

    def turn_dial(
        self,
        name: str,
        new_value: float,
        relax: bool = True,
        active_tiers: Optional[Sequence[ScaleTier]] = None,
    ) -> RelaxationReport:
        """
        Turns a Variable-Resistor Dial (or cause) and naturally flows consequences
        through Layer 1 (FORMULA) and Layer 2 (MECHANISM) equality laws by default,
        preserving genuine physical consequences for the Upper Rotor to inspect.
        """
        if name not in self.quantities:
            raise KeyError(f"Unknown quantity '{name}'")
        q = self.quantities[name]
        q.pin_mode = PinMode.DIAL
        q.update_value(new_value, respect_bounds=True)
        self.last_changed_dials = [name]

        if not relax:
            vals = self.values_dict()
            tiers = active_tiers if active_tiers is not None else [ScaleTier.FORMULA, ScaleTier.MECHANISM]
            bw = self._collect_bonds(active_tiers=tiers)
            _, raw, action = self._eval_vector_and_action(vals, bw)
            self._refresh_scale_rotors()
            return RelaxationReport(
                converged=True,
                iterations=0,
                initial_action=action,
                final_action=action,
                max_residual=float(np.max(np.abs(raw))) if len(raw) else 0.0,
                free_variables=[],
                nullspace_dof=self._last_nullspace_dof,
            )

        tiers = active_tiers if active_tiers is not None else [ScaleTier.FORMULA, ScaleTier.MECHANISM]
        return self.relax_least_action(active_tiers=tiers)

    def pin_value(
        self,
        name: str,
        value: float,
        mode: PinMode = PinMode.FIXED_BOUNDARY,
        relax: bool = True,
    ) -> RelaxationReport:
        if name not in self.quantities:
            raise KeyError(f"Unknown quantity '{name}'")
        q = self.quantities[name]
        q.pin_mode = mode
        q.update_value(value, respect_bounds=True)
        self.last_changed_dials = [name]
        if relax:
            return self.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        vals = self.values_dict()
        bw = self._collect_bonds(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        _, raw, action = self._eval_vector_and_action(vals, bw)
        self._refresh_scale_rotors()
        return RelaxationReport(
            converged=True,
            iterations=0,
            initial_action=action,
            final_action=action,
            max_residual=float(np.max(np.abs(raw))) if len(raw) else 0.0,
            free_variables=[],
            nullspace_dof=self._last_nullspace_dof,
        )

    def inject_tamper(self, name: str, forced_value: float) -> None:
        """
        Simulates a discrete/non-causal external injection (e.g. direct RAM/stat
        cheat) that forces a variable without relaxing connected equality bonds.
        """
        if name not in self.quantities:
            raise KeyError(f"Unknown quantity '{name}'")
        q = self.quantities[name]
        q.pin_mode = PinMode.FORCED_TAMPER
        q.update_value(forced_value, respect_bounds=False)
        self.last_changed_dials = [name]
        self.history.append(
            {"event": "inject_tamper", "variable": name, "forced_value": float(forced_value)}
        )
        self._refresh_scale_rotors()

    def observe_lower_perspective(self, tol: float = 1e-6) -> Dict[str, Any]:
        """
        Returns ONLY what an isolated Layer-1 (`FORMULA`) subsystem perceives.
        Explicitly contrasts local syntactic validity against holistic coherence.
        """
        vals = self.values_dict()
        formula_bonds = [b for b in self.bonds.values() if b.tier == ScaleTier.FORMULA]
        checked: List[Dict[str, Any]] = []
        max_res = 0.0
        for bond in formula_bonds:
            lhs_val, rhs_val, res = bond.evaluate(vals)
            max_res = max(max_res, abs(res))
            checked.append(
                {
                    "bond": bond.name,
                    "expression": f"{bond.lhs_expr} {bond.relation_symbol()} {bond.rhs_expr}",
                    "lhs": lhs_val,
                    "rhs": rhs_val,
                    "residual": res,
                    "satisfied": abs(res) <= tol,
                }
            )
        return {
            "lower_valid": all(item["satisfied"] for item in checked),
            "max_residual": float(max_res),
            "checked_bonds": checked,
        }

    def compute_causal_sensitivities(self) -> Dict[str, Dict[str, float]]:
        """
        Applies the Implicit Function Theorem to coupled equality bonds
        (FORMULA + MECHANISM) to obtain exact directional couplings:

            S = d(free) / d(pinned) = - J_free^+ J_pinned
        """
        eq_bonds = [
            (b, 1.0)
            for b in self.bonds.values()
            if b.tier in (ScaleTier.FORMULA, ScaleTier.MECHANISM) and b.kind == "equality"
        ]
        if not eq_bonds:
            return {}

        eq_vars = {v for b, _ in eq_bonds for v in b.variables}
        free_vars = [
            n
            for n, q in self.quantities.items()
            if q.pin_mode == PinMode.FREE and n in eq_vars
        ]
        pinned_vars = [
            n
            for n, q in self.quantities.items()
            if q.pin_mode != PinMode.FREE and n in eq_vars
        ]
        if not free_vars or not pinned_vars:
            return {}

        vals = self.values_dict()
        j_free = self._jacobian_subset(vals, free_vars, eq_bonds)
        j_pinned = self._jacobian_subset(vals, pinned_vars, eq_bonds)
        s_mat = -np.linalg.pinv(j_free, rcond=1e-10) @ j_pinned

        sensitivities: Dict[str, Dict[str, float]] = {}
        for i, f_name in enumerate(free_vars):
            sensitivities[f_name] = {
                p_name: float(s_mat[i, j]) for j, p_name in enumerate(pinned_vars)
            }
        return sensitivities

    def trace_causal_chains(
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        max_chains: int = 12,
    ) -> List[List[str]]:
        """
        BFS across the bipartite (Quantity <-> EqualityBond) topology returning
        human- and machine-readable causal proof paths from sources to targets.
        """
        var_to_bonds: Dict[str, List[EqualityBond]] = {n: [] for n in self.quantities}
        for bond in self.bonds.values():
            for v in bond.variables:
                if v in var_to_bonds:
                    var_to_bonds[v].append(bond)

        chains: List[List[str]] = []
        seen_signatures: set[Tuple[str, ...]] = set()
        target_set = set(targets)

        for src in sources:
            if src not in self.quantities:
                continue
            queue: deque[Tuple[str, List[str], set[str]]] = deque([(src, [src], {src})])
            while queue and len(chains) < max_chains:
                curr_var, path, visited_vars = queue.popleft()
                if curr_var in target_set and len(path) > 1:
                    sig = tuple(path)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        chains.append(path)
                    continue
                for bond in var_to_bonds.get(curr_var, []):
                    bond_label = f"=[{bond.name}: {bond.lhs_expr} {bond.relation_symbol()} {bond.rhs_expr}]"
                    for nxt in bond.variables:
                        if nxt == curr_var or nxt in visited_vars:
                            continue
                        nxt_path = path + [bond_label, nxt]
                        if nxt in target_set:
                            sig = tuple(nxt_path)
                            if sig not in seen_signatures:
                                seen_signatures.add(sig)
                                chains.append(nxt_path)
                        else:
                            queue.append((nxt, nxt_path, visited_vars | {nxt}))
        return chains

    def _build_intent_bonds(self) -> List[EqualityBond]:
        intent_bonds: List[EqualityBond] = []
        for intent in self.intents.values():
            kind_map = {
                "equal": ("equality", "=="),
                "at_most": ("fence_le", "<="),
                "at_least": ("fence_ge", ">="),
            }
            kind, sym = kind_map[intent.relation]
            v_name = intent.variable
            tgt = float(intent.target_value)
            # Normalize scale so residual sits cleanly on physical scale
            scale = max(1.0, abs(tgt))
            intent_bonds.append(
                EqualityBond(
                    name=f"intent::{intent.name}",
                    tier=ScaleTier.UPPER_ROTOR,
                    variables=(v_name,),
                    lhs_fn=lambda vals, vn=v_name, sc=scale: vals[vn] / sc,
                    rhs_fn=lambda _vals, t=tgt, sc=scale: t / sc,
                    lhs_expr=v_name,
                    rhs_expr=f"{tgt:.4g}",
                    kind=kind,
                    weight=1.0,
                    description=f"Intent {v_name} {sym} {tgt}",
                )
            )
        return intent_bonds

    def _upper_fences_satisfied(self, vals: Dict[str, float], tol: float) -> bool:
        for bond in self.bonds.values():
            if bond.tier in (ScaleTier.MECHANISM, ScaleTier.UPPER_ROTOR):
                if not bond.is_satisfied(vals, tol=tol):
                    return False
        return True

    def prescribe_counterfactual(
        self,
        adjustable_dials: Optional[Sequence[str]] = None,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Uses a counterfactual sandbox clone to compute exact least-action dial
        adjustments (or tamper healing targets) that restore coherence.
        Safety/homeostasis fences ALWAYS take precedence over unsafe intents.
        """
        tampered = [
            n for n, q in self.quantities.items() if q.pin_mode == PinMode.FORCED_TAMPER
        ]
        if tampered:
            sandbox = self.clone()
            for name in tampered:
                sandbox.quantities[name].pin_mode = PinMode.FREE
            sandbox.relax_least_action(
                active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM],
                tol=1e-10,
            )
            return {name: sandbox.quantities[name].value for name in tampered}

        dials = (
            [d for d in adjustable_dials if d in self.quantities]
            if adjustable_dials is not None
            else [n for n, q in self.quantities.items() if q.pin_mode == PinMode.DIAL]
        )
        if not dials:
            return {}

        sandbox = self.clone()
        for d in dials:
            sandbox.quantities[d].pin_mode = PinMode.FREE

        intent_bonds = self._build_intent_bonds()
        fence_weights = {
            b.name: 100.0
            for b in sandbox.bonds.values()
            if b.tier == ScaleTier.UPPER_ROTOR or b.kind != "equality"
        }

        sandbox.relax_least_action(
            active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM, ScaleTier.UPPER_ROTOR],
            extra_bonds=intent_bonds,
            weight_overrides=fence_weights,
            tol=1e-10,
        )

        # If reckless intent pulled against safety fence and violated it,
        # re-solve enforcing physical/organism/narrative fences strictly first.
        if not self._upper_fences_satisfied(sandbox.values_dict(), tol=tol):
            sandbox = self.clone()
            for d in dials:
                sandbox.quantities[d].pin_mode = PinMode.FREE
            sandbox.relax_least_action(
                active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM, ScaleTier.UPPER_ROTOR],
                weight_overrides=fence_weights,
                tol=1e-10,
            )

        return {d: float(sandbox.quantities[d].value) for d in dials}

    def observe_and_discern(
        self,
        tol: float = 1e-5,
        adjustable_dials: Optional[Sequence[str]] = None,
    ) -> UpperRotorVerdict:
        """
        Holo-observational self-diagnosis by the Upper Rotor across Value,
        Formula, Mechanism, and Upper Fence/Intent tiers.
        """
        vals = self.values_dict()
        tier_actions: Dict[str, float] = {
            "FORMULA": 0.0,
            "MECHANISM": 0.0,
            "UPPER_ROTOR": 0.0,
        }
        broken_bonds: List[Dict[str, Any]] = []
        lower_ok = True
        mech_ok = True
        upper_ok = True

        for bond in self.bonds.values():
            lhs_val, rhs_val, res = bond.evaluate(vals)
            action_contrib = 0.5 * bond.weight * (res * res)
            if bond.tier == ScaleTier.FORMULA:
                tier_actions["FORMULA"] += action_contrib
                if abs(res) > tol:
                    lower_ok = False
            elif bond.tier == ScaleTier.MECHANISM:
                tier_actions["MECHANISM"] += action_contrib
                if abs(res) > tol:
                    mech_ok = False
            elif bond.tier == ScaleTier.UPPER_ROTOR:
                tier_actions["UPPER_ROTOR"] += action_contrib
                if abs(res) > tol:
                    upper_ok = False

            if abs(res) > tol:
                broken_bonds.append(
                    {
                        "bond": bond.name,
                        "tier": bond.tier.name,
                        "kind": bond.kind,
                        "expression": f"{bond.lhs_expr} {bond.relation_symbol()} {bond.rhs_expr}",
                        "lhs": lhs_val,
                        "rhs": rhs_val,
                        "residual": res,
                        "variables": list(bond.variables),
                        "description": bond.description,
                    }
                )

        intent_gaps: List[Dict[str, Any]] = []
        intent_ok = True
        for intent in self.intents.values():
            actual, gap, sat = intent.evaluate(vals)
            if not sat:
                intent_ok = False
                intent_gaps.append(
                    {
                        "intent": intent.name,
                        "variable": intent.variable,
                        "relation": intent.relation,
                        "actual": actual,
                        "target": intent.target_value,
                        "gap": gap,
                        "description": intent.description,
                    }
                )

        tampered_vars = [
            n for n, q in self.quantities.items() if q.pin_mode == PinMode.FORCED_TAMPER
        ]
        eq_broken = any(item["kind"] == "equality" for item in broken_bonds)

        if not lower_ok or (tampered_vars and eq_broken):
            status = DiscernmentStatus.STRUCTURAL_NON_CAUSAL_RUPTURE
        elif not mech_ok or not upper_ok:
            status = DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
        elif not intent_ok:
            status = DiscernmentStatus.INTENT_DIVERGENCE_LAWFUL
        else:
            status = DiscernmentStatus.HARMONIOUS_CONVERGENCE

        sensitivities = self.compute_causal_sensitivities()
        sources = list(self.last_changed_dials) or [
            n for n, q in self.quantities.items() if q.pin_mode in (PinMode.DIAL, PinMode.FORCED_TAMPER)
        ]
        targets = [i.variable for i in self.intents.values()]
        for b in broken_bonds:
            targets.extend(b["variables"])
        if not targets:
            targets = [n for n, q in self.quantities.items() if q.role == "outcome"]
        causal_chains = self.trace_causal_chains(sources=sources, targets=list(dict.fromkeys(targets)))

        prescribed: Dict[str, float] = {}
        if status != DiscernmentStatus.HARMONIOUS_CONVERGENCE:
            prescribed = self.prescribe_counterfactual(
                adjustable_dials=adjustable_dials,
                tol=tol,
            )

        total_action = sum(tier_actions.values())
        free_qs = [q for q in self.quantities.values() if q.pin_mode == PinMode.FREE]
        flux_val = (
            float(np.mean([abs(q.value - q.previous_value) / max(1.0, abs(q.value)) for q in free_qs]))
            if free_qs
            else 0.0
        )
        red_flux = min(1.0, flux_val)
        blue_order = 1.0 / (1.0 + total_action)
        yellow_entropy = min(1.0, total_action / (1.0 + total_action) + 0.1 * self._last_nullspace_dof)
        c_sum = max(1e-9, red_flux + blue_order + yellow_entropy)
        chromatic_balance = {
            "red_flux": red_flux / c_sum,
            "blue_order": blue_order / c_sum,
            "yellow_entropy": yellow_entropy / c_sum,
        }

        four_axis = {
            "internal_origin": f"{len(self.bonds)} equality/fence bonds across Tiers 1..3 under Least Action",
            "external_dial_or_tamper": {
                "dials": {
                    n: q.value for n, q in self.quantities.items() if q.pin_mode == PinMode.DIAL
                },
                "tampered": {
                    n: q.value for n, q in self.quantities.items() if q.pin_mode == PinMode.FORCED_TAMPER
                },
            },
            "boundary_ignorance": (
                f"Underdetermined nullspace DoF={self._last_nullspace_dof} (infinite solution manifold)"
                if self._last_nullspace_dof > 0
                else "Well-determined equality manifold (nullspace DoF=0); uncertainty limited to unmodeled external disturbances."
            ),
            "upper_regulation": (
                "No adjustment needed"
                if not prescribed
                else f"Prescribed counterfactual equilibrium targets: {prescribed}"
            ),
        }

        explanation = self._format_explanation(
            status=status,
            lower_ok=lower_ok,
            mech_ok=mech_ok,
            upper_ok=upper_ok,
            intent_ok=intent_ok,
            broken_bonds=broken_bonds,
            intent_gaps=intent_gaps,
            causal_chains=causal_chains,
            sensitivities=sensitivities,
            prescribed=prescribed,
        )

        return UpperRotorVerdict(
            status=status,
            lower_subsystem_verdict=lower_ok,
            mechanism_layer_verdict=mech_ok,
            upper_fence_verdict=upper_ok,
            intent_satisfied=intent_ok,
            total_action=float(total_action),
            tier_actions=tier_actions,
            broken_bonds=broken_bonds,
            intent_gaps=intent_gaps,
            causal_chains=causal_chains,
            sensitivities=sensitivities,
            prescribed_adjustments=prescribed,
            nullspace_dof=int(self._last_nullspace_dof),
            chromatic_balance=chromatic_balance,
            four_axis_attribution=four_axis,
            explanation=explanation,
        )

    def _format_explanation(
        self,
        status: DiscernmentStatus,
        lower_ok: bool,
        mech_ok: bool,
        upper_ok: bool,
        intent_ok: bool,
        broken_bonds: List[Dict[str, Any]],
        intent_gaps: List[Dict[str, Any]],
        causal_chains: List[List[str]],
        sensitivities: Dict[str, Dict[str, float]],
        prescribed: Dict[str, float],
    ) -> str:
        lines: List[str] = [
            f"[{self.name}] Verdict={status.value} | "
            f"Layer1(Formula)={'VALID' if lower_ok else 'BROKEN'}, "
            f"Layer2(Mechanism)={'VALID' if mech_ok else 'BROKEN'}, "
            f"Layer3(UpperFence)={'VALID' if upper_ok else 'BROKEN'}, "
            f"Intent={'MET' if intent_ok else 'DIVERGED'}."
        ]
        if status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION and lower_ok:
            lines.append(
                "Trinitarian Contrast: Lower subsystem passed 100% of local Layer-1 rules, "
                "yet Upper Rotor detects boundary/narrative fence rupture at macro scale."
            )
        elif status == DiscernmentStatus.STRUCTURAL_NON_CAUSAL_RUPTURE:
            lines.append(
                "Causal Rupture: An equality law was violated by an uncoupled forced value rather than continuous dial flow."
            )
        if broken_bonds:
            for item in broken_bonds:
                lines.append(
                    f"  - Broken [{item['tier']}] {item['bond']} ({item['expression']}): "
                    f"lhs={item['lhs']:.4g}, rhs={item['rhs']:.4g}, residual={item['residual']:+.4g}"
                )
        if intent_gaps:
            for gap in intent_gaps:
                lines.append(
                    f"  - Intent Gap '{gap['intent']}': {gap['variable']}={gap['actual']:.4g} "
                    f"(target {gap['relation']} {gap['target']:.4g}, gap={gap['gap']:+.4g})"
                )
        if causal_chains:
            lines.append(f"  - Proof Chain: {' '.join(causal_chains[0])}")
        if prescribed:
            adj_str = ", ".join(f"{k} -> {v:.6g}" for k, v in prescribed.items())
            lines.append(f"  - Upper Rotor Prescription: {adj_str}")
        if sensitivities:
            sample_pairs: List[str] = []
            for f_name, deps in list(sensitivities.items())[:2]:
                for d_name, val in list(deps.items())[:2]:
                    sample_pairs.append(f"d({f_name})/d({d_name})={val:+.4g}")
            if sample_pairs:
                lines.append(f"  - Sensitivity Couplings: {', '.join(sample_pairs)}")
        return "\n".join(lines)

    def regulate_to_equilibrium(
        self,
        adjustable_dials: Optional[Sequence[str]] = None,
        tol: float = 1e-5,
    ) -> Tuple[UpperRotorVerdict, UpperRotorVerdict]:
        """
        Closed-loop Upper Rotor governance:
        1. Observes and diagnoses the current state across all tiers.
        2. Strips forced non-causal tampers OR adjusts variable-resistor dials
           along the counterfactual least-action manifold.
        3. Re-relaxes all physical/mechanism equalities and returns `(before, after)`.
        """
        before = self.observe_and_discern(tol=tol, adjustable_dials=adjustable_dials)

        if before.status == DiscernmentStatus.STRUCTURAL_NON_CAUSAL_RUPTURE:
            tampered = [
                n for n, q in self.quantities.items() if q.pin_mode == PinMode.FORCED_TAMPER
            ]
            for name in tampered:
                self.quantities[name].pin_mode = PinMode.FREE
            self.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
        elif before.status in (
            DiscernmentStatus.INTENT_DIVERGENCE_LAWFUL,
            DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION,
        ):
            for dial_name, target_val in before.prescribed_adjustments.items():
                self.quantities[dial_name].pin_mode = PinMode.DIAL
                self.quantities[dial_name].update_value(target_val, respect_bounds=True)
            self.last_changed_dials = list(before.prescribed_adjustments.keys())
            self.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])

        after = self.observe_and_discern(tol=tol, adjustable_dials=adjustable_dials)
        return before, after


# ---------------------------------------------------------------------------
# Built-in Real-World Multi-Scale Builders
# ---------------------------------------------------------------------------


def build_variable_resistor_circuit_system() -> TrinitarianScaleRotorSystem:
    """
    Builder 1: Coupled Variable-Resistor Electrical -> Thermal -> Safety Circuit.
    Turning Resistance dial `R` flows continuously through Ohm's law, Joule
    heating, and thermal rise into the Upper Rotor's thermal safety fence.
    """
    sys = TrinitarianScaleRotorSystem(name="variable_resistor_circuit")

    sys.add_quantity(CoupledQuantity("V", 12.0, ScaleTier.VALUE, "V", "fixed", PinMode.FIXED_BOUNDARY, 0.1, 240.0))
    sys.add_quantity(CoupledQuantity("R", 4.0, ScaleTier.VALUE, "Ohm", "dial", PinMode.DIAL, 0.2, 1000.0))
    sys.add_quantity(CoupledQuantity("dt", 5.0, ScaleTier.VALUE, "s", "fixed", PinMode.FIXED_BOUNDARY, 0.1, 3600.0))
    sys.add_quantity(CoupledQuantity("T_ambient", 25.0, ScaleTier.VALUE, "C", "fixed", PinMode.FIXED_BOUNDARY, -50.0, 200.0))
    sys.add_quantity(CoupledQuantity("R_th", 0.2, ScaleTier.VALUE, "C/J", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 10.0))
    sys.add_quantity(CoupledQuantity("T_max", 85.0, ScaleTier.UPPER_ROTOR, "C", "fence_metric", PinMode.FIXED_BOUNDARY, 0.0, 500.0))

    sys.add_quantity(CoupledQuantity("I", 3.0, ScaleTier.FORMULA, "A", "process", PinMode.FREE, 0.0, 1000.0))
    sys.add_quantity(CoupledQuantity("P", 36.0, ScaleTier.FORMULA, "W", "outcome", PinMode.FREE, 0.0, 100000.0))
    sys.add_quantity(CoupledQuantity("E_heat", 180.0, ScaleTier.MECHANISM, "J", "process", PinMode.FREE, 0.0, 1e7))
    sys.add_quantity(CoupledQuantity("T_final", 61.0, ScaleTier.MECHANISM, "C", "outcome", PinMode.FREE, -50.0, 5000.0))

    sys.add_bond(
        EqualityBond(
            name="ohm_law",
            tier=ScaleTier.FORMULA,
            variables=("V", "I", "R"),
            lhs_fn=lambda v: v["V"],
            rhs_fn=lambda v: v["I"] * v["R"],
            lhs_expr="V",
            rhs_expr="I * R",
            description="Ohm's Law voltage-current-resistance equality",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="power_law",
            tier=ScaleTier.FORMULA,
            variables=("P", "V", "I"),
            lhs_fn=lambda v: v["P"],
            rhs_fn=lambda v: v["V"] * v["I"],
            lhs_expr="P",
            rhs_expr="V * I",
            description="Electrical power transfer equality",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="joule_heat",
            tier=ScaleTier.MECHANISM,
            variables=("E_heat", "P", "dt"),
            lhs_fn=lambda v: v["E_heat"],
            rhs_fn=lambda v: v["P"] * v["dt"],
            lhs_expr="E_heat",
            rhs_expr="P * dt",
            description="Temporal energy integration across duration dt",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="thermal_rise",
            tier=ScaleTier.MECHANISM,
            variables=("T_final", "T_ambient", "E_heat", "R_th"),
            lhs_fn=lambda v: v["T_final"],
            rhs_fn=lambda v: v["T_ambient"] + v["E_heat"] * v["R_th"],
            lhs_expr="T_final",
            rhs_expr="T_ambient + E_heat * R_th",
            description="Joule heating temperature elevation",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="thermal_safety_fence",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("T_final", "T_max"),
            lhs_fn=lambda v: v["T_final"],
            rhs_fn=lambda v: v["T_max"],
            lhs_expr="T_final",
            rhs_expr="T_max",
            kind="fence_le",
            description="Upper thermal integrity fence preventing circuit burnout",
        )
    )

    sys.add_intent(
        IntentTarget(
            name="target_power",
            variable="P",
            target_value=36.0,
            tolerance=1e-2,
            relation="equal",
            description="Desired nominal output power 36 W",
        )
    )
    sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
    return sys


def build_cellular_homeostasis_system() -> TrinitarianScaleRotorSystem:
    """
    Builder 2: Cellular Mitosis vs Whole-Organism Homeostasis ("Digital Cancer").
    Demonstrates that a cell turning its division rate `r_div` upward passes
    100% of its own Layer-1 rules (`mitosis_law`, `atp_demand_law`) while
    destroying the Upper Organism Rotor's tissue capacity and vascular oxygen fences.
    """
    sys = TrinitarianScaleRotorSystem(name="cellular_homeostasis")

    sys.add_quantity(CoupledQuantity("N_curr", 100.0, ScaleTier.VALUE, "cells", "fixed", PinMode.FIXED_BOUNDARY, 1.0, 1e6))
    sys.add_quantity(CoupledQuantity("r_div", 0.05, ScaleTier.VALUE, "ratio", "dial", PinMode.DIAL, 0.0, 2.0))
    sys.add_quantity(CoupledQuantity("k_atp", 1.0, ScaleTier.VALUE, "u/cell", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 100.0))
    sys.add_quantity(CoupledQuantity("eta_oxphos", 1.0, ScaleTier.VALUE, "eff", "fixed", PinMode.FIXED_BOUNDARY, 0.01, 10.0))
    sys.add_quantity(CoupledQuantity("O2_supply", 140.0, ScaleTier.UPPER_ROTOR, "u", "fence_metric", PinMode.FIXED_BOUNDARY, 1.0, 1e6))
    sys.add_quantity(CoupledQuantity("K_tissue", 125.0, ScaleTier.UPPER_ROTOR, "cells", "fence_metric", PinMode.FIXED_BOUNDARY, 1.0, 1e6))

    sys.add_quantity(CoupledQuantity("N_next", 105.0, ScaleTier.FORMULA, "cells", "outcome", PinMode.FREE, 0.0, 1e6))
    sys.add_quantity(CoupledQuantity("ATP_per_cell", 1.05, ScaleTier.FORMULA, "u/cell", "process", PinMode.FREE, 0.0, 1e4))
    sys.add_quantity(CoupledQuantity("O2_demand", 110.25, ScaleTier.MECHANISM, "u", "outcome", PinMode.FREE, 0.0, 1e7))

    sys.add_bond(
        EqualityBond(
            name="mitosis_law",
            tier=ScaleTier.FORMULA,
            variables=("N_next", "N_curr", "r_div"),
            lhs_fn=lambda v: v["N_next"],
            rhs_fn=lambda v: v["N_curr"] * (1.0 + v["r_div"]),
            lhs_expr="N_next",
            rhs_expr="N_curr * (1 + r_div)",
            description="Local cell division law",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="atp_demand_law",
            tier=ScaleTier.FORMULA,
            variables=("ATP_per_cell", "k_atp", "r_div"),
            lhs_fn=lambda v: v["ATP_per_cell"],
            rhs_fn=lambda v: v["k_atp"] * (1.0 + v["r_div"]),
            lhs_expr="ATP_per_cell",
            rhs_expr="k_atp * (1 + r_div)",
            description="Per-cell mitotic ATP energy requirement",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="tissue_oxygen_flux",
            tier=ScaleTier.MECHANISM,
            variables=("O2_demand", "N_next", "ATP_per_cell", "eta_oxphos"),
            lhs_fn=lambda v: v["O2_demand"],
            rhs_fn=lambda v: (v["N_next"] * v["ATP_per_cell"]) / v["eta_oxphos"],
            lhs_expr="O2_demand",
            rhs_expr="N_next * ATP_per_cell / eta_oxphos",
            description="Whole-tissue vascular oxygen demand",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="tissue_capacity_fence",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("N_next", "K_tissue"),
            lhs_fn=lambda v: v["N_next"],
            rhs_fn=lambda v: v["K_tissue"],
            lhs_expr="N_next",
            rhs_expr="K_tissue",
            kind="fence_le",
            description="Upper organism carrying-capacity boundary",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="vascular_oxygen_fence",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("O2_demand", "O2_supply"),
            lhs_fn=lambda v: v["O2_demand"],
            rhs_fn=lambda v: v["O2_supply"],
            lhs_expr="O2_demand",
            rhs_expr="O2_supply",
            kind="fence_le",
            description="Upper vascular oxygen budget fence preventing hypoxia",
        )
    )

    sys.add_intent(
        IntentTarget(
            name="tissue_maintenance",
            variable="N_next",
            target_value=102.0,
            tolerance=1e-2,
            relation="at_least",
            description="Maintain minimum healthy tissue population >= 102",
        )
    )
    sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
    return sys


def build_narrative_causal_worldline_system() -> TrinitarianScaleRotorSystem:
    """
    Builder 3: Continuous Narrative-Causal Free-Will Worldline.
    Exposes why discrete tick checks (`local_energy_balance`, `effort_cost <= E_present`)
    pass contextless jumps (`claimed_mastery_jump`), whereas Layer 2/3 Narrative
    Continuity (`Past -> Present -> Future`) accepts genuine free-will effort and
    filters out severed outcomes.
    """
    sys = TrinitarianScaleRotorSystem(name="narrative_causal_worldline")

    sys.add_quantity(CoupledQuantity("prep_past", 8.0, ScaleTier.VALUE, "pts", "cause", PinMode.FIXED_BOUNDARY, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("trust_past", 7.0, ScaleTier.VALUE, "pts", "cause", PinMode.FIXED_BOUNDARY, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("E_present", 10.0, ScaleTier.VALUE, "pts", "cause", PinMode.FIXED_BOUNDARY, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("friction", 1.0, ScaleTier.VALUE, "coef", "fixed", PinMode.FIXED_BOUNDARY, 0.1, 50.0))
    sys.add_quantity(CoupledQuantity("S_present", 20.0, ScaleTier.VALUE, "pts", "cause", PinMode.FIXED_BOUNDARY, 0.0, 1000.0))

    sys.add_quantity(CoupledQuantity("effort_cost", 4.0, ScaleTier.VALUE, "pts", "dial", PinMode.DIAL, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("claimed_mastery_jump", 15.2, ScaleTier.UPPER_ROTOR, "pts", "dial", PinMode.DIAL, 0.0, 1000.0))

    sys.add_quantity(CoupledQuantity("E_after", 6.0, ScaleTier.FORMULA, "pts", "process", PinMode.FREE, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("readiness_present", 7.6, ScaleTier.MECHANISM, "pts", "process", PinMode.FREE, 0.0, 100.0))
    sys.add_quantity(CoupledQuantity("causal_mastery_gain", 15.2, ScaleTier.MECHANISM, "pts", "process", PinMode.FREE, 0.0, 1000.0))
    sys.add_quantity(CoupledQuantity("S_future", 35.2, ScaleTier.MECHANISM, "pts", "outcome", PinMode.FREE, 0.0, 2000.0))

    sys.add_bond(
        EqualityBond(
            name="local_energy_balance",
            tier=ScaleTier.FORMULA,
            variables=("E_after", "E_present", "effort_cost"),
            lhs_fn=lambda v: v["E_after"],
            rhs_fn=lambda v: v["E_present"] - v["effort_cost"],
            lhs_expr="E_after",
            rhs_expr="E_present - effort_cost",
            description="Single-tick resource deduction rule",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="local_energy_nonneg",
            tier=ScaleTier.FORMULA,
            variables=("effort_cost", "E_present"),
            lhs_fn=lambda v: v["effort_cost"],
            rhs_fn=lambda v: v["E_present"],
            lhs_expr="effort_cost",
            rhs_expr="E_present",
            kind="fence_le",
            description="Single-tick non-negative resource fence",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="narrative_readiness",
            tier=ScaleTier.MECHANISM,
            variables=("readiness_present", "prep_past", "trust_past"),
            lhs_fn=lambda v: v["readiness_present"],
            rhs_fn=lambda v: 0.6 * v["prep_past"] + 0.4 * v["trust_past"],
            lhs_expr="readiness_present",
            rhs_expr="0.6 * prep_past + 0.4 * trust_past",
            description="Accumulated past preparation and relational trust",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="causal_gain_law",
            tier=ScaleTier.MECHANISM,
            variables=("causal_mastery_gain", "readiness_present", "effort_cost", "friction"),
            lhs_fn=lambda v: v["causal_mastery_gain"],
            rhs_fn=lambda v: (v["readiness_present"] * v["effort_cost"]) / (1.0 + v["friction"]),
            lhs_expr="causal_mastery_gain",
            rhs_expr="readiness_present * effort_cost / (1 + friction)",
            description="Free-will effort transformed through past readiness and world friction",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="worldline_continuity",
            tier=ScaleTier.MECHANISM,
            variables=("S_future", "S_present", "causal_mastery_gain"),
            lhs_fn=lambda v: v["S_future"],
            rhs_fn=lambda v: v["S_present"] + v["causal_mastery_gain"],
            lhs_expr="S_future",
            rhs_expr="S_present + causal_mastery_gain",
            description="Continuous Past -> Present -> Future state worldline",
        )
    )
    sys.add_bond(
        EqualityBond(
            name="narrative_coherence_fence",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("claimed_mastery_jump", "causal_mastery_gain"),
            lhs_fn=lambda v: v["claimed_mastery_jump"],
            rhs_fn=lambda v: v["causal_mastery_gain"],
            lhs_expr="claimed_mastery_jump",
            rhs_expr="causal_mastery_gain",
            kind="fence_le",
            description="Upper narrative horizon forbidding contextless mastery leaps",
        )
    )

    sys.add_intent(
        IntentTarget(
            name="meaningful_growth",
            variable="S_future",
            target_value=30.0,
            tolerance=1e-2,
            relation="at_least",
            description="Sustained narrative growth >= 30.0",
        )
    )
    sys.relax_least_action(active_tiers=[ScaleTier.FORMULA, ScaleTier.MECHANISM])
    return sys


# ---------------------------------------------------------------------------
# CausalField Integration
# ---------------------------------------------------------------------------


def sync_to_causal_field(
    system: TrinitarianScaleRotorSystem,
    verdict: Optional[UpperRotorVerdict] = None,
    causal_field: Optional["CausalField"] = None,
    prefix: str = "bind",
) -> Dict[str, Any]:
    """
    Projects coupled quantities, bidirectional equality bonds, and the Upper
    Rotor apex into Elysia's physical `CausalField` with uniform `(4,)` float32
    voxel tensors so torn equality bonds physically weaken `ConnectivityBeam`s.
    """
    from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

    obs = verdict if verdict is not None else system.observe_and_discern()
    field = causal_field if causal_field is not None else CausalField(dimensions=3)
    vals = system.values_dict()

    q_voxels: Dict[str, str] = {}
    q_list = list(system.quantities.values())
    for idx, q in enumerate(q_list):
        ang = (2.0 * math.pi * idx) / float(max(1, len(q_list)))
        rad = 1.0 + 0.35 * float(q.tier)
        pos = np.array([math.cos(ang) * rad, math.sin(ang) * rad, float(q.tier)], dtype=np.float32)
        vid = f"{prefix}_q_{q.name}"
        field.add_voxel(
            InformationVoxel(
                id=vid,
                content={"quantity": q.name, "value": q.value, "pin_mode": q.pin_mode.value},
                tensor=np.array(
                    [
                        q.value / max(1.0, abs(q.value)),
                        q.value - q.previous_value,
                        float(q.tier) / 3.0,
                        math.sin(q.phase_rad),
                    ],
                    dtype=np.float32,
                ),
                mass=max(1.0, 1.0 + 0.1 * abs(q.value)),
                position=pos[: field.dimensions],
                phase_angle=q.phase_rad,
                scale_level=float(q.tier) + 1.0,
            )
        )
        q_voxels[q.name] = vid

    b_voxels: Dict[str, str] = {}
    beams_linked = 0
    beam_strengths: List[float] = []
    b_list = list(system.bonds.values())
    for idx, bond in enumerate(b_list):
        lhs_val, rhs_val, res = bond.evaluate(vals)
        ang = (2.0 * math.pi * idx) / float(max(1, len(b_list))) + 0.25
        rad = 0.6 + 0.35 * float(bond.tier)
        pos = np.array([math.cos(ang) * rad, math.sin(ang) * rad, float(bond.tier) + 0.2], dtype=np.float32)
        b_vid = f"{prefix}_bond_{bond.name}"
        strength = max(0.01, 1.0 / (1.0 + abs(res)))
        field.add_voxel(
            InformationVoxel(
                id=b_vid,
                content={"bond": bond.name, "residual": res, "kind": bond.kind},
                tensor=np.array(
                    [lhs_val, rhs_val, res, float(bond.tier) / 3.0],
                    dtype=np.float32,
                ),
                mass=max(1.0, 1.0 + abs(res)),
                position=pos[: field.dimensions],
                phase_angle=ang % (2.0 * math.pi),
                scale_level=float(bond.tier) + 1.0,
            )
        )
        b_voxels[bond.name] = b_vid
        for var_name in bond.variables:
            if var_name in q_voxels:
                field.link_voxels(q_voxels[var_name], b_vid, strength=strength)
                beams_linked += 1
                beam_strengths.append(strength)

    apex_vid = f"{prefix}_upper_rotor_apex"
    apex_pos = np.array([0.0, 0.0, 3.5], dtype=np.float32)
    field.add_voxel(
        InformationVoxel(
            id=apex_vid,
            content={"status": obs.status.value, "total_action": obs.total_action},
            tensor=np.array(
                [
                    obs.total_action,
                    1.0 if obs.lower_subsystem_verdict else 0.0,
                    1.0 if obs.upper_fence_verdict else 0.0,
                    1.0 if obs.intent_satisfied else 0.0,
                ],
                dtype=np.float32,
            ),
            mass=max(2.0, 2.0 + obs.total_action),
            position=apex_pos[: field.dimensions],
            scale_level=4.0,
        )
    )
    field.register_engram(
        EngramAttractor(
            id=f"{prefix}_engram_upper_rotor",
            name=f"UpperRotor_{system.name}",
            position=apex_pos[: field.dimensions],
            intensity=2.5,
            tier="macro",
            symbolic_intent=f"Trinitarian verdict={obs.status.value}",
        )
    )

    for bond_name, b_vid in b_voxels.items():
        _, _, res = system.bonds[bond_name].evaluate(vals)
        strength = max(0.01, 1.0 / (1.0 + abs(res)))
        field.link_voxels(b_vid, apex_vid, strength=strength)
        beams_linked += 1
        beam_strengths.append(strength)

    min_beam_strength = float(min(beam_strengths)) if beam_strengths else 1.0
    return {
        "causal_field": field,
        "quantity_voxels": q_voxels,
        "bond_voxels": b_voxels,
        "apex_voxel": apex_vid,
        "beams_linked": beams_linked,
        "min_beam_strength": min_beam_strength,
    }
