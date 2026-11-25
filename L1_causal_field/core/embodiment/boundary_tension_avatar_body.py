r"""
ELYSIA Boundary-Tension Skeleton-Muscle-Skin Avatar Body & Scale-Axis Spectral Manifold
======================================================================================

Implements the physical inverse-causal motor hierarchy:
  Boundary Tension Gradient (flexor contraction vs. non-stretchable constraint)
    -> Muscle Fiber Strain & Volume-Preserving Bulge (lambda, 1/sqrt(lambda))
      -> Moment Arm & Multi-Joint Load Sharing (shoulder <- elbow <- wrist)
        -> External Contact Normal Reaction & Passive Compliance
          -> Fascia-Sliding Elastic Skin & Compression-Driven Wrinkles

Once a constraint equilibrium is solved iteratively, its configuration and outcome are
reclassified onto the `Cl(3,0)` Scale-Axis Spectral Manifold (`ScaleSpectralMotorManifold`).
Subsequent compatible or nearby movements dock via `batch_clifford_docking_affinity` and
reproduce the full pose, muscle bulges, and skin/wrinkle envelope in O(1) without
re-running iterative physics integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.topology.geometric_rotor_docking import (
    MV_DIM,
    batch_clifford_docking_affinity,
    clifford_geometric_product,
    normalize_multivector,
)


@dataclass
class JointSpec:
    """Anatomical & physical parameters for one articulated skeletal-muscle segment."""

    name: str
    length: float
    min_angle: float
    max_angle: float
    base_radius: float
    moment_inertia: float = 0.22
    damping: float = 1.9
    constraint_stiffness: float = 4.2
    max_muscle_force: float = 14.0
    base_moment_arm: float = 0.075


def default_arm_joint_specs() -> List[JointSpec]:
    """Default 3-joint articulated arm (Shoulder, Elbow, Wrist) in meters."""
    return [
        JointSpec(
            name="shoulder",
            length=0.30,
            min_angle=-0.45,
            max_angle=1.85,
            base_radius=0.048,
            moment_inertia=0.14,
            damping=0.85,
            constraint_stiffness=7.5,
            max_muscle_force=26.0,
            base_moment_arm=0.085,
        ),
        JointSpec(
            name="elbow",
            length=0.26,
            min_angle=-0.20,
            max_angle=2.35,
            base_radius=0.038,
            moment_inertia=0.09,
            damping=0.65,
            constraint_stiffness=7.0,
            max_muscle_force=24.0,
            base_moment_arm=0.075,
        ),
        JointSpec(
            name="wrist",
            length=0.13,
            min_angle=-0.95,
            max_angle=1.25,
            base_radius=0.024,
            moment_inertia=0.04,
            damping=0.40,
            constraint_stiffness=5.5,
            max_muscle_force=16.0,
            base_moment_arm=0.045,
        ),
    ]


@dataclass
class MuscleBoundaryState:
    """Instantaneous muscle fiber tension, volume-preserving bulge, and joint torque."""

    flexor_tension: float = 0.0
    extensor_tension: float = 0.0
    fiber_stretch_flex: float = 1.0
    fiber_stretch_ext: float = 1.0
    flex_bulge: float = 1.0
    ext_bulge: float = 1.0
    delta_tension: float = 0.0
    effective_moment_arm: float = 0.075
    muscle_torque: float = 0.0
    overflow_torque: float = 0.0


@dataclass
class SkinEnvelopeSnapshot:
    """Continuously sampled upper (flexor/inner) and lower (extensor/outer) skin contours."""

    upper_pts: np.ndarray        # (M, 2)
    lower_pts: np.ndarray        # (M, 2)
    slide_offsets: np.ndarray    # (M,)
    upper_strain: np.ndarray     # (M-1,)
    lower_strain: np.ndarray     # (M-1,)
    upper_wrinkle: np.ndarray    # (M-1,)
    lower_wrinkle: np.ndarray    # (M-1,)


@dataclass
class BodyPoseSnapshot:
    """Complete biomechanical pose, skin envelope, and spectral provenance snapshot."""

    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    bone_points: np.ndarray
    muscles: List[MuscleBoundaryState]
    skin: SkinEnvelopeSnapshot
    contact_force_xy: Tuple[float, float]
    contact_penetration: float
    source_mode: str            # "physics_solve" | "spectral_reproduction"
    docking_affinity: float
    residual_torque: float
    spectral_rotor: np.ndarray


@dataclass
class MicroSurfacePatch:
    """
    Visibility-culled micro-scale skin surface hydrated ONLY for segments inside
    the active camera viewport at arbitrary scale resolution `1.0 / 0.25 / 0.1`.
    Unifies stretch -> compression wrinkle -> stretch sheerness -> tear spectrum.
    """

    upper_pts: np.ndarray        # (N_vis, 2)
    lower_pts: np.ndarray        # (N_vis, 2)
    upper_strain: np.ndarray     # (K_vis,)
    lower_strain: np.ndarray     # (K_vis,)
    upper_wrinkle: np.ndarray    # (K_vis,)
    lower_wrinkle: np.ndarray    # (K_vis,)
    upper_sheerness: np.ndarray  # (K_vis,)
    lower_sheerness: np.ndarray  # (K_vis,)
    upper_tear: np.ndarray       # (K_vis,)
    lower_tear: np.ndarray       # (K_vis,)
    visible_segments: List[int]
    culled_segments: List[int]
    resolution_scale: float
    hydrated_micro_cells: int


@dataclass
class TrajectoryFrame:
    """Instantaneous O(1) evaluation of a Trajectory-as-a-Frame rotor at phase `tau`."""

    tau: float
    macro_rotor: np.ndarray
    joint_angles: np.ndarray
    bone_points: np.ndarray
    muscles: List[MuscleBoundaryState]
    micro_surface: MicroSurfacePatch
    source_mode: str = "trajectory_rotor_o1"


class ArticulatedTensionChain:
    """
    Multi-joint boundary-tension physics chain:
    Skeleton support + antagonistic fiber tension + volume preservation +
    proximal load sharing + unilateral contact compliance + fascia-sliding skin.
    """

    SAMPLES_PER_SEGMENT = 6

    def __init__(self, specs: Optional[Sequence[JointSpec]] = None) -> None:
        self.specs: List[JointSpec] = list(specs) if specs else default_arm_joint_specs()
        self.n_joints = len(self.specs)
        self.angles = np.zeros(self.n_joints, dtype=np.float64)
        self.velocities = np.zeros(self.n_joints, dtype=np.float64)
        self.flex_tensions = np.zeros(self.n_joints, dtype=np.float64)
        self.ext_tensions = np.zeros(self.n_joints, dtype=np.float64)
        self.rest_upper_lens, self.rest_lower_lens = self._compute_rest_skin_lengths()

    def reset_state(self) -> None:
        self.angles.fill(0.0)
        self.velocities.fill(0.0)
        self.flex_tensions.fill(0.0)
        self.ext_tensions.fill(0.0)

    def _compute_rest_skin_lengths(self) -> Tuple[np.ndarray, np.ndarray]:
        zero_angles = np.zeros(self.n_joints, dtype=np.float64)
        zero_muscles = [
            MuscleBoundaryState(effective_moment_arm=s.base_moment_arm)
            for s in self.specs
        ]
        skin = self._build_skin_envelope(zero_angles, zero_muscles, rest_override=None)
        du = np.linalg.norm(np.diff(skin.upper_pts, axis=0), axis=1)
        dl = np.linalg.norm(np.diff(skin.lower_pts, axis=0), axis=1)
        return np.maximum(1e-5, du), np.maximum(1e-5, dl)

    def forward_kinematics(self, angles: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns cumulative world joint positions `(J+1, 2)` and link angles `(J,)`."""
        pts = np.zeros((self.n_joints + 1, 2), dtype=np.float64)
        cum_angles = np.zeros(self.n_joints, dtype=np.float64)
        phi = 0.0
        x, y = 0.0, 0.0
        for j, spec in enumerate(self.specs):
            phi += float(angles[j])
            cum_angles[j] = phi
            x += spec.length * math.cos(phi)
            y += spec.length * math.sin(phi)
            pts[j + 1] = (x, y)
        return pts, cum_angles

    def _compute_obstacle_contact(
        self,
        bone_pts: np.ndarray,
        angles: np.ndarray,
        velocities: np.ndarray,
        obstacle: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Computes closest obstacle penetration, unilateral normal force, and contact point."""
        if obstacle is None:
            return np.zeros(2, dtype=np.float64), 0.0, None

        ox, oy, orad = float(obstacle[0]), float(obstacle[1]), float(obstacle[2])
        obs_c = np.array([ox, oy], dtype=np.float64)

        best_pen = 0.0
        best_normal = np.zeros(2, dtype=np.float64)
        best_pt = bone_pts[-1].copy()
        best_seg = self.n_joints - 1

        for j, spec in enumerate(self.specs):
            a = bone_pts[j]
            b = bone_pts[j + 1]
            ab = b - a
            ab_len2 = float(np.dot(ab, ab))
            t = float(np.clip(np.dot(obs_c - a, ab) / max(1e-9, ab_len2), 0.0, 1.0))
            closest = a + t * ab
            diff = closest - obs_c
            dist = float(np.linalg.norm(diff))
            pen = (orad + spec.base_radius * 0.85) - dist
            if pen > best_pen:
                best_pen = pen
                best_normal = diff / max(1e-6, dist) if dist > 1e-6 else np.array([0.0, 1.0])
                best_pt = closest
                best_seg = j

        if best_pen <= 0.0:
            return np.zeros(2, dtype=np.float64), 0.0, None

        # Contact point velocity from active angular velocities up to best_seg
        pt_vel = np.zeros(2, dtype=np.float64)
        for j in range(best_seg + 1):
            r = best_pt - bone_pts[j]
            pt_vel += velocities[j] * np.array([-r[1], r[0]], dtype=np.float64)

        vn = float(np.dot(pt_vel, best_normal))
        f_mag = max(0.0, 180.0 * best_pen - 14.0 * vn)
        return best_normal * f_mag, best_pen, best_pt

    def _evaluate_muscle_states(
        self,
        angles: np.ndarray,
        flex_tensions: np.ndarray,
        ext_tensions: np.ndarray,
        overflow_torques: np.ndarray,
    ) -> List[MuscleBoundaryState]:
        states: List[MuscleBoundaryState] = []
        for j, spec in enumerate(self.specs):
            ang = float(angles[j])
            tf = float(flex_tensions[j])
            te = float(ext_tensions[j])
            cf = min(1.25, max(0.0, tf / max(1e-6, spec.max_muscle_force)))
            ce = min(1.25, max(0.0, te / max(1e-6, spec.max_muscle_force)))

            # Muscle length ratio (flexor shortens on positive bend & contraction)
            lam_f = float(np.clip(1.0 - 0.22 * cf - 0.14 * math.sin(ang), 0.64, 1.32))
            lam_e = float(np.clip(1.0 - 0.22 * ce + 0.14 * math.sin(ang), 0.64, 1.32))
            # Incompressible cross-sectional bulge (lambda * radius^2 ≈ const -> bulge = 1/sqrt(lambda))
            bulge_f = 1.0 / math.sqrt(lam_f)
            bulge_e = 1.0 / math.sqrt(lam_e)

            delta_t = tf - te
            eff_arm = spec.base_moment_arm * (0.70 + 0.30 * math.cos(ang))
            m_torque = delta_t * eff_arm
            states.append(
                MuscleBoundaryState(
                    flexor_tension=round(tf, 4),
                    extensor_tension=round(te, 4),
                    fiber_stretch_flex=round(lam_f, 4),
                    fiber_stretch_ext=round(lam_e, 4),
                    flex_bulge=round(bulge_f, 4),
                    ext_bulge=round(bulge_e, 4),
                    delta_tension=round(delta_t, 4),
                    effective_moment_arm=round(eff_arm, 5),
                    muscle_torque=round(m_torque, 4),
                    overflow_torque=round(float(overflow_torques[j]), 4),
                )
            )
        return states

    def _build_skin_envelope(
        self,
        angles: np.ndarray,
        muscles: Sequence[MuscleBoundaryState],
        rest_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> SkinEnvelopeSnapshot:
        """
        Generates continuous fascia-sliding skin contours over articulated bones,
        combining volume-preserving muscle bulge + tangential curvature sliding +
        emergent strain-driven crease wrinkles.
        """
        bone_pts, cum_angles = self.forward_kinematics(angles)
        n_seg = self.SAMPLES_PER_SEGMENT
        center_pts: List[np.ndarray] = []
        tangents: List[np.ndarray] = []
        upper_rads: List[float] = []
        lower_rads: List[float] = []
        slides: List[float] = []

        for j, spec in enumerate(self.specs):
            p0 = bone_pts[j]
            p1 = bone_pts[j + 1]
            t_dir = np.array([math.cos(cum_angles[j]), math.sin(cum_angles[j])], dtype=np.float64)
            next_rad = self.specs[j + 1].base_radius if j + 1 < self.n_joints else spec.base_radius * 0.72
            ms = muscles[j]
            # Curvature-coupled fascia tangential shift concentrated near joints
            k_bend = math.tanh(float(angles[j]))
            k_next = math.tanh(float(angles[j + 1])) if j + 1 < self.n_joints else 0.0

            count = n_seg + (1 if j == self.n_joints - 1 else 0)
            for s_idx in range(count):
                s = s_idx / float(n_seg)
                pt = (1.0 - s) * p0 + s * p1
                r_interp = (1.0 - s) * spec.base_radius + s * next_rad
                belly = math.sin(math.pi * s)
                r_up = r_interp * (1.0 + 0.55 * (ms.flex_bulge - 1.0) * (0.25 + 0.75 * belly))
                r_lo = r_interp * (1.0 + 0.55 * (ms.ext_bulge - 1.0) * (0.25 + 0.75 * belly))

                # Blend joint tangent smoothly across ends to prevent self-intersection
                if j > 0 and s < 0.25:
                    w = s / 0.25
                    t_prev = np.array([math.cos(cum_angles[j - 1]), math.sin(cum_angles[j - 1])], dtype=np.float64)
                    t_local = (1.0 - 0.5 * (1.0 - w)) * t_dir + 0.5 * (1.0 - w) * t_prev
                elif j + 1 < self.n_joints and s > 0.75:
                    w = (s - 0.75) / 0.25
                    t_nxt = np.array([math.cos(cum_angles[j + 1]), math.sin(cum_angles[j + 1])], dtype=np.float64)
                    t_local = (1.0 - 0.5 * w) * t_dir + 0.5 * w * t_nxt
                else:
                    t_local = t_dir
                t_norm = float(np.linalg.norm(t_local))
                t_local = t_local / max(1e-8, t_norm)

                # Tangential fascia slide driven by joint curvature & local tension bias
                slide_val = (
                    0.012 * k_bend * math.exp(-3.5 * s)
                    - 0.012 * k_next * math.exp(-3.5 * (1.0 - s))
                    + 0.004 * math.tanh(ms.delta_tension / 14.0) * belly
                )
                center_pts.append(pt)
                tangents.append(t_local)
                upper_rads.append(r_up)
                lower_rads.append(r_lo)
                slides.append(slide_val)

        m = len(center_pts)
        up_pts = np.zeros((m, 2), dtype=np.float64)
        lo_pts = np.zeros((m, 2), dtype=np.float64)
        for i in range(m):
            tx, ty = tangents[i]
            nx, ny = -ty, tx
            c = center_pts[i]
            sl = slides[i]
            up_pts[i] = [c[0] + upper_rads[i] * nx + sl * tx, c[1] + upper_rads[i] * ny + sl * ty]
            lo_pts[i] = [c[0] - lower_rads[i] * nx - 0.5 * sl * tx, c[1] - lower_rads[i] * ny - 0.5 * sl * ty]

        du = np.linalg.norm(np.diff(up_pts, axis=0), axis=1)
        dl = np.linalg.norm(np.diff(lo_pts, axis=0), axis=1)
        if rest_override is not None:
            ru, rl = rest_override
            u_strain = du / np.maximum(1e-5, ru)
            l_strain = dl / np.maximum(1e-5, rl)
        else:
            u_strain = np.ones_like(du)
            l_strain = np.ones_like(dl)

        u_wrinkle = np.clip((0.96 - u_strain) / 0.28, 0.0, 1.0)
        l_wrinkle = np.clip((0.96 - l_strain) / 0.28, 0.0, 1.0)
        return SkinEnvelopeSnapshot(
            upper_pts=up_pts,
            lower_pts=lo_pts,
            slide_offsets=np.asarray(slides, dtype=np.float64),
            upper_strain=u_strain,
            lower_strain=l_strain,
            upper_wrinkle=u_wrinkle,
            lower_wrinkle=l_wrinkle,
        )

    def reconstruct_pose_from_state(
        self,
        angles: np.ndarray,
        velocities: np.ndarray,
        flex_tensions: np.ndarray,
        ext_tensions: np.ndarray,
        overflow_torques: np.ndarray,
        contact_force_xy: Tuple[float, float],
        contact_penetration: float,
        source_mode: str,
        docking_affinity: float,
        residual_torque: float,
        spectral_rotor: np.ndarray,
    ) -> BodyPoseSnapshot:
        """Builds full kinematics + muscle bulges + sliding skin in one forward pass."""
        bone_pts, _ = self.forward_kinematics(angles)
        muscles = self._evaluate_muscle_states(angles, flex_tensions, ext_tensions, overflow_torques)
        skin = self._build_skin_envelope(angles, muscles, rest_override=(self.rest_upper_lens, self.rest_lower_lens))
        return BodyPoseSnapshot(
            joint_angles=angles.copy(),
            joint_velocities=velocities.copy(),
            bone_points=bone_pts,
            muscles=muscles,
            skin=skin,
            contact_force_xy=contact_force_xy,
            contact_penetration=round(float(contact_penetration), 5),
            source_mode=source_mode,
            docking_affinity=round(float(docking_affinity), 5),
            residual_torque=round(float(residual_torque), 5),
            spectral_rotor=spectral_rotor.copy(),
        )

    def step(
        self,
        flex_drives: Sequence[float],
        ext_drives: Sequence[float],
        *,
        ext_force_xy: Tuple[float, float] = (0.0, 0.0),
        obstacle: Optional[Tuple[float, float, float]] = None,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """Advances one causal biomechanical step: tension -> torque/overflow -> contact compliance."""
        f_drv = np.clip([float(flex_drives[j]) if j < len(flex_drives) else 0.0 for j in range(self.n_joints)], 0.0, 1.0)
        e_drv = np.clip([float(ext_drives[j]) if j < len(ext_drives) else 0.0 for j in range(self.n_joints)], 0.0, 1.0)

        # 1. Update asymmetric flexor / extensor boundary tensions
        for j, spec in enumerate(self.specs):
            stretch_bias = math.sin(self.angles[j]) * spec.constraint_stiffness
            t_flex_target = max(0.0, f_drv[j] * spec.max_muscle_force - stretch_bias)
            t_ext_target = max(0.0, e_drv[j] * spec.max_muscle_force + stretch_bias)

            # Active contraction is brisk; viscoelastic relaxation glides smoothly
            flex_rate = 14.0 if t_flex_target > self.flex_tensions[j] else 7.5
            ext_rate = 14.0 if t_ext_target > self.ext_tensions[j] else 7.5

            self.flex_tensions[j] += (t_flex_target - self.flex_tensions[j]) * min(1.0, flex_rate * dt)
            self.ext_tensions[j] += (t_ext_target - self.ext_tensions[j]) * min(1.0, ext_rate * dt)

        # 2. Muscle torques & multi-joint proximal tension overflow sharing (distal -> proximal)
        raw_torques = np.zeros(self.n_joints, dtype=np.float64)
        for j, spec in enumerate(self.specs):
            delta_t = self.flex_tensions[j] - self.ext_tensions[j]
            eff_arm = spec.base_moment_arm * (0.70 + 0.30 * math.cos(self.angles[j]))
            raw_torques[j] = delta_t * eff_arm

        # 3. External force + obstacle contact normal reaction via Jacobian transpose
        bone_pts, _ = self.forward_kinematics(self.angles)
        obs_force, pen, obs_pt = self._compute_obstacle_contact(bone_pts, self.angles, self.velocities, obstacle)
        ext_f = np.array([float(ext_force_xy[0]), float(ext_force_xy[1])], dtype=np.float64)
        tip = bone_pts[-1]

        ext_torques = np.zeros(self.n_joints, dtype=np.float64)
        for j in range(self.n_joints):
            r_tip = tip - bone_pts[j]
            tau = r_tip[0] * ext_f[1] - r_tip[1] * ext_f[0]
            if obs_pt is not None:
                r_obs = obs_pt - bone_pts[j]
                tau += r_obs[0] * obs_force[1] - r_obs[1] * obs_force[0]
            ext_torques[j] = tau

        # Share distal tension/load overflow proximally across coupled fascia
        overflow_torques = np.zeros(self.n_joints, dtype=np.float64)
        for j in range(self.n_joints - 1, 0, -1):
            excess = raw_torques[j] + 0.55 * ext_torques[j]
            if abs(excess) > 0.15:
                overflow_torques[j - 1] += 0.24 * excess

        # 4. Integrate joint accelerations with anatomical limits
        for j, spec in enumerate(self.specs):
            total_tau = raw_torques[j] + overflow_torques[j] + ext_torques[j] - spec.damping * self.velocities[j]
            acc = total_tau / spec.moment_inertia
            self.velocities[j] += acc * dt
            self.angles[j] += self.velocities[j] * dt
            if self.angles[j] < spec.min_angle:
                self.angles[j] = spec.min_angle
                self.velocities[j] = max(0.0, self.velocities[j])
            elif self.angles[j] > spec.max_angle:
                self.angles[j] = spec.max_angle
                self.velocities[j] = min(0.0, self.velocities[j])

        total_contact = (round(float(ext_f[0] + obs_force[0]), 4), round(float(ext_f[1] + obs_force[1]), 4))
        return overflow_torques, total_contact, pen

    def solve_to_equilibrium(
        self,
        flex_drives: Sequence[float],
        ext_drives: Sequence[float],
        *,
        ext_force_xy: Tuple[float, float] = (0.0, 0.0),
        obstacle: Optional[Tuple[float, float, float]] = None,
        steps: int = 35,
        dt: float = 0.02,
        reset_before_solve: bool = True,
        spectral_rotor: Optional[np.ndarray] = None,
    ) -> BodyPoseSnapshot:
        """Runs iterative boundary-tension physics and returns final equilibrium pose."""
        if reset_before_solve:
            self.reset_state()
        overflow = np.zeros(self.n_joints, dtype=np.float64)
        contact_f = (0.0, 0.0)
        pen = 0.0
        for _ in range(max(1, int(steps))):
            overflow, contact_f, pen = self.step(
                flex_drives,
                ext_drives,
                ext_force_xy=ext_force_xy,
                obstacle=obstacle,
                dt=dt,
            )

        if spectral_rotor is None:
            spectral_rotor = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return self.reconstruct_pose_from_state(
            angles=self.angles,
            velocities=self.velocities,
            flex_tensions=self.flex_tensions,
            ext_tensions=self.ext_tensions,
            overflow_torques=overflow,
            contact_force_xy=contact_f,
            contact_penetration=pen,
            source_mode="physics_solve",
            docking_affinity=1.0,
            residual_torque=0.0,
            spectral_rotor=spectral_rotor,
        )


@dataclass
class _SpectralMotorEntry:
    rotor: np.ndarray
    has_obstacle: bool
    angles: np.ndarray
    velocities: np.ndarray
    flex_tensions: np.ndarray
    ext_tensions: np.ndarray
    overflow_torques: np.ndarray
    contact_force_xy: Tuple[float, float]
    contact_penetration: float


class TrajectoryScaleRotor:
    """
    Trajectory-as-a-Frame Scale Rotor (`R(tau)`, `tau in [0, 1]`):
    Captures an entire causal physics arc (`start -> target` boundary constraint rollout)
    into compact Cosine spectral modes + a continuous `Cl(3,0)` bivector rotor generator,
    enabling O(1) playback at ANY phase `tau` and ANY micro-scale resolution (`1.0 / 0.1`)
    while lazily hydrating ONLY visible segments inside the camera viewport.
    """

    def __init__(
        self,
        angle_coeffs: np.ndarray,      # (K_modes, J)
        flex_coeffs: np.ndarray,       # (K_modes, J)
        ext_coeffs: np.ndarray,        # (K_modes, J)
        overflow_coeffs: np.ndarray,   # (K_modes, J)
        bivector_generator: np.ndarray,  # (3,) [e12, e23, e31] net displacement arc
        n_modes: int = 8,
    ) -> None:
        self.angle_coeffs = angle_coeffs
        self.flex_coeffs = flex_coeffs
        self.ext_coeffs = ext_coeffs
        self.overflow_coeffs = overflow_coeffs
        self.bivector_generator = bivector_generator
        self.n_modes = int(n_modes)

    @staticmethod
    def _basis_matrix(taus: np.ndarray, n_modes: int) -> np.ndarray:
        """Cosine spectral basis `cos(k * pi * tau)` on `[0, 1]` avoiding endpoint ringing."""
        ks = np.arange(n_modes, dtype=np.float64)[None, :]
        return np.cos(math.pi * taus[:, None] * ks)

    @classmethod
    def fit_from_rollout(
        cls,
        angles_seq: np.ndarray,    # (T, J)
        flex_t_seq: np.ndarray,    # (T, J)
        ext_t_seq: np.ndarray,     # (T, J)
        overflow_seq: np.ndarray,  # (T, J)
        n_modes: int = 8,
    ) -> "TrajectoryScaleRotor":
        """Fits compact spectral trajectory coefficients once from a full physics rollout."""
        t_steps = int(angles_seq.shape[0])
        k_eff = max(2, min(int(n_modes), t_steps))
        taus = np.linspace(0.0, 1.0, t_steps, dtype=np.float64)
        phi = cls._basis_matrix(taus, k_eff)

        ang_c, *_ = np.linalg.lstsq(phi, angles_seq, rcond=None)
        flx_c, *_ = np.linalg.lstsq(phi, flex_t_seq, rcond=None)
        ext_c, *_ = np.linalg.lstsq(phi, ext_t_seq, rcond=None)
        ovf_c, *_ = np.linalg.lstsq(phi, overflow_seq, rcond=None)

        delta = angles_seq[-1] - angles_seq[0]
        b_gen = np.zeros(3, dtype=np.float64)
        for j in range(min(3, delta.size)):
            b_gen[j] = float(delta[j])

        return cls(
            angle_coeffs=ang_c,
            flex_coeffs=flx_c,
            ext_coeffs=ext_c,
            overflow_coeffs=ovf_c,
            bivector_generator=b_gen,
            n_modes=k_eff,
        )

    def macro_rotor_at(self, tau: float) -> np.ndarray:
        """Evaluates `R(tau) = exp(-tau * B / 2)` as an 8-D `Cl(3,0)` unit rotor."""
        t = float(np.clip(tau, 0.0, 1.0))
        b_norm = float(np.linalg.norm(self.bivector_generator))
        if b_norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        half = 0.5 * t * b_norm
        b_hat = self.bivector_generator / b_norm
        return np.array(
            [
                math.cos(half),
                0.0,
                0.0,
                0.0,
                math.sin(half) * float(b_hat[0]),
                math.sin(half) * float(b_hat[1]),
                math.sin(half) * float(b_hat[2]),
                0.0,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _segment_in_viewport(
        p0: np.ndarray,
        p1: np.ndarray,
        pad: float,
        viewport: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        if viewport is None:
            return True
        vx0, vx1, vy0, vy1 = [float(x) for x in viewport]
        sx0 = min(float(p0[0]), float(p1[0])) - pad
        sx1 = max(float(p0[0]), float(p1[0])) + pad
        sy0 = min(float(p0[1]), float(p1[1])) - pad
        sy1 = max(float(p0[1]), float(p1[1])) + pad
        return not (sx1 < vx0 or sx0 > vx1 or sy1 < vy0 or sy0 > vy1)

    def sample_at_phase(
        self,
        chain: ArticulatedTensionChain,
        tau: float,
        *,
        viewport: Optional[Tuple[float, float, float, float]] = None,
        resolution_scale: float = 1.0,
        tear_threshold: float = 1.18,
    ) -> TrajectoryFrame:
        """
        O(1) evaluation across coupled Macro -> Meso -> Micro layers at continuous
        phase `tau in [0, 1]` and arbitrary spatial `resolution_scale`, skipping
        micro hydration for any segment outside `viewport`.
        """
        t = float(np.clip(tau, 0.0, 1.0))
        basis = np.cos(math.pi * t * np.arange(self.n_modes, dtype=np.float64))

        # 1. Macro Layer (Skeleton & Cl(3,0) Bivector Phase Rotor)
        angles = np.asarray(basis @ self.angle_coeffs, dtype=np.float64)
        for j, spec in enumerate(chain.specs):
            angles[j] = float(np.clip(angles[j], spec.min_angle, spec.max_angle))
        bone_pts, cum_angles = chain.forward_kinematics(angles)
        macro_rotor = self.macro_rotor_at(t)

        # 2. Meso Layer (Antagonistic Tensions & Volume-Preserving Bulges)
        flex_t = np.maximum(0.0, np.asarray(basis @ self.flex_coeffs, dtype=np.float64))
        ext_t = np.maximum(0.0, np.asarray(basis @ self.ext_coeffs, dtype=np.float64))
        ovf_t = np.asarray(basis @ self.overflow_coeffs, dtype=np.float64)
        muscles = chain._evaluate_muscle_states(angles, flex_t, ext_t, ovf_t)

        # 3. Micro Layer (Visibility-Culled Lazy Skin + Stretch/Wrinkle/Sheerness/Tear)
        res_scale = max(0.05, float(resolution_scale))
        n_sub = max(2, min(80, int(round(6.0 / res_scale))))

        vis_segs: List[int] = []
        culled_segs: List[int] = []
        all_up_pts: List[np.ndarray] = []
        all_lo_pts: List[np.ndarray] = []
        all_u_strain: List[np.ndarray] = []
        all_l_strain: List[np.ndarray] = []

        for j, spec in enumerate(chain.specs):
            p0 = bone_pts[j]
            p1 = bone_pts[j + 1]
            if not self._segment_in_viewport(p0, p1, spec.base_radius * 2.2, viewport):
                culled_segs.append(j)
                continue

            vis_segs.append(j)
            ms = muscles[j]
            next_rad = chain.specs[j + 1].base_radius if j + 1 < chain.n_joints else spec.base_radius * 0.72
            t_dir = np.array([math.cos(cum_angles[j]), math.sin(cum_angles[j])], dtype=np.float64)
            k_bend = math.tanh(float(angles[j]))
            k_next = math.tanh(float(angles[j + 1])) if j + 1 < chain.n_joints else 0.0

            seg_up = np.zeros((n_sub + 1, 2), dtype=np.float64)
            seg_lo = np.zeros((n_sub + 1, 2), dtype=np.float64)
            for i in range(n_sub + 1):
                s = i / float(n_sub)
                pt = (1.0 - s) * p0 + s * p1
                r_int = (1.0 - s) * spec.base_radius + s * next_rad
                belly = math.sin(math.pi * s)
                r_up = r_int * (1.0 + 0.55 * (ms.flex_bulge - 1.0) * (0.25 + 0.75 * belly))
                r_lo = r_int * (1.0 + 0.55 * (ms.ext_bulge - 1.0) * (0.25 + 0.75 * belly))

                if j > 0 and s < 0.25:
                    w = s / 0.25
                    t_prev = np.array([math.cos(cum_angles[j - 1]), math.sin(cum_angles[j - 1])], dtype=np.float64)
                    t_loc = (1.0 - 0.5 * (1.0 - w)) * t_dir + 0.5 * (1.0 - w) * t_prev
                elif j + 1 < chain.n_joints and s > 0.75:
                    w = (s - 0.75) / 0.25
                    t_nxt = np.array([math.cos(cum_angles[j + 1]), math.sin(cum_angles[j + 1])], dtype=np.float64)
                    t_loc = (1.0 - 0.5 * w) * t_dir + 0.5 * w * t_nxt
                else:
                    t_loc = t_dir
                t_loc = t_loc / max(1e-8, float(np.linalg.norm(t_loc)))
                tx, ty = t_loc
                nx, ny = -ty, tx
                sl = (
                    0.012 * k_bend * math.exp(-3.5 * s)
                    - 0.012 * k_next * math.exp(-3.5 * (1.0 - s))
                    + 0.004 * math.tanh(ms.delta_tension / 14.0) * belly
                )
                seg_up[i] = [pt[0] + r_up * nx + sl * tx, pt[1] + r_up * ny + sl * ty]
                seg_lo[i] = [pt[0] - r_lo * nx - 0.5 * sl * tx, pt[1] - r_lo * ny - 0.5 * sl * ty]

            # Exact analytical rest micro-strip length at current `n_sub` resolution
            rest_step = max(1e-6, math.hypot(spec.length / float(n_sub), (next_rad - spec.base_radius) / float(n_sub)))
            du = np.linalg.norm(np.diff(seg_up, axis=0), axis=1) / rest_step
            dl = np.linalg.norm(np.diff(seg_lo, axis=0), axis=1) / rest_step

            all_up_pts.append(seg_up)
            all_lo_pts.append(seg_lo)
            all_u_strain.append(du)
            all_l_strain.append(dl)

        if vis_segs:
            up_pts = np.vstack(all_up_pts)
            lo_pts = np.vstack(all_lo_pts)
            u_strain = np.concatenate(all_u_strain)
            l_strain = np.concatenate(all_l_strain)
            u_wrinkle = np.clip((0.96 - u_strain) / 0.28, 0.0, 1.0)
            l_wrinkle = np.clip((0.96 - l_strain) / 0.28, 0.0, 1.0)
            u_sheer = np.clip((u_strain - 1.02) / 0.22, 0.0, 1.0)
            l_sheer = np.clip((l_strain - 1.02) / 0.22, 0.0, 1.0)
            u_tear = np.clip((u_strain - float(tear_threshold)) / 0.16, 0.0, 1.0)
            l_tear = np.clip((l_strain - float(tear_threshold)) / 0.16, 0.0, 1.0)
            cells = int(u_strain.size + l_strain.size)
        else:
            up_pts = np.zeros((0, 2), dtype=np.float64)
            lo_pts = np.zeros((0, 2), dtype=np.float64)
            empty = np.zeros(0, dtype=np.float64)
            u_strain = l_strain = u_wrinkle = l_wrinkle = u_sheer = l_sheer = u_tear = l_tear = empty
            cells = 0

        patch = MicroSurfacePatch(
            upper_pts=up_pts,
            lower_pts=lo_pts,
            upper_strain=u_strain,
            lower_strain=l_strain,
            upper_wrinkle=u_wrinkle,
            lower_wrinkle=l_wrinkle,
            upper_sheerness=u_sheer,
            lower_sheerness=l_sheer,
            upper_tear=u_tear,
            lower_tear=l_tear,
            visible_segments=vis_segs,
            culled_segments=culled_segs,
            resolution_scale=res_scale,
            hydrated_micro_cells=cells,
        )
        return TrajectoryFrame(
            tau=t,
            macro_rotor=macro_rotor,
            joint_angles=angles,
            bone_points=bone_pts,
            muscles=muscles,
            micro_surface=patch,
            source_mode="trajectory_rotor_o1",
        )


class ScaleSpectralMotorManifold:
    """
    Scale-Axis Spectral Reclassification & O(1) Reproduction Engine:
    Encodes muscle drives and boundary contact constraints into `Cl(3,0)` bivector rotors
    across Scale 0 (per-joint drive) -> Scale 1 (chain composition) -> Scale 2 (contact regime),
    allowing exact or interpolated O(1) pose reproduction whenever `Cl(3,0)` docking aligns.
    """

    def __init__(self, chain: Optional[ArticulatedTensionChain] = None) -> None:
        self.chain = chain or ArticulatedTensionChain()
        self.entries: List[_SpectralMotorEntry] = []
        self.trajectory_rotors: List[Tuple[np.ndarray, bool, TrajectoryScaleRotor]] = []
        self.total_queries: int = 0
        self.physics_solves: int = 0
        self.spectral_hits: int = 0
        self.trajectory_solves: int = 0
        self.trajectory_hits: int = 0

    def encode_constraint_rotor(
        self,
        flex_drives: Sequence[float],
        ext_drives: Sequence[float],
        ext_force_xy: Tuple[float, float] = (0.0, 0.0),
        obstacle: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Encodes multi-joint drives + external load onto distinct `Cl(3,0)` bivector planes:
          - Joint 0 (Shoulder) -> e12 plane rotation
          - Joint 1 (Elbow)    -> e23 plane rotation
          - Joint 2 (Wrist)    -> e31 plane rotation
          - Contact/Co-tension -> Grade-1 & pseudoscalar modulation (separating unloaded vs loaded)
        """
        n = self.chain.n_joints
        f_drv = [float(flex_drives[j]) if j < len(flex_drives) else 0.0 for j in range(n)]
        e_drv = [float(ext_drives[j]) if j < len(ext_drives) else 0.0 for j in range(n)]

        # Scale 0: Individual joint drive rotors on orthogonal bivector planes
        joint_rotors: List[np.ndarray] = []
        biv_indices = (4, 5, 6)  # e12, e23, e31
        co_sum = 0.0
        for j in range(n):
            diff = float(np.clip(f_drv[j] - e_drv[j], -1.0, 1.0))
            co_sum += 0.5 * (f_drv[j] + e_drv[j])
            half_ang = diff * (math.pi / 2.6)
            r = np.zeros(MV_DIM, dtype=np.float64)
            r[0] = math.cos(half_ang)
            r[biv_indices[j % 3]] = math.sin(half_ang)
            joint_rotors.append(r)

        # Scale 1: Exact non-commutative geometric product across articulated chain
        chain_rotor = joint_rotors[0]
        for rj in joint_rotors[1:]:
            chain_rotor = normalize_multivector(clifford_geometric_product(chain_rotor, rj))

        # Scale 2: External contact force, obstacle geometry, & co-contraction tension in odd grades
        fx = float(ext_force_xy[0]) / 12.0
        fy = float(ext_force_xy[1]) / 12.0
        obs_sig = 0.0
        if obstacle is not None:
            fx += float(obstacle[0]) * 0.45
            fy += float(obstacle[1]) * 0.45
            obs_sig = 0.42
        co_norm = co_sum / max(1, n)

        composed = chain_rotor.copy()
        composed[1] += math.tanh(fx) * 0.42
        composed[2] += math.tanh(fy) * 0.42
        composed[3] += (co_norm - 0.35) * 0.28 + obs_sig
        composed[7] += 0.18 * math.tanh(fx - fy)
        return normalize_multivector(composed)

    def evaluate_or_reproduce(
        self,
        flex_drives: Sequence[float],
        ext_drives: Sequence[float],
        *,
        ext_force_xy: Tuple[float, float] = (0.0, 0.0),
        obstacle: Optional[Tuple[float, float, float]] = None,
        min_affinity: float = 0.92,
        max_torque: float = 0.22,
        solve_steps: int = 35,
    ) -> BodyPoseSnapshot:
        """
        Reproduces the equilibrium pose in O(1) if a geometrically docked spectral entry
        exists; otherwise runs the iterative physics solve once and classifies the result.
        """
        self.total_queries += 1
        q_rotor = self.encode_constraint_rotor(
            flex_drives,
            ext_drives,
            ext_force_xy=ext_force_xy,
            obstacle=obstacle,
        )
        has_obs = obstacle is not None

        matching_indices = [idx for idx, ent in enumerate(self.entries) if ent.has_obstacle == has_obs]
        if matching_indices:
            cand_mvs = np.stack([self.entries[i].rotor for i in matching_indices], axis=0)
            affs, torques = batch_clifford_docking_affinity(cand_mvs, q_rotor)
            best_pos = int(np.argmax(affs))
            best_aff = float(affs[best_pos])
            best_tq = float(torques[best_pos])

            if best_aff >= min_affinity and best_tq <= max_torque:
                self.spectral_hits += 1
                # Affinity-weighted interpolation over top-3 docked neighbors in manifold
                good = [
                    (float(affs[p]), float(torques[p]), self.entries[matching_indices[p]])
                    for p in range(len(matching_indices))
                    if float(affs[p]) >= min_affinity - 0.03 and float(torques[p]) <= max_torque + 0.08
                ]
                good.sort(key=lambda item: (-item[0], item[1]))
                top_neighbors = good[:3] or [(best_aff, best_tq, self.entries[matching_indices[best_pos]])]

                raw_weights = np.array(
                    [math.exp(-10.0 * tq) * max(1e-4, af) for af, tq, _e in top_neighbors],
                    dtype=np.float64,
                )
                w = raw_weights / np.sum(raw_weights)

                angles = np.sum([w[i] * top_neighbors[i][2].angles for i in range(len(top_neighbors))], axis=0)
                vels = np.sum([w[i] * top_neighbors[i][2].velocities for i in range(len(top_neighbors))], axis=0)
                f_ten = np.sum([w[i] * top_neighbors[i][2].flex_tensions for i in range(len(top_neighbors))], axis=0)
                e_ten = np.sum([w[i] * top_neighbors[i][2].ext_tensions for i in range(len(top_neighbors))], axis=0)
                ovf = np.sum([w[i] * top_neighbors[i][2].overflow_torques for i in range(len(top_neighbors))], axis=0)
                cf_x = float(sum(w[i] * top_neighbors[i][2].contact_force_xy[0] for i in range(len(top_neighbors))))
                cf_y = float(sum(w[i] * top_neighbors[i][2].contact_force_xy[1] for i in range(len(top_neighbors))))
                pen = float(sum(w[i] * top_neighbors[i][2].contact_penetration for i in range(len(top_neighbors))))

                return self.chain.reconstruct_pose_from_state(
                    angles=angles,
                    velocities=vels,
                    flex_tensions=f_ten,
                    ext_tensions=e_ten,
                    overflow_torques=ovf,
                    contact_force_xy=(round(cf_x, 4), round(cf_y, 4)),
                    contact_penetration=pen,
                    source_mode="spectral_reproduction",
                    docking_affinity=best_aff,
                    residual_torque=best_tq,
                    spectral_rotor=q_rotor,
                )

        # Novel boundary/constraint regime -> run full iterative physics solve once & classify
        self.physics_solves += 1
        solved = self.chain.solve_to_equilibrium(
            flex_drives,
            ext_drives,
            ext_force_xy=ext_force_xy,
            obstacle=obstacle,
            steps=solve_steps,
            reset_before_solve=True,
            spectral_rotor=q_rotor,
        )
        self.entries.append(
            _SpectralMotorEntry(
                rotor=q_rotor.copy(),
                has_obstacle=has_obs,
                angles=solved.joint_angles.copy(),
                velocities=solved.joint_velocities.copy(),
                flex_tensions=np.array([m.flexor_tension for m in solved.muscles], dtype=np.float64),
                ext_tensions=np.array([m.extensor_tension for m in solved.muscles], dtype=np.float64),
                overflow_torques=np.array([m.overflow_torque for m in solved.muscles], dtype=np.float64),
                contact_force_xy=solved.contact_force_xy,
                contact_penetration=solved.contact_penetration,
            )
        )
        return solved

    def capture_or_get_trajectory_rotor(
        self,
        flex_drives: Sequence[float],
        ext_drives: Sequence[float],
        *,
        ext_force_xy: Tuple[float, float] = (0.0, 0.0),
        obstacle: Optional[Tuple[float, float, float]] = None,
        min_affinity: float = 0.92,
        max_torque: float = 0.22,
        steps: int = 40,
        dt: float = 0.02,
    ) -> Tuple[TrajectoryScaleRotor, str]:
        """
        Captures a full boundary-tension movement arc into ONE `TrajectoryScaleRotor` on first
        encounter, or retrieves an existing geometrically docked trajectory rotor in O(1).
        """
        q_rotor = self.encode_constraint_rotor(
            flex_drives,
            ext_drives,
            ext_force_xy=ext_force_xy,
            obstacle=obstacle,
        )
        has_obs = obstacle is not None
        matches = [item for item in self.trajectory_rotors if item[1] == has_obs]
        if matches:
            cand_mvs = np.stack([item[0] for item in matches], axis=0)
            affs, torques = batch_clifford_docking_affinity(cand_mvs, q_rotor)
            best_idx = int(np.argmax(affs))
            if float(affs[best_idx]) >= min_affinity and float(torques[best_idx]) <= max_torque:
                self.trajectory_hits += 1
                return matches[best_idx][2], "spectral_trajectory_hit"

        self.trajectory_solves += 1
        self.chain.reset_state()
        n_steps = max(4, int(steps))
        n_j = self.chain.n_joints
        angles_seq = np.zeros((n_steps + 1, n_j), dtype=np.float64)
        flex_t_seq = np.zeros((n_steps + 1, n_j), dtype=np.float64)
        ext_t_seq = np.zeros((n_steps + 1, n_j), dtype=np.float64)
        overflow_seq = np.zeros((n_steps + 1, n_j), dtype=np.float64)

        for k in range(1, n_steps + 1):
            ovf, _cf, _pen = self.chain.step(
                flex_drives,
                ext_drives,
                ext_force_xy=ext_force_xy,
                obstacle=obstacle,
                dt=dt,
            )
            angles_seq[k] = self.chain.angles
            flex_t_seq[k] = self.chain.flex_tensions
            ext_t_seq[k] = self.chain.ext_tensions
            overflow_seq[k] = ovf

        traj_rotor = TrajectoryScaleRotor.fit_from_rollout(
            angles_seq,
            flex_t_seq,
            ext_t_seq,
            overflow_seq,
            n_modes=8,
        )
        self.trajectory_rotors.append((q_rotor.copy(), has_obs, traj_rotor))
        return traj_rotor, "physics_trajectory_solve"

    def stats(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "physics_solves": self.physics_solves,
            "spectral_hits": self.spectral_hits,
            "hit_rate": round(self.spectral_hits / max(1, self.total_queries), 4),
            "manifold_size": len(self.entries),
            "trajectory_solves": self.trajectory_solves,
            "trajectory_hits": self.trajectory_hits,
            "trajectory_manifold_size": len(self.trajectory_rotors),
        }
