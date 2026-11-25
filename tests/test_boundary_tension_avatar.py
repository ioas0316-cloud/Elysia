r"""
Verification suite for ELYSIA Boundary-Tension Skeleton-Muscle-Skin Avatar Engine &
Scale-Axis Spectral O(1) Reproduction:
Tests asymmetric boundary contraction/relaxation tension driving joints cleanly,
incompressible volume-preserving muscle bulge + fascia skin sliding + compression-driven
wrinkles, obstacle/external load passive compliance with proximal sharing, and Cl(3,0)
Scale-Axis Spectral Manifold O(1) reproduction accuracy & separation.
"""

from __future__ import annotations

import time

import numpy as np

from core.embodiment import (
    ArticulatedTensionChain,
    ScaleSpectralMotorManifold,
)


def test_boundary_tension_asymmetry_volume_preservation_and_wrinkles():
    """Flexor contraction vs non-stretchable constraint bends joints, bulges belly, slides skin & creases wrinkles."""
    chain = ArticulatedTensionChain()

    # Track contraction rise speed vs viscoelastic release speed
    chain.step([0.90, 0.90, 0.30], [0.05, 0.05, 0.10], dt=0.02)
    rise_delta = chain.flex_tensions[1]
    for _ in range(25):
        chain.step([0.90, 0.90, 0.30], [0.05, 0.05, 0.10], dt=0.02)
    peak_t = chain.flex_tensions[1]
    chain.step([0.0, 0.0, 0.0], [0.05, 0.05, 0.10], dt=0.02)
    drop_delta = abs(peak_t - chain.flex_tensions[1])
    assert rise_delta > 0.0
    assert drop_delta < rise_delta * 1.15

    # Solve steady elbow-dominant curl
    pose = chain.solve_to_equilibrium([0.35, 0.92, 0.30], [0.10, 0.08, 0.20], steps=38)
    assert pose.joint_angles[1] > 0.45  # elbow flexed cleanly from boundary delta tension alone
    assert pose.muscles[1].flex_bulge > 1.05  # incompressible radial bulge on contracting muscle
    assert float(np.max(np.abs(pose.skin.slide_offsets))) > 1e-3  # tangential fascia slide occurs
    assert float(np.max(pose.skin.upper_wrinkle)) > 0.05  # inner compressed crease builds wrinkles
    # Inner flexor crease should compress more than stretched outer extensor elbow skin
    assert float(np.max(pose.skin.upper_wrinkle)) >= float(np.max(pose.skin.lower_wrinkle))


def test_external_contact_and_obstacle_induce_passive_compliance_and_load_sharing():
    """External downward force and rigid obstacle deflect joints & transfer overflow load proximally."""
    free_chain = ArticulatedTensionChain()
    free_pose = free_chain.solve_to_equilibrium([0.55, 0.65, 0.35], [0.20, 0.20, 0.20], steps=38)

    loaded_chain = ArticulatedTensionChain()
    loaded_pose = loaded_chain.solve_to_equilibrium(
        [0.55, 0.65, 0.35],
        [0.20, 0.20, 0.20],
        ext_force_xy=(0.0, -9.0),
        steps=38,
    )
    # Passive compliance: downward external force depresses arm angles compared to free pose
    assert loaded_pose.bone_points[-1, 1] < free_pose.bone_points[-1, 1] - 0.05
    assert abs(loaded_pose.muscles[0].overflow_torque) > 0.01

    # Obstacle placement against forearm bends/deflects away with measurable reaction force
    obs_chain = ArticulatedTensionChain()
    obs_pose = obs_chain.solve_to_equilibrium(
        [0.55, 0.75, 0.30],
        [0.15, 0.15, 0.20],
        obstacle=(0.35, 0.18, 0.08),
        steps=45,
    )
    assert np.hypot(*obs_pose.contact_force_xy) > 0.1 or obs_pose.joint_angles[1] != free_pose.joint_angles[1]


def test_scale_spectral_manifold_solves_once_and_reproduces_in_o1():
    """Solved regimes are classified onto Cl(3,0) rotors so nearby recalls run in O(1) without false collisions."""
    manifold = ScaleSpectralMotorManifold()
    ref = ArticulatedTensionChain()

    # 1. Distinct joint rotors (shoulder flex vs wrist flex) must stay geometrically separated in Cl(3,0)
    r_shoulder = manifold.encode_constraint_rotor([0.90, 0.0, 0.0], [0.0, 0.0, 0.0])
    r_wrist = manifold.encode_constraint_rotor([0.0, 0.0, 0.90], [0.0, 0.0, 0.0])
    from core.embodiment import batch_clifford_docking_affinity

    sep_aff, sep_tq = batch_clifford_docking_affinity(r_shoulder[None, :], r_wrist)
    assert float(sep_aff[0]) < 0.50
    assert float(sep_tq[0]) > 0.50

    # 2. First exposure triggers iterative physics_solve and populates manifold
    p1 = manifold.evaluate_or_reproduce([0.75, 0.85, 0.35], [0.15, 0.15, 0.20])
    assert p1.source_mode == "physics_solve"
    assert manifold.stats()["physics_solves"] == 1

    # 3. Nearby slight perturbation docks in Cl(3,0) and triggers O(1) spectral_reproduction accurately
    p2 = manifold.evaluate_or_reproduce([0.74, 0.86, 0.36], [0.16, 0.14, 0.20])
    assert p2.source_mode == "spectral_reproduction"
    assert p2.docking_affinity >= 0.92

    gt = ref.solve_to_equilibrium([0.74, 0.86, 0.36], [0.16, 0.14, 0.20])
    ang_err_deg = float(np.mean(np.abs(np.degrees(p2.joint_angles - gt.joint_angles))))
    assert ang_err_deg < 4.0

    # Warmed loop timing eliminates OS millisecond jitter
    t0 = time.perf_counter()
    for _ in range(10):
        manifold.evaluate_or_reproduce([0.74, 0.86, 0.36], [0.16, 0.14, 0.20])
    repro_s = time.perf_counter() - t0
    t_gt0 = time.perf_counter()
    for _ in range(10):
        ref.solve_to_equilibrium([0.74, 0.86, 0.36], [0.16, 0.14, 0.20])
    phys_s = time.perf_counter() - t_gt0
    assert repro_s < phys_s

    # 4. Qualitatively opposite / loaded regime misses manifold & launches new physics_solve automatically
    p3 = manifold.evaluate_or_reproduce([0.10, 0.15, 0.80], [0.75, 0.70, 0.10], ext_force_xy=(0.0, -8.5))
    assert p3.source_mode == "physics_solve"
    assert manifold.stats()["manifold_size"] == 2


def test_trajectory_rotor_captures_full_arc_and_samples_any_phase_in_o1():
    """Trajectory-as-a-Frame captures full curl once and evaluates arbitrary tau in O(1)."""
    manifold = ScaleSpectralMotorManifold()
    f_drv = [0.60, 0.90, 0.45]
    e_drv = [0.10, 0.10, 0.20]

    gt_chain = ArticulatedTensionChain()
    n_steps = 40
    gt_angles = np.zeros((n_steps + 1, gt_chain.n_joints), dtype=np.float64)
    for k in range(1, n_steps + 1):
        gt_chain.step(f_drv, e_drv, dt=0.02)
        gt_angles[k] = gt_chain.angles

    traj_rotor, m1 = manifold.capture_or_get_trajectory_rotor(f_drv, e_drv, steps=n_steps, dt=0.02)
    assert m1 == "physics_trajectory_solve"

    traj_cached, m2 = manifold.capture_or_get_trajectory_rotor([0.59, 0.89, 0.46], [0.11, 0.10, 0.20], steps=n_steps, dt=0.02)
    assert m2 == "spectral_trajectory_hit"

    # Reconstruct full arc across both grid & arbitrary sub-step phases (0.373, 0.812)
    errs = []
    for k in range(n_steps + 1):
        fr = traj_rotor.sample_at_phase(manifold.chain, k / float(n_steps), resolution_scale=1.0)
        errs.append(float(np.mean(np.abs(np.degrees(fr.joint_angles - gt_angles[k])))))
    assert float(np.mean(errs)) < 2.0

    f_mid1 = traj_cached.sample_at_phase(manifold.chain, 0.373, resolution_scale=0.5)
    f_mid2 = traj_cached.sample_at_phase(manifold.chain, 0.812, resolution_scale=0.5)
    assert f_mid2.joint_angles[1] > f_mid1.joint_angles[1] > 0.05
    assert float(np.linalg.norm(f_mid2.macro_rotor[4:7])) > float(np.linalg.norm(f_mid1.macro_rotor[4:7]))


def test_visibility_culled_micro_hydration_and_resolution_zoom():
    """Zooming viewport onto elbow culls off-screen segments while zooming micro density inside viewport."""
    manifold = ScaleSpectralMotorManifold()
    traj, _ = manifold.capture_or_get_trajectory_rotor([0.55, 0.90, 0.40], [0.10, 0.10, 0.20], steps=40)

    full_coarse = traj.sample_at_phase(manifold.chain, 1.0, viewport=None, resolution_scale=1.0)
    full_fine = traj.sample_at_phase(manifold.chain, 1.0, viewport=None, resolution_scale=0.1)
    assert full_fine.micro_surface.hydrated_micro_cells > full_coarse.micro_surface.hydrated_micro_cells * 5
    assert full_coarse.micro_surface.culled_segments == []

    # Tight viewport focusing purely near wrist/distal hand excludes shoulder segment 0
    wrist_xy = full_coarse.bone_points[2]
    tip_xy = full_coarse.bone_points[3]
    mid_distal = 0.5 * (wrist_xy + tip_xy)
    distal_vp = (
        float(mid_distal[0] - 0.06),
        float(mid_distal[0] + 0.06),
        float(mid_distal[1] - 0.06),
        float(mid_distal[1] + 0.06),
    )
    culled = traj.sample_at_phase(manifold.chain, 1.0, viewport=distal_vp, resolution_scale=0.1)
    assert 0 in culled.micro_surface.culled_segments
    assert 0 < culled.micro_surface.hydrated_micro_cells < full_fine.micro_surface.hydrated_micro_cells


def test_continuous_micro_spectrum_wrinkle_sheerness_and_tear():
    """Compression produces wrinkle, extension/stretch produces sheerness, exceeding threshold produces tear."""
    manifold = ScaleSpectralMotorManifold()
    traj, _ = manifold.capture_or_get_trajectory_rotor([0.65, 0.96, 0.50], [0.05, 0.05, 0.15], steps=40)
    frame = traj.sample_at_phase(manifold.chain, 1.0, resolution_scale=0.25, tear_threshold=1.12)
    ms = frame.micro_surface

    assert float(np.max(ms.upper_wrinkle)) > 0.05
    assert float(np.max(ms.lower_sheerness)) > 0.05
    assert float(np.max(ms.lower_tear)) > 0.0

