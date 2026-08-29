"""
Unit tests and simulation verification for PerceptualLensControl module.
"""

import numpy as np
import pytest
from core.physics.causal_field import CausalField, InformationVoxel
from core.lens.perceptual_lens_control import (
    PerceptualLensControl,
    ObservationBandwidth,
    CausalInvariant,
    SealedAttractor,
    ReframingShift
)


def test_perceptual_lens_initialization():
    cf = CausalField(dimensions=3)
    plc = PerceptualLensControl(causal_field=cf)

    assert plc.bandwidth.current_scale == 1.0
    assert plc.accumulated_friction == 0.0
    assert len(plc.sealed_attractors) == 0


def test_scale_adaptive_reframing_and_friction_escalation():
    """
    1. 마찰 장력(V_t) 상승 시 은하-유체-물방울 간 스케일 자율 리프레이밍 검증.
    """
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("star_1", "Galaxy Node 1", tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32), position=np.array([10.0, 0.0, 0.0], dtype=np.float32))
    v2 = InformationVoxel("star_2", "Galaxy Node 2", tensor=np.array([0.0, 1.0, 0.0], dtype=np.float32), position=np.array([20.0, 0.0, 0.0], dtype=np.float32))
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("star_1", "star_2", strength=5.0)

    bandwidth = ObservationBandwidth(scale_min=0.1, scale_max=100.0, current_scale=1.0, max_friction_capacity=10.0)
    plc = PerceptualLensControl(causal_field=cf, bandwidth=bandwidth, critical_friction_threshold=50.0)

    # Initially stable scale
    shift_1 = plc.evaluate_and_reframe()
    assert plc.bandwidth.current_scale == 1.0

    # Inject high velocity / tension to escalate friction Vt > C_lens
    v1.velocity = np.array([15.0, -10.0, 5.0], dtype=np.float32)
    v2.velocity = np.array([-15.0, 10.0, -5.0], dtype=np.float32)
    cf.beams[0].current_tension = 12.0  # Vt exceeds max_friction_capacity (10.0)

    shift_2 = plc.evaluate_and_reframe()
    assert shift_2.shift_type == "MACRO_SHIFT"
    assert plc.bandwidth.current_scale > 1.0  # Zoomed out to galactic macro scale to absorb fluid friction


def test_isomorphic_structural_invariant_preservation():
    """
    2. 스케일 전환 전후의 구조 불변량(I_c) 가역적 동형성 보존 여부 검증.
    """
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("drop_1", "Fluid Droplet 1", tensor=np.array([1.0, 1.0, 0.0], dtype=np.float32), position=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = InformationVoxel("drop_2", "Fluid Droplet 2", tensor=np.array([0.5, 0.5, 0.0], dtype=np.float32), position=np.array([2.0, 0.0, 0.0], dtype=np.float32))
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("drop_1", "drop_2", strength=2.0)

    plc = PerceptualLensControl(causal_field=cf)
    inv_pre = plc.extract_structural_invariant("pre_shift")

    # Zoom scale dynamically (Macro refocus)
    refocus_result = plc.refocus_scale(target_scale=5.0, preserve_invariants=True)
    inv_post = plc.extract_structural_invariant("post_shift")

    assert refocus_result["isomorphism_preserved"] is True
    assert plc.is_isomorphic(inv_pre, inv_post) is True
    # Ensure directional morphisms are preserved
    assert set(inv_pre.morphisms.keys()) == set(inv_post.morphisms.keys())


def test_sealed_attractor_isolation_on_critical_breach():
    """
    3. 임계 장력(V_critical) 초과 시 충돌 노드의 SealedAttractor 격리 처리 검증.
    """
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("core_1", "Stable Voxel", tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32), position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v2 = InformationVoxel("rupture_1", "Rupturing Node", tensor=np.array([0.0, 0.0, 1.0], dtype=np.float32), position=np.array([100.0, 0.0, 0.0], dtype=np.float32))
    v2.potential = 50.0  # Massive potential mismatch
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("core_1", "rupture_1", strength=10.0)

    # Force beam break to simulate catastrophic rupture
    cf.beams[0].is_broken = True

    bandwidth = ObservationBandwidth(max_friction_capacity=5.0)
    plc = PerceptualLensControl(causal_field=cf, bandwidth=bandwidth, critical_friction_threshold=15.0)

    # Evaluate reframing under critical friction
    shift = plc.evaluate_and_reframe()

    assert shift.shift_type == "MICRO_SHIFT_SEALED"
    assert shift.isolated_attractor_id is not None
    assert len(plc.sealed_attractors) == 1

    sealed = list(plc.sealed_attractors.values())[0]
    assert sealed.critical_friction_level > 15.0
    # Ruptured node should be isolated from active field
    assert "rupture_1" in sealed.sealed_voxels
    assert "rupture_1" not in cf.voxels
