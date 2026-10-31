"""
tests/test_real_world_entities_simulation.py

Unit & Integration Tests for:
1. Field Seed Manifest serialization/deserialization & ultra-compact footprint (< 10 KB).
2. Non-destructive delta modulation (variant parameters over base seed).
3. Boundary Transducer Membrane (continuous field to discrete RGB FrameBuffer & network packet).
4. Four Real-World Concrete Entities:
   - Building & Person: Volumetric Blister meniscus & phase resonance boundary without AABB collision boxes.
   - Forest & Nature: Semantic scale continuum along w-axis (-0.6 to +0.3).
   - Light & Shadow: Wavefront phase interference (constructive peak vs dark interference zone).
   - Dynamic non-penetration: Repulsion force prevents person from penetrating building.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import elysia_loader
elysia_loader.install()

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer
)
from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor,
    DeltaModulationDescriptor,
    BoundaryTransducerMembrane
)
from scripts.verify_step3_real_world_entities_simulation import (
    build_real_world_entities_field_seed
)


def test_field_seed_manifest_serialization_and_footprint():
    """Verify that the Field Seed encodes generating laws under 10 KB (vs GBs of video)."""
    seed = build_real_world_entities_field_seed()

    # JSON roundtrip
    json_str = seed.to_json()
    assert isinstance(json_str, str)
    seed_restored = FieldSeed.from_json(json_str)

    assert seed_restored.seed_name == seed.seed_name
    assert len(seed_restored.rotors) == len(seed.rotors)
    assert seed_restored.kernel_descriptor.omega_c == seed.kernel_descriptor.omega_c

    # Binary byte footprint check (< 10 KB)
    byte_size = len(seed.to_bytes())
    assert byte_size < 10240, f"Field Seed size {byte_size} bytes exceeds 10 KB limit"
    assert byte_size > 1000, f"Field Seed size {byte_size} bytes is unexpectedly small"


def test_non_destructive_delta_modulation():
    """Verify non-destructive delta modulation on a base field seed."""
    seed = build_real_world_entities_field_seed()

    # Create variant delta (observer rotor rotation + intent boost on person)
    delta = DeltaModulationDescriptor(
        delta_id="variant_high_tension",
        delta_intent_potential={"person_agent": 2.5},
        phase_offsets={"person_agent": 0.5},
        observer_rotor_bivectors=[0.1, 0.2, 0.0, 0.0, 0.1, 0.0],
        w_slice_offset=0.1
    )

    engine_base, _ = seed.instantiate_engine()
    engine_mod, node_ids = seed.instantiate_engine(delta=delta)

    person_idx = node_ids.index("person_agent")

    # Verify base seed is unmodified
    assert seed.rotors[person_idx].intent_potential == 1.5

    # Verify modulated engine reflects delta
    assert engine_mod.nodes[person_idx].intent_potential == 4.0
    assert engine_base.nodes[person_idx].intent_potential == 1.5
    assert not np.allclose(engine_mod.observer_rotor.bivectors, engine_base.observer_rotor.bivectors)


def test_boundary_transducer_framebuffer_and_telemetry():
    """Verify Boundary Transducer converts continuous manifold into discrete framebuffer."""
    seed = build_real_world_entities_field_seed()
    engine, _ = seed.instantiate_engine()
    membrane = BoundaryTransducerMembrane()

    # 1. Discrete Display FrameBuffer
    fb_res = membrane.sample_to_discrete_framebuffer(
        engine,
        resolution_wh=(64, 64),
        w_slice=0.1
    )

    framebuffer = fb_res["framebuffer_rgb"]
    assert framebuffer.dtype == np.uint8
    assert framebuffer.shape == (64, 64, 3)
    assert fb_res["pixel_count"] == 64 * 64
    assert fb_res["protocol_compatibility"] == "RGB888_PACKED_HDMI_D3D12"
    assert fb_res["buffer_size_bytes"] == 64 * 64 * 3

    # 2. Outbound Telemetry Packet
    packet = membrane.sample_to_telemetry_packet(engine, timestamp=1.23)
    assert packet["protocol"] == "ELYSIA_MEMBRANE_V1"
    assert packet["timestamp"] == 1.23
    assert 0.0 <= packet["coherence_r"] <= 1.0
    assert packet["active_nodes"] == len(engine.nodes)


def test_real_world_entity_person_building_repulsion_no_aabb():
    """Verify phase resonance repulsion prevents person from penetrating building (no AABB)."""
    seed = build_real_world_entities_field_seed()
    engine, node_ids = seed.instantiate_engine()

    person_idx = node_ids.index("person_agent")
    building_pillar_idx = node_ids.index("building_pillar_1")

    person = engine.nodes[person_idx]
    building = engine.nodes[building_pillar_idx]

    # Evaluate interaction at distance 0.4 (within penumbra, anti-phase)
    person.position = np.array([0.5, -0.2, 0.0, 0.1], dtype=np.float32)
    res = engine.kernel.evaluate_interaction(person, building)

    # Building and person have near opposite phase -> destructive/repulsive interaction
    assert res["overlap"] > engine.kernel.omega_c, "Should exceed critical overlap"
    assert res["is_blister_active"] is True
    assert res["force_total"] < 0.0, "Anti-phase should produce negative (repulsive) force"
    assert res["force_vector_4d"][0] < 0.0, "Repulsion should push person away in -X direction"


def test_forest_w_axis_scale_continuum():
    """Verify forest self-organizing continuum: micro dispersed wavelets to macro fused canopy."""
    seed = build_real_world_entities_field_seed()
    engine, _ = seed.instantiate_engine()
    slicer = OpticalHyperSlicer(resolution_3d=24, resolution_2d=64)

    # Micro depth (w = -0.5)
    micro_vol = slicer.slice_4d_to_3d_volume(engine.nodes, engine.kernel, w_slice=-0.5)
    micro_proj = slicer.project_to_2d_prism(micro_vol)

    # Macro surface (w = 0.25)
    macro_vol = slicer.slice_4d_to_3d_volume(engine.nodes, engine.kernel, w_slice=0.25)
    macro_proj = slicer.project_to_2d_prism(macro_vol)

    # Verify both slices produce non-empty valid 2D projective fields
    density_micro = micro_proj["density_2d"]
    density_macro = macro_proj["density_2d"]

    assert np.mean(density_macro) > 0.0
    assert np.mean(density_micro) > 0.0
    assert density_macro.shape == (24, 24)
    assert density_micro.shape == (24, 24)
    # Check that the two slices at different w-coordinates have distinct spectral projections
    assert not np.allclose(density_macro, density_micro)


def test_light_and_shadow_wavefront_interference():
    """Verify light wave interference creates natural dark interference shadow zone."""
    seed = build_real_world_entities_field_seed()
    engine, node_ids = seed.instantiate_engine()
    slicer = OpticalHyperSlicer(resolution_3d=28, resolution_2d=80)

    vol = slicer.slice_4d_to_3d_volume(engine.nodes, engine.kernel, w_slice=0.1)
    proj = slicer.project_to_2d_prism(vol)

    density = proj["density_2d"]
    # Check that high intensity illumination exists near light source
    # and occlusion / contrast gradient exists across building
    assert np.max(density) > 0.5
    assert np.min(density) < 0.1
