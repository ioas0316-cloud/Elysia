"""
Unit tests for AsuraCheonmu Causal Nexus Data Structure, 4-Stage Rendering Pipeline, and Shader Modules.
"""

import pytest
from core.engine.asura_nexus import (
    AsuraCheonmuNexus,
    NEXUS_BIT_ANNIHILATE,
    NEXUS_BIT_DOMAIN_LOCKED,
    NEXUS_BIT_HIT_ACTIVE,
    NEXUS_BIT_PALETTE_INVERTED,
    NEXUS_BIT_TRIGGERED,
)
from core.engine.asura_render_pipeline import AsuraRenderPipeline, MockGPUContext, MockShaderProgram
from modules.asura_nexus.shaders import get_asura_shader_code


def test_asura_nexus_instantiation_and_validation():
    nexus = AsuraCheonmuNexus()
    assert nexus.nexus_id == "SKILL_ASURA_CHEONMU"
    assert nexus.cause_vectors["required_weapon"] == "WEAPON_ASURA"
    assert len(nexus.spatial_trajectory["teleport_sequence"]) == 16
    assert nexus.hit_causality["hit_count"] == 16

    # Test activation validation
    valid = nexus.validate_activation(
        caster_sp=150, caster_tp=160, equipped_weapon="WEAPON_ASURA", current_state="STATE_NORMAL"
    )
    assert valid is True

    invalid_sp = nexus.validate_activation(
        caster_sp=50, caster_tp=160, equipped_weapon="WEAPON_ASURA", current_state="STATE_NORMAL"
    )
    assert invalid_sp is False


def test_asura_nexus_flat_bit_array():
    nexus = AsuraCheonmuNexus()
    state_flags = NEXUS_BIT_TRIGGERED | NEXUS_BIT_DOMAIN_LOCKED
    buf = nexus.to_flat_bit_array(state_flags)
    assert len(buf) == 64
    assert buf[0] == state_flags


def test_asura_render_pipeline_execution():
    gpu = MockGPUContext()
    shader = MockShaderProgram(gpu)
    pipeline = AsuraRenderPipeline(gpu, shader)
    nexus = AsuraCheonmuNexus()

    pipeline.dispatch_nexus_state(nexus, active=True)
    assert pipeline.bitmask_state & NEXUS_BIT_DOMAIN_LOCKED

    # Process ticks 0..15 (16 ticks)
    for tick in range(16):
        res = pipeline.render_frame(nexus, current_causal_tick=tick)
        assert res["hit_triggered"] is True

    # At tick 16, state should have reached ANNIHILATION
    assert bool(pipeline.bitmask_state & NEXUS_BIT_ANNIHILATE) is True
    assert gpu.uniforms["u_GlobalTimeScale"] == 0.0
    assert gpu.uniforms["u_RedBlackToggle"] == 1.0
    assert len(gpu.drawn_sprites) == 16 * 16  # 16 nodes per frame * 16 frames
    assert len(gpu.particle_bursts) == 16


def test_asura_shader_generation():
    std_shader = get_asura_shader_code(optimized=False)
    opt_shader = get_asura_shader_code(optimized=True)

    assert "u_EnableRedBlackPalette" in std_shader
    assert "u_RedBlackToggle" in opt_shader
    assert "u_PaletteLUT" in opt_shader
