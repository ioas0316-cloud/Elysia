r"""
test_sangsaeng_sanggeuk_field.py
=================================
Unit tests for `core/physics/sangsaeng_sanggeuk_field.py`

Verifies:
1. Sangsaeng (Attraction & Void Complementarity) vs Sanggeuk (Repulsion & Heat Generation).
2. Relational Network Tension, beam stretching, vibration propagation, and tearing.
3. Magnetism (Potential Well & orbital attraction) and Torque (Constructive vs Destructive Interference).
4. User Non-deterministic Perturbation (breaking static equilibrium symmetry).
5. Dual-Axis Scale & Phase Friction Engine (Scale Twisting, Void Tension Energy relaxation, abductive invariant condensation).
"""

import pytest
import math
import numpy as np
from core.physics.sangsaeng_sanggeuk_field import (
    SangsaengSanggeukField,
    DynamicEntity,
    RelationalBeam
)

def test_sangsaeng_attraction_and_void_complementarity():
    field = SangsaengSanggeukField()

    # Warrior has void deficit in Order (index 1), Priest provides Order in chromatic vector
    warrior = DynamicEntity(
        id="warrior",
        name="ShieldWarrior",
        faction="Alliance",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        void_deficit=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        intent_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    priest = DynamicEntity(
        id="priest",
        name="HealPriest",
        faction="Alliance",
        position=np.array([4.0, 0.0, 0.0], dtype=np.float32),
        chromatic_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        intent_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    field.add_entity(warrior)
    field.add_entity(priest)

    forces, heat = field.compute_sangsaeng_sanggeuk_forces()

    # Warrior should feel an attractive force pulling towards Priest (+X direction)
    assert forces["warrior"][0] > 0.0
    # Priest should feel an attractive force pulling towards Warrior (-X direction)
    assert forces["priest"][0] < 0.0
    # Friction heat should be low for Sangsaeng
    assert heat < 0.1

def test_sanggeuk_repulsion_and_friction_heat():
    field = SangsaengSanggeukField()

    # Opposing factions with opposing intent vectors
    guild_a = DynamicEntity(
        id="guild_a",
        name="GuildA",
        faction="RedGuild",
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        intent_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        chromatic_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )

    guild_b = DynamicEntity(
        id="guild_b",
        name="GuildB",
        faction="BlueGuild",
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        velocity=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
        intent_vector=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        chromatic_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    field.add_entity(guild_a)
    field.add_entity(guild_b)

    forces, heat = field.compute_sangsaeng_sanggeuk_forces()

    # Guild A pushed away (-X direction)
    assert forces["guild_a"][0] < 0.0
    # Guild B pushed away (+X direction)
    assert forces["guild_b"][0] > 0.0
    # Sanggeuk generates significant friction heat
    assert heat > 0.1

def test_relational_beam_tension_and_tearing():
    field = SangsaengSanggeukField()

    node_1 = DynamicEntity("n1", "Node1", position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    node_2 = DynamicEntity("n2", "Node2", position=np.array([20.0, 0.0, 0.0], dtype=np.float32)) # Stretched dist 20 - rest 5 = extension 15 > threshold 10

    field.add_entity(node_1)
    field.add_entity(node_2)
    field.link_entities("n1", "n2", strength=1.0, rest_length=5.0)

    # Initial beam is intact
    assert len(field.beams) == 1
    assert not field.beams[0].is_broken

    # Step simulation - tension exceeds break_threshold 10.0, causing beam tear
    report = field.step(dt=0.1)

    assert report["broken_beams"] == 1
    assert report["clash_logs_count"] >= 1

def test_magnetism_and_torque_dynamics():
    field = SangsaengSanggeukField()
    field.macro_flow_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Macro flow along +Y

    # Magnet center (Hero) with high magnetic_mass
    hero = DynamicEntity("hero", "HeroHero", magnetic_mass=3.0, position=np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # Follower 1 aligned with macro flow (+Y) -> Constructive interference
    follower_aligned = DynamicEntity(
        "f_aligned",
        "AlignedFollower",
        position=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        intent_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    # Follower 2 opposed to macro flow (-Y) -> Destructive interference & Torque
    follower_opposed = DynamicEntity(
        "f_opposed",
        "OpposedFollower",
        position=np.array([-5.0, 0.0, 0.0], dtype=np.float32),
        intent_vector=np.array([0.0, -1.0, 0.0], dtype=np.float32)
    )

    field.add_entity(hero)
    field.add_entity(follower_aligned)
    field.add_entity(follower_opposed)

    mag_forces, torque_reports = field.compute_magnetism_and_torque(dt=0.1)

    # Followers pulled towards magnet hero
    assert mag_forces["f_aligned"][0] < 0.0 # Towards x=0
    assert mag_forces["f_opposed"][0] > 0.0 # Towards x=0

    # Step physics to trigger wave interference effects
    initial_amp_aligned = follower_aligned.amplitude
    initial_amp_opposed = follower_opposed.amplitude

    field.step(dt=0.1)

    # Aligned amplitude increased (Constructive)
    assert follower_aligned.amplitude > initial_amp_aligned
    # Opposed amplitude decreased (Destructive)
    assert follower_opposed.amplitude < initial_amp_opposed

def test_user_non_deterministic_perturbation():
    field = SangsaengSanggeukField()

    player = DynamicEntity("player", "UserPlayer", is_player=True, position=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    npc = DynamicEntity("npc", "ObserverNPC", position=np.array([3.0, 0.0, 0.0], dtype=np.float32))

    field.add_entity(player)
    field.add_entity(npc)
    field.link_entities("player", "npc", strength=1.0)

    # Before user action: field is in static equilibrium
    assert field.static_equilibrium is True

    # User throws perturbation stone into static water
    res = field.throw_user_perturbation(
        player_id="player",
        perturbation_vector=np.array([10.0, 5.0, 0.0], dtype=np.float32),
        speech_or_action="I shatter this peaceful order!"
    )

    assert res["status"] == "symmetry_broken"
    assert field.static_equilibrium is False
    assert field.total_perturbation_energy > 0.0
    assert len(field.clash_logs) >= 1

def test_dual_axis_scale_twist_and_void_relaxation():
    field = SangsaengSanggeukField()

    e1 = DynamicEntity("e1", "Entity1", scale_exponent=0.0, phase=0.0)
    e2 = DynamicEntity("e2", "Entity2", scale_exponent=0.5, phase=0.2)

    field.add_entity(e1)
    field.add_entity(e2)

    relaxation_report = field.apply_scale_twist_and_void_relaxation(dt=0.1)

    assert "void_energy" in relaxation_report
    assert "converged" in relaxation_report

    # Step multiple times to test Kuramoto phase relaxation convergence
    for _ in range(20):
        rep = field.step(dt=0.1)

    # Void tension energy should decrease as phases relax
    assert rep["void_tension_energy"] < 0.3
