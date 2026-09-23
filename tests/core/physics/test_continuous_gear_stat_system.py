"""
test_continuous_gear_stat_system.py
===================================
Unit tests for the Continuous Gear Transmission Stat System.
"""

import math
import pytest
from core.physics.continuous_gear_stat_system import (
    FixedGear,
    VariableGear,
    GearTransmissionSystem
)


def test_fixed_gear_ratio_and_inertia():
    fg = FixedGear(base_level=1, growth_exponent=1.2, base_inertia=10.0)
    assert fg.current_level == 1
    assert math.isclose(fg.get_fixed_ratio(), 1.0)
    assert math.isclose(fg.get_moment_of_inertia(), 10.0)

    # Level up to 10
    fg.set_level(10)
    expected_ratio = math.pow(10.0, 1.2)
    assert math.isclose(fg.get_fixed_ratio(), expected_ratio)
    assert math.isclose(fg.get_moment_of_inertia(), 10.0 * expected_ratio)


def test_variable_gear_equipping_and_clutch():
    vg = VariableGear(shift_speed=10.0, initial_ratio=0.0)
    assert vg.current_ratio == 0.0
    assert vg.target_ratio == 0.0  # Initialized with initial_ratio = 0.0

    # Equip weapon with ratio 2.0
    vg.equip_weapon(2.0)
    assert vg.target_ratio == 2.0

    # Unequip weapon -> target ratio becomes 0.0 (Clutch Disengage)
    vg.unequip_weapon()
    assert vg.target_ratio == 0.0

    # Step forward -> ratio smoothly decays to 0
    vg.tick(1.0)
    assert vg.current_ratio == 0.0
    assert vg.get_effective_ratio() == 0.0
    assert vg.is_clutch_engaged is False


def test_variable_gear_overdrive_and_friction():
    vg = VariableGear(shift_speed=100.0, initial_ratio=1.0)
    vg.equip_weapon(1.5)
    vg.set_overdrive_buff(1.5)  # Overdrive region: 1.5 * 1.5 = 2.25
    vg.tick(0.1)

    assert math.isclose(vg.target_ratio, 2.25, rel_tol=1e-3)
    assert math.isclose(vg.current_ratio, 2.25, rel_tol=1e-3)

    # Apply 20% friction debuff
    vg.set_friction_debuff(0.2)
    expected_effective = 2.25 * (1.0 - 0.2)
    assert math.isclose(vg.get_effective_ratio(), expected_effective, rel_tol=1e-3)


def test_smooth_shift_damping_continuity():
    vg = VariableGear(shift_speed=2.0, initial_ratio=1.0)
    vg.equip_weapon(3.0)

    prev_ratio = vg.current_ratio
    dt = 0.1
    # Check that transition is monotonic and smooth over time
    for _ in range(10):
        vg.tick(dt)
        assert vg.current_ratio > prev_ratio
        assert vg.current_ratio <= 3.0
        prev_ratio = vg.current_ratio


def test_gear_transmission_scalar_derived_stat():
    fg = FixedGear(base_level=1, growth_exponent=1.0)  # Level 1 -> R_fixed = 1.0
    fg.set_level(5)  # Level 5 -> R_fixed = 5.0

    vg = VariableGear(shift_speed=100.0, initial_ratio=2.0)
    vg.equip_weapon(2.0)
    vg.tick(0.1)  # R_var = 2.0

    system = GearTransmissionSystem(base_flux=100.0, fixed_gear=fg, variable_gear=vg)

    # Derived Output Torque = BaseFlux(100) * R_fixed(5.0) * R_var(2.0) = 1000.0
    output_torque = system.evaluate_output_torque()
    assert isinstance(output_torque, float)
    assert math.isclose(output_torque, 1000.0, rel_tol=1e-3)


def test_gear_transmission_vector_derived_stat():
    fg = FixedGear(base_level=1, growth_exponent=1.0)
    fg.set_level(2)  # R_fixed = 2.0

    vg = VariableGear(shift_speed=100.0, initial_ratio=1.5)
    vg.equip_weapon(1.5)
    vg.tick(0.1)  # R_var = 1.5

    base_vector = {
        "attack_power": 50.0,
        "health_pool": 200.0,
        "movement_speed": 10.0
    }

    system = GearTransmissionSystem(base_flux=base_vector, fixed_gear=fg, variable_gear=vg)

    output = system.evaluate_output_torque()
    assert isinstance(output, dict)
    # Total Transmission Ratio = 2.0 * 1.5 = 3.0
    assert math.isclose(output["attack_power"], 150.0, rel_tol=1e-3)
    assert math.isclose(output["health_pool"], 600.0, rel_tol=1e-3)
    assert math.isclose(output["movement_speed"], 30.0, rel_tol=1e-3)

    # System Diagnostics
    diag = system.get_system_diagnostics()
    assert diag["level"] == 2.0
    assert diag["R_fixed"] == 2.0
    assert diag["R_var_effective"] == 1.5
    assert diag["output_torque_total"] == 780.0
