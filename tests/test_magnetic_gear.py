import pytest
from core.physics.magnetic_gear import MagneticGear, KinematicInduction, TensionVector

def test_magnetic_gear_resonance():
    induction = KinematicInduction(resonance_threshold=0.8)
    
    gear_a = MagneticGear("A", TensionVector(0.9, 0.9, 0.1, 0.1, 0.5))
    gear_b = MagneticGear("B", TensionVector(0.85, 0.85, 0.15, 0.15, 0.55))
    gear_c = MagneticGear("C", TensionVector(0.1, 0.1, 0.9, 0.9, 0.1))
    
    res_ab = induction.calculate_resonance(gear_a, gear_b)
    res_ac = induction.calculate_resonance(gear_a, gear_c)
    
    assert res_ab.total_resonance >= 0.8
    assert res_ac.total_resonance < 0.5

def test_kinematic_induction():
    induction = KinematicInduction(resonance_threshold=0.8)
    
    gear_a = MagneticGear("A", TensionVector(0.9, 0.9, 0.1, 0.1, 0.5))
    gear_b = MagneticGear("B", TensionVector(0.85, 0.85, 0.15, 0.15, 0.55))
    gear_c = MagneticGear("C", TensionVector(0.1, 0.1, 0.9, 0.9, 0.1))
    
    induction.add_gear(gear_a)
    induction.add_gear(gear_b)
    induction.add_gear(gear_c)
    
    gear_a.turn()
    induced = induction.propagate_rotation("A")
    
    assert "B" in induced
    assert "C" not in induced
    assert gear_b.is_rotating is True
    assert gear_c.is_rotating is False
