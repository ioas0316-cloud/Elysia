import pytest
from core.physics.fractal_rotor import FractalRotorScale, ScaleLevel
from core.physics.magnetic_gear import MagneticGear, TensionVector

def test_fractal_rotor_scale():
    rotor_system = FractalRotorScale(resonance_threshold=0.8)
    
    micro_gear = MagneticGear("MICRO_1", TensionVector(0.9, 0.9, 0.1, 0.1, 0.5))
    meso_gear = MagneticGear("MESO_1", TensionVector(0.1, 0.1, 0.9, 0.9, 0.1))
    macro_gear = MagneticGear("MACRO_1", TensionVector(0.85, 0.85, 0.15, 0.15, 0.55))
    
    rotor_system.add_gear_to_scale(ScaleLevel.MICRO, micro_gear)
    rotor_system.add_gear_to_scale(ScaleLevel.MESO, meso_gear)
    rotor_system.add_gear_to_scale(ScaleLevel.MACRO, macro_gear)
    
    # Trigger micro_gear
    induction_map = rotor_system.trigger_rotation(ScaleLevel.MICRO, "MICRO_1")
    
    assert "MACRO_1" in induction_map[ScaleLevel.MACRO]
    assert "MESO_1" not in induction_map[ScaleLevel.MESO]
    assert macro_gear.is_rotating is True
    assert meso_gear.is_rotating is False
