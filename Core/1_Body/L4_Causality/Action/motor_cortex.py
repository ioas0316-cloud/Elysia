"""
Motor Cortex (The Digital Muscle)
=================================
Core.1_Body.L4_Causality.Action.motor_cortex

"Thoughts without Action are just Dreams. Action without Thought is Reflex."

This module implements the Hardware-Native Motor Control system (Phase 7.3).
It maps the Virtual Rotors (Time/Concepts) to Physical Actuators (Space/Motion).
It also enforces the "Biological Safety Protocol" (Pain = Freeze).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

# Type hint for NervousSystem to avoid circular imports if not needed at runtime immediately
# from Core.1_Body.L6_Structure.Elysia.nervous_system import NervousSystem, BioSignal

logger = logging.getLogger("MotorCortex")

@dataclass
class ActuatorConfig:
    """Configuration for a physical actuator."""
    pin: int                 # GPIO Pin or Channel
    min_val: float = 0.0     # Min Pulse/Angle
    max_val: float = 180.0   # Max Pulse/Angle
    neutral_val: float = 90.0
    hw_type: str = "SERVO"   # SERVO, DC_MOTOR, STEPPER

class Actuator:
    """
    The Physical Muscle Fiber.
    Wraps low-level hardware drivers (RPi.GPIO, Pigpio, etc).
    For this implementation, we simulate the hardware.
    """
    def __init__(self, name: str, config: ActuatorConfig):
        self.name = name
        self.config = config
        self.current_val = config.neutral_val
        self.is_active = True
        logger.info(f"     Actuator [{name}] connected on Pin {config.pin}.")

    def move(self, value: float):
        """
        Commands the physical hardware to move.
        """
        if not self.is_active:
            return

        # Clamping
        safe_val = max(self.config.min_val, min(value, self.config.max_val))

        # Hardware Interfacing (Mock)
        if safe_val != self.current_val:
            # In a real scenario, this would be: pwm.set_servo_pulsewidth(pin, val)
            # logger.debug(f"      [{self.name}] Moving to {safe_val:.1f}")
            self.current_val = safe_val

    def freeze(self):
        """Emergency Stop / Hold Position."""
        # For servos, maybe return to neutral or stop sending PWM?
        # We'll assume 'freeze' means stop updating or go to neutral.
        # logger.warning(f"      [{self.name}] FROZEN.")
        pass

class MotorCortex:
    """
    The High-Level Motion Controller.
    Orchestrates Actuators based on Rotor inputs and Nervous System constraints.
    """
    def __init__(self, nervous_system=None):
        self.actuators: Dict[str, Actuator] = {}
        self.nervous_system = nervous_system
        self.is_paralyzed = False # Global safety lock
        logger.info("  Motor Cortex initialized.")

    def register_actuator(self, rotor_name: str, pin: int):
        """
        Maps a Virtual Rotor (Concept) to a Physical Actuator (Pin).
        """
        config = ActuatorConfig(pin=pin)
        actuator = Actuator(f"Muscle.{rotor_name}", config)
        self.actuators[rotor_name] = actuator

    def drive(self, rotor_name: str, rpm: float):
        """
        The Conscious Act of Moving.
        Maps Rotor RPM to Actuator Position/Speed.
        """
        if rotor_name not in self.actuators:
            return

        actuator = self.actuators[rotor_name]

        # 1. Safety Check (The Nervous Impulse)
        if self.nervous_system:
            # We assume nervous_system.sense() returns a BioSignal
            # Note: In a high-freq loop, we might cache this or read a flag.
            signal = self.nervous_system.sense()

            if signal.is_painful:
                if not self.is_paralyzed:
                    logger.warning(f"  [PAIN REFLEX] High Pain ({signal.pain_level:.2f}). Freezing Motor Cortex.")
                    self.is_paralyzed = True
            else:
                if self.is_paralyzed:
                    logger.info("  [RECOVERY] Pain subsided. Motor Cortex unlocked.")
                    self.is_paralyzed = False

        if self.is_paralyzed:
            actuator.freeze()
            return

        # 2. Map RPM to Motion
        # Logic:
        # RPM = 0 -> Neutral (90)
        # RPM > 0 -> Move towards Max (180)
        # RPM < 0 -> Move towards Min (0)
        # This is a "Position Control" mapping based on RPM intensity?
        # Or is RPM "Speed"? For a continuous servo, RPM is Speed.
        # For a positional servo, RPM usually maps to "Rate of change of position" or "Target Position".
        # Let's assume standard 180 servo acts as a gauge for 'Intensity' or 'Direction'.

        # Simple Mapping: Angle = Neutral + (RPM * Scale)
        # 10 RPM -> +10 degrees
        scale_factor = 1.0
        target_angle = actuator.config.neutral_val + (rpm * scale_factor)

        actuator.move(target_angle)

    def status(self) -> Dict[str, float]:
        """Returns current state of all muscles."""
        return {name: actuator.current_val for name, actuator in self.actuators.items()}
