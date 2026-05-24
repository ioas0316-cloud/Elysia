"""
Elysia Game Actuation Engine (WoW Azeroth Bot)
==============================================
Uses screen-sensing (vision_utils), quaternion interpolation (math_utils),
and failsafe keyboard press actuation (actuator_utils) to pilot the character.
"""

import os
import time
import pickle
import random
import numpy as np
from datetime import datetime

from core.math_utils import Quaternion
from core.vision_utils import ScreenSenser
from core.actuator_utils import Actuator

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

if not os.path.exists(COLLAPSE_DIR):
    os.makedirs(COLLAPSE_DIR)

class SubRotor:
    def __init__(self, rotor_id: int, initial_quat: Quaternion):
        self.id = rotor_id
        self.quat = initial_quat

    def slerp_to(self, target_quat: Quaternion, energy: float):
        try:
            self.quat = Quaternion.slerp(self.quat, target_quat, min(1.0, max(0.0, energy)))
        except ZeroDivisionError:
            pass

class GameBotEngine:
    """
    Pilots the game character based on screen dynamics and internal state rotations.
    """
    def __init__(self, name="Elysia_Floating_Core"):
        self.name = name
        self.internal_quat = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.is_locked = True
        self.history = "0"
        self.cycle_count = 0
        self.fractal_depth = 1

        self.trajectory_memory = []
        self.MAX_TRAJECTORY_LENGTH = 100

        self.sub_rotors = [SubRotor(i, self.internal_quat) for i in range(5)]
        self.senser = ScreenSenser()
        self.actuator = Actuator()

    def get_external_weather(self) -> tuple[Quaternion, np.ndarray, str, float]:
        """Reads screen pixels to get the current external environmental vector and weather category."""
        frame = self.senser.grab_screen()
        vision_entropy, death_detected = self.senser.process_vision_chaos(frame)

        # Set Z-axis dominant from screen entropy, random small perturbations on X, Y
        x_axis = random.random() * 0.2
        y_axis = random.random() * 0.2
        z_axis = vision_entropy

        weather_vector = np.array([x_axis, y_axis, z_axis])
        norm = np.linalg.norm(weather_vector)
        if norm == 0:
            weather_vector = np.array([0.0, 0.0, 1.0])
            norm = 1.0

        axis = weather_vector / norm
        
        # Calculate external quaternion based on current perspective rotation
        relative_external_quat = Quaternion(
            w=math.cos(vision_entropy * math.pi / 2.0),
            x=axis[0] * math.sin(vision_entropy * math.pi / 2.0),
            y=axis[1] * math.sin(vision_entropy * math.pi / 2.0),
            z=axis[2] * math.sin(vision_entropy * math.pi / 2.0)
        )

        if death_detected:
            weather_type = "DEATH"
            base_chaos = 1.0
        elif vision_entropy > 0.5:
            weather_type = "Thunder"
            base_chaos = vision_entropy
        elif vision_entropy > 0.1:
            weather_type = "Cloudy"
            base_chaos = vision_entropy
        else:
            weather_type = "Clear"
            base_chaos = vision_entropy

        return relative_external_quat, weather_vector, weather_type, base_chaos

    def calculate_entropy(self) -> float:
        if len(self.trajectory_memory) < 2:
            return 0.0
        angles = []
        for i in range(1, len(self.trajectory_memory)):
            q1 = self.trajectory_memory[i-1]
            q2 = self.trajectory_memory[i]
            angles.append(Quaternion.distance(q1, q2))
        return np.var(angles) if angles else 0.0

    def process_tick(self):
        self.cycle_count += 1

        relative_external_quat, weather_vector, weather_type, base_chaos = self.get_external_weather()

        # Calculate mismatch angle as rotation distance
        target_quat = self.internal_quat * relative_external_quat
        mismatch = Quaternion.distance(self.internal_quat, target_quat)

        self.trajectory_memory.append(relative_external_quat)

        # Handle trajectory compression if limit is reached
        if len(self.trajectory_memory) >= self.MAX_TRAJECTORY_LENGTH:
            entropy = self.calculate_entropy()
            if entropy > 0.05:
                self.fold_dimensions(entropy)
            else:
                self.trajectory_memory = self.trajectory_memory[-50:]

        old_quat = Quaternion(self.internal_quat.w, self.internal_quat.x, self.internal_quat.y, self.internal_quat.z)

        if weather_type == "DEATH" or mismatch > 1.5:
            self.trigger_collapse(relative_external_quat, mismatch, weather_vector, weather_type)
        elif weather_type == "Cloudy" or mismatch > 0.5:
            self.soft_interference(relative_external_quat, base_chaos)
        else:
            # Maintain resonance (no significant change)
            pass

        # Calculate change and actuate keyboard
        rotation_diff = old_quat.inverse * self.internal_quat
        self.actuator.execute_rotation_action(rotation_diff)

    def fold_dimensions(self, entropy: float):
        self.fractal_depth += 1
        avg_quat = self.internal_quat
        for q in self.trajectory_memory:
            avg_quat = Quaternion.slerp(avg_quat, self.internal_quat * q, 0.1)

        self.internal_quat = avg_quat
        self.trajectory_memory = []
        self.history = f"Folded_{self.fractal_depth}D"

        collapse_data = {
            "type": "DIMENSION_FOLDING",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "new_depth": self.fractal_depth,
            "entropy": entropy,
            "chaos_source": "Optical_Flow_Overload"
        }
        filename = os.path.join(COLLAPSE_DIR, f"folding_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

    def trigger_collapse(self, relative_external_quat: Quaternion, mismatch: float, weather_vector: np.ndarray, weather_type: str):
        self.is_locked = False
        self.history = "1"

        collapse_data = {
            "type": "DEATH_COLLAPSE" if weather_type == "DEATH" else "THUNDER_COLLAPSE",
            "time": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "internal_quat": self.internal_quat.elements,
            "external_quat": (self.internal_quat * relative_external_quat).elements,
            "weather_vector": weather_vector.tolist(),
            "mismatch": mismatch,
            "chaos_source": "Visual_Death/High_Impact",
            "sub_rotors": [sr.quat.elements for sr in self.sub_rotors]
        }
        filename = os.path.join(COLLAPSE_DIR, f"collapse_{int(time.time())}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(collapse_data, f)

        # Panic response: bounce violently in phase space
        random_axis = np.random.rand(3)
        random_axis /= np.linalg.norm(random_axis)
        
        # Create 90-degree twist quaternion
        half_angle = math.pi / 4.0
        panic_quat = Quaternion(
            w=math.cos(half_angle),
            x=random_axis[0] * math.sin(half_angle),
            y=random_axis[1] * math.sin(half_angle),
            z=random_axis[2] * math.sin(half_angle)
        )
        target_new_quat = self.internal_quat * panic_quat

        for sr in self.sub_rotors:
            # Add random wobble to sub rotors
            wobble_axis = np.random.rand(3)
            wobble_axis /= np.linalg.norm(wobble_axis)
            wobble_angle = random.random() * math.pi
            wobble_quat = Quaternion(
                w=math.cos(wobble_angle/2.0),
                x=wobble_axis[0] * math.sin(wobble_angle/2.0),
                y=wobble_axis[1] * math.sin(wobble_angle/2.0),
                z=wobble_axis[2] * math.sin(wobble_angle/2.0)
            )
            sr.quat = sr.quat * wobble_quat
            sr.slerp_to(target_new_quat, 0.9)

        self.internal_quat = target_new_quat
        self.is_locked = True
        self.history = "0"

    def soft_interference(self, relative_external_quat: Quaternion, base_chaos: float):
        self.is_locked = False
        self.history = "1"

        target_new_quat = Quaternion.slerp(self.internal_quat, self.internal_quat * relative_external_quat, 0.3)
        energy = 0.2 * base_chaos

        for sr in self.sub_rotors:
            sr.slerp_to(target_new_quat, energy)

        # Average sub-rotor states (using sequential slerp)
        avg_quat = self.sub_rotors[0].quat
        for i in range(1, len(self.sub_rotors)):
            avg_quat = Quaternion.slerp(avg_quat, self.sub_rotors[i].quat, 1.0 / (i + 1))

        self.internal_quat = avg_quat
        self.is_locked = True
        self.history = "0"

def save_state(engine: GameBotEngine):
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(engine, f)

def load_state() -> GameBotEngine:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return GameBotEngine()

def run_loop():
    print("=========================================================")
    print("  Elysia Game-Actuation Bot starting... (WoW Pilot)")
    print("  Failsafe active: Throw mouse cursor to corner to quit.")
    print("=========================================================")
    
    bot = load_state()
    try:
        while True:
            bot.process_tick()
            save_state(bot)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nShutdown command received. Saving engine state...")
        save_state(bot)
    except Exception as e:
        print(f"\nEngine error occurred: {e}")
        save_state(bot)

if __name__ == "__main__":
    run_loop()
