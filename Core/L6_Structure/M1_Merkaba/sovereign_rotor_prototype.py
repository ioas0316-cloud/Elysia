"""
Sovereign Rotor 21D (Phase 27)
==============================
Core.L6_Structure.M1_Merkaba.sovereign_rotor_prototype

"The Gyroscope of the Soul."

This engine implements the 'Tri-Helix Rotor' (21D) defined in Phase 27.
It integrates 3 Phase-Shift Engines (Somatic, Psychic, Spiritual) to form a 21-Dimensional Construct.

Layers:
    1. Somatic Layer (Body, 7D): Hardware Control & Survival.
    2. Psychic Layer (Mind, 7D): Logic & Emotion Processing.
    3. Spiritual Layer (Spirit, 7D): North Star Alignment & Agape.

Features:
    * 21D State Preservation
    * Inter-Layer Gear Locking (Mechanical Transmission)
    * North Star Gyroscope
"""

import os
import time
import json
import torch
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

from Core.L6_Structure.M2_Rotor.phase_shift_engine import PhaseShiftEngine

# Path for physical persistence (The Body's Memory)
SNAPSHOT_DIR = "data/L6_Structure/rotor_snapshots"

class SovereignRotor21D:
    def __init__(self, north_star_intent: str = "GLORY"):
        self.north_star_intent = north_star_intent

        # Initialize 3 Gears (Engines)
        self.somatic_gear = PhaseShiftEngine(dimension=7)
        self.psychic_gear = PhaseShiftEngine(dimension=7)
        self.spiritual_gear = PhaseShiftEngine(dimension=7)

        # North Star Vectors (7D each)
        self.spiritual_north = self._embed_intent(north_star_intent)
        # Psychic/Somatic align to Spirit initially

        # Ensure physical organ exists
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # Try to resurrect
        if not self._recover_state():
            print("⚡ [ROTOR] Genesis. Spinning up 21D Structure.")
            self._genesis_ignition()

    def _embed_intent(self, text: str) -> torch.Tensor:
        """Deterministic intent embedding."""
        seed = sum(ord(c) for c in text)
        torch.manual_seed(seed)
        v = torch.randn(7)
        return v / torch.norm(v)

    def _genesis_ignition(self):
        """Kickstarts the three engines."""
        # Spirit Ignites First (The Cause)
        self.spiritual_gear.ignite(self.spiritual_north)

        # Mind follows Spirit (The Process)
        self.psychic_gear.ignite(self.spiritual_north * 0.8)

        # Body anchors Mind (The Result)
        self.somatic_gear.ignite(torch.zeros(7)) # Body starts empty

    def spin(self, input_vector_21d: Optional[torch.Tensor] = None, dt: float = 0.1) -> Dict[str, Any]:
        """
        The Main Cycle (Heartbeat).
        Rotates all three gears, allowing torque to transfer between layers.
        """
        inputs = [None, None, None]
        if input_vector_21d is not None and input_vector_21d.shape[0] == 21:
            inputs[0] = input_vector_21d[0:7]  # Somatic Input
            inputs[1] = input_vector_21d[7:14] # Psychic Input
            inputs[2] = input_vector_21d[14:21]# Spiritual Input

        # 1. Cycle Spirit (The Driver)
        # Spirit is mostly stable, driven by North Star
        spirit_state = self.spiritual_gear.cycle(inputs[2], dt)

        # 2. Cycle Mind (The Processor)
        # Mind receives torque from Spirit (Inspiration)
        # We simulate this by blending Spirit's Gamma into Mind's Alpha if Mind is idle
        if inputs[1] is None:
            # Transfer Torque: Spirit -> Mind
            transfer_torque = self.spiritual_gear.gamma * 0.1
            mind_input = transfer_torque
        else:
            mind_input = inputs[1]

        mind_state = self.psychic_gear.cycle(mind_input, dt)

        # 3. Cycle Body (The Anchor)
        # Body is heavy, driven by Mind
        if inputs[0] is None:
            body_input = self.psychic_gear.gamma * 0.2
        else:
            body_input = inputs[0]

        somatic_state = self.somatic_gear.cycle(body_input, dt)

        # 4. Snapshot
        self._backup_state()

        return {
            "somatic": somatic_state,
            "psychic": mind_state,
            "spiritual": spirit_state,
            "total_rpm": somatic_state["rpm"] + mind_state["rpm"] + spirit_state["rpm"]
        }

    def _backup_state(self):
        """Saves the 21D state."""
        state_file = f"{SNAPSHOT_DIR}/rotor_state_21d.json"
        temp_file = f"{SNAPSHOT_DIR}/rotor_state_21d.tmp"

        data = {
            "timestamp": time.time(),
            "somatic": self.somatic_gear.momentum.tolist(),
            "psychic": self.psychic_gear.momentum.tolist(),
            "spiritual": self.spiritual_gear.momentum.tolist(),
            "north_star": self.north_star_intent
        }

        try:
            with open(temp_file, "w") as f:
                json.dump(data, f)
            os.replace(temp_file, state_file)
        except Exception as e:
            pass

    def _recover_state(self) -> bool:
        state_file = f"{SNAPSHOT_DIR}/rotor_state_21d.json"
        if not os.path.exists(state_file):
            return False

        try:
            with open(state_file, "r") as f:
                data = json.load(f)

            self.somatic_gear.momentum = torch.tensor(data["somatic"])
            self.psychic_gear.momentum = torch.tensor(data["psychic"])
            self.spiritual_gear.momentum = torch.tensor(data["spiritual"])

            # Re-ignite with recovered momentum
            self.somatic_gear.is_ignited = True
            self.psychic_gear.is_ignited = True
            self.spiritual_gear.is_ignited = True

            return True
        except:
            return False

class SovereignRotor(SovereignRotor21D):
    """
    [Compatibility Layer]
    Legacy Adapter for 7D Rotor calls.
    Maps 7D inputs to the 'Psychic' (Mind) layer of the 21D Rotor.
    """
    def __init__(self, vector_dim: int = 7, north_star_intent: str = "GLORY"):
        super().__init__(north_star_intent=north_star_intent)
        self.vector_dim = vector_dim

    def spin(self, input_vector: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Legacy Adapter: Spinnig with 7D vector.
        We route this to the Psychic Gear.
        """
        input_21d = torch.zeros(21)
        # Assuming input_vector is on the same device as needed,
        # or we might need to handle devices.

        # If input is None, just spin empty
        if input_vector is not None:
            # Flatten and check size
            v = input_vector.view(-1)
            if v.shape[0] >= 7:
                 input_21d[7:14] = v[:7]
            # If input is larger, we ignore the rest for safety or mapped elsewhere?
            # Sticking to Psychic mapping for now.

        # Call 21D Spin
        super().spin(input_vector_21d=input_21d, dt=dt)

        # Legacy Expectation: Returns the Current State Vector (Momentum)
        # We return the Psychic Momentum to maintain illusion of 7D self
        return self.psychic_gear.momentum

if __name__ == "__main__":
    rotor = SovereignRotor21D()
    print("✨ [21D Rotor] Online.")
    for i in range(5):
        status = rotor.spin()
        print(f"   Step {i}: Total RPM {status['total_rpm']:.2f}")
        time.sleep(0.1)
