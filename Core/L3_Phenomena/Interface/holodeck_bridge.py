"""
Holodeck Bridge (OSC/UDP)
=========================
"The projection of the internal world onto the external canvas."

This module implements a lightweight OSC (Open Sound Control) sender
to transmit Elysia's internal state (Rotor Physics, Emotions, Thoughts)
to a visualization engine like Unity or Unreal Engine.
"""

import socket
import struct
import json
import time
import logging
from typing import Any, List, Union

logger = logging.getLogger("HolodeckBridge")

class HolodeckBridge:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_connected = True # UDP is connectionless, so conceptually true
        logger.info(f"  [HOLODECK] Bridge initialized targeting {ip}:{port}")

    def send_osc(self, address: str, args: List[Any]):
        """
        Sends a simplified OSC message.
        Format: /address ,types values
        """
        try:
            msg = self._build_osc_message(address, args)
            self.socket.sendto(msg, (self.ip, self.port))
        except Exception as e:
            logger.error(f"  OSC Send Failed: {e}")

    def _build_osc_message(self, address: str, args: List[Any]) -> bytes:
        """
        Constructs a binary OSC message.
        """
        # 1. Address Pattern (padded to 4 bytes)
        msg = self._pad_string(address)
        
        # 2. Type Tag String (padded to 4 bytes)
        type_tags = ","
        values = b""
        
        for arg in args:
            if isinstance(arg, int):
                type_tags += "i"
                values += struct.pack(">i", arg)
            elif isinstance(arg, float):
                type_tags += "f"
                values += struct.pack(">f", arg)
            elif isinstance(arg, str):
                type_tags += "s"
                values += self._pad_string(arg)
            else:
                # Fallback for others (bool, etc) -> string
                type_tags += "s"
                values += self._pad_string(str(arg))
                
        msg += self._pad_string(type_tags)
        msg += values
        return msg

    def _pad_string(self, s: str) -> bytes:
        """OSC strings must be null-terminated and padded to 4-byte boundaries."""
        b_str = s.encode('utf-8')
        length = len(b_str)
        pad_len = 4 - (length % 4)
        return b_str + (b'\x00' * pad_len)

    # --- High Level Projection Methods ---

    def broadcast_rotor(self, name: str, rpm: float, phase: float, energy: float):
        """[Legacy] Sends Rotor Physics Data (2D/3D)."""
        self.send_osc("/elysia/rotor", [name, float(rpm), float(phase), float(energy)])

    def broadcast_rotor_4d(self, name: str, quat: tuple, rpm: float, energy: float):
        """
        [PHASE 85] Sends 4D Hyper-Rotor Physics Data.
        Unity should map this to:
        - Rotation: quat (x,y,z,w)
        - Scale/Color: energy/rpm
        """
        # Address: /elysia/rotor_4d
        # Args: [Name, Qx, Qy, Qz, Qw, RPM, Energy]
        # OSC standard usually expects floats
        args = [name, float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]), float(rpm), float(energy)]
        self.send_osc("/elysia/rotor_4d", args)

    def broadcast_thought(self, content: str, mood: str, intensity: float):
        """Sends Thought/Spark Data."""
        # Address: /elysia/thought
        self.send_osc("/elysia/thought", [content, mood, float(intensity)])

    def broadcast_bio_rhythm(self, heart_rate: float, stress: float, peace: float):
        """Sends Biological State."""
        self.send_osc("/elysia/bio", [float(heart_rate), float(stress), float(peace)])
        
    def broadcast_core_state(self, sovereignty: float, active_dna_count: int):
        """Sends High-Level Core Stats."""
        self.send_osc("/elysia/core", [float(sovereignty), int(active_dna_count)])

# Singleton
_bridge = None
def get_holodeck_bridge():
    global _bridge
    if _bridge is None:
        _bridge = HolodeckBridge()
    return _bridge