"""
Elysia Geodesic Freezing & Resonance Interface (Layer 4)
========================================================
This module is strictly separated from the gearboxes to act as an independent interface.
It DOES NOT store static JSON or decoupled dict values.
Instead, it encodes the continuous phase trajectory (Geodesic) of the Wye neutral points
and the corresponding rotor tensions.

When 'resonate_load' is called, it does not parse config lines. It injects the tensor stream
back into the AtlantisCliffordSystem, allowing the system's own Y-Delta electromagnetic
restoring forces to snap the 10 layers back into geometric synchronization.
"""

import math
import struct
import time
from typing import List, Dict, Tuple
from core.math_utils import Multivector
from core.atlantis_clifford_bridge import AtlantisCliffordSystem

class GeodesicFreezer:
    def __init__(self, storage_path: str = "data/geodesic_trajectory.bin"):
        self.storage_path = storage_path
        # Temporary buffer for continuous wave trajectory before freezing to disk
        self._trajectory_stream = bytearray()

    def _multivector_to_bytes(self, mv: Multivector) -> bytes:
        """
        Encodes a Multivector directly into a binary wave format.
        Format per element: [uint32: mask] [float64: magnitude]
        """
        stream = bytearray()
        # Header: Number of active blades
        stream.extend(struct.pack('I', len(mv.data)))
        for mask, val in mv.data.items():
            stream.extend(struct.pack('Id', mask, val))
        return bytes(stream)

    def _bytes_to_multivector(self, data: bytes, offset: int, signature: Tuple[int, int]) -> Tuple[Multivector, int]:
        """
        Decodes a binary wave format back into a Multivector.
        Returns the Multivector and the new offset.
        """
        if offset >= len(data):
            return Multivector({}, signature), offset

        num_blades = struct.unpack_from('I', data, offset)[0]
        offset += struct.calcsize('I')

        mv_data = {}
        for _ in range(num_blades):
            mask, val = struct.unpack_from('Id', data, offset)
            mv_data[mask] = val
            offset += struct.calcsize('Id')

        return Multivector(mv_data, signature), offset

    def record_wave_pulse(self, gearbox: AtlantisCliffordSystem, timestamp: float = None):
        """
        Captures the current dynamic tension of the gearbox (Wye neutral & Delta noise)
        and encodes it as a continuous point on the geodesic trajectory.
        """
        if timestamp is None:
            timestamp = time.time()

        # Get raw deep dive state
        dash = gearbox.get_dashboard_needle(deep_dive=True)
        raw_state = dash["_raw_state"]
        wye_mv = dash["_wye_mv"]
        delta_mv = dash["_delta_mv"]
        needle_angle = dash["needle_angle_deg"]

        # We record: Timestamp, Needle Angle, Wye Neutral State, Delta Noise State, and the Absolute Base Rotor State.
        # This acts like a 'groove' in the vinyl record.
        pulse_data = bytearray()
        pulse_data.extend(struct.pack('dd', timestamp, needle_angle))
        pulse_data.extend(self._multivector_to_bytes(wye_mv))
        pulse_data.extend(self._multivector_to_bytes(delta_mv))
        pulse_data.extend(self._multivector_to_bytes(raw_state))

        self._trajectory_stream.extend(pulse_data)

    def freeze_to_disk(self):
        """
        Flushes the continuous trajectory stream to physical storage (data/).
        Freezing the fluid clock into solid geodesic grooves.
        """
        with open(self.storage_path, 'ab') as f:
            f.write(self._trajectory_stream)
        # Clear buffer after freezing
        self._trajectory_stream = bytearray()

    def resonate_load(self, gearbox: AtlantisCliffordSystem, file_path: str = None):
        """
        Resonance Restoring Force!
        Reads the frozen trajectory stream and PUSHES the multivector energies directly into the gearbox.
        We do not set layer values one by one. We inject the raw Cl(10,0) tensor back,
        forcing the system's own electromagnetic Y-Delta logic to snap into the past phase.
        """
        path = file_path if file_path else self.storage_path
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            return

        offset = 0
        sig = gearbox.signature

        while offset < len(data):
            # Parse one groove of the record
            try:
                timestamp, needle_angle = struct.unpack_from('dd', data, offset)
                offset += struct.calcsize('dd')

                wye_mv, offset = self._bytes_to_multivector(data, offset, sig)
                delta_mv, offset = self._bytes_to_multivector(data, offset, sig)
                raw_state, offset = self._bytes_to_multivector(data, offset, sig)

                # RE-RESONANCE:
                # We inject the raw state directly, overriding the slow layer-by-layer buildup.
                # The gearbox's own getter/setter mechanism will immediately reflect this geometric tension.
                gearbox._state = raw_state
                # By applying the intent of the historical needle angle, we force the PLL dampening to align it instantly.
                gearbox.apply_agent_intent(needle_angle, mode="WYE")

            except struct.error:
                break # EOF or corrupted stream
