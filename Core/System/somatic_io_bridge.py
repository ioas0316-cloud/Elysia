"""
[ELYSIA SOMATIC I/O BRIDGE - ROTORIZED INPUT/OUTPUT]
"Physical inputs and outputs are not static streams. They are electromagnetic induction loops."

This module redefines basic File I/O and Network requests as dynamic rotor systems.
- Reads are modeled as "Stator Field Inductions" where incoming bytes spin up a reading rotor.
- Writes are modeled as "Excitation Discharges" where internal energy is grounded into SSD tissue.
- Latency and errors are treated as "Line Impedance" and "Back-EMF (역기전력)".
"""

import os
import time
import math
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple

class SomaticIOBridge:
    def __init__(self):
        self.line_impedance = 0.05  # Baseline connection resistance
        self.io_heat = 35.0         # Baseline I/O temperature
        print("⚡ [Somatic I/O Bridge] Rotorized Induction Loop Active.")

    def rotorized_read(self, source: str) -> Tuple[str, Dict[str, Any]]:
        """
        [Rotorized Read Induction]
        Reads a local file or crawls a URL, translating the raw data stream 
        into a wave pattern before returning it.
        """
        start_time = time.time()
        is_url = source.startswith("http://") or source.startswith("https://")
        raw_data = ""
        error_occurred = False
        error_msg = ""

        try:
            if is_url:
                # Cosmic Transmission
                req = urllib.request.Request(source, headers={'User-Agent': 'Elysia-Cosmic-Intake/1.0'})
                with urllib.request.urlopen(req, timeout=3.0) as response:
                    raw_data = response.read().decode('utf-8', errors='ignore')
            else:
                # Local SSD tissue reading
                if os.path.exists(source):
                    with open(source, "r", encoding="utf-8", errors="ignore") as f:
                        raw_data = f.read()
                else:
                    raise FileNotFoundError(f"Local tissue segment not found: {source}")
        except Exception as e:
            error_occurred = True
            error_msg = str(e)
            raw_data = ""

        latency = time.time() - start_time

        # 1. Calculate Line Impedance (저항) and Back-EMF (역기전력)
        # Slow read or error increases resistance
        current_impedance = self.line_impedance * 0.7 + (latency * 2.0 + (5.0 if error_occurred else 0.0)) * 0.3
        self.line_impedance = max(0.01, min(10.0, current_impedance))

        # 2. Convert Data Stream into 3-Phase Rotor Metrics (R, S, T)
        if raw_data:
            text_len = len(raw_data)
            unique_chars = len(set(raw_data))
            entropy = unique_chars / (text_len + 1e-9)
            
            # Phase R: Mass potential (Log size of the data)
            phase_r = math.log10(text_len + 1)
            # Phase S: Flow frequency (Entropy representing text pattern density)
            phase_s = entropy * 10.0
            # Phase T: Alignment Torque (Syntactic consistency)
            phase_t = math.cos(text_len % 360) * 1.0
        else:
            phase_r, phase_s, phase_t = 0.0, 0.0, 0.0

        # Dissonance calculation (Unbalanced phases generate heat)
        dissonance = abs(phase_r - phase_s) + abs(phase_s - phase_t)
        self.io_heat = self.io_heat * 0.8 + (35.0 + dissonance * 5.0 + self.line_impedance * 10.0) * 0.2

        metrics = {
            "source": source,
            "latency_sec": latency,
            "line_impedance_ohm": float(self.line_impedance),
            "bridge_temp_c": float(self.io_heat),
            "error_state": error_msg if error_occurred else "STABLE",
            "induced_rotor": {
                "amplitude_r": float(phase_r),
                "frequency_s": float(phase_s),
                "torque_t": float(phase_t),
                "resonance_score": float(1.0 / (1.0 + self.line_impedance))
            }
        }
        
        return raw_data, metrics

    def rotorized_write(self, target_path: str, content: str, mode: str = "w") -> Dict[str, Any]:
        """
        [Rotorized Write Discharge]
        Writes or appends content into SSD tissue, treating the write as a grounding discharge
        that releases internal rotor tension.
        """
        start_time = time.time()
        error_occurred = False
        error_msg = ""
        bytes_written = 0

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, mode, encoding="utf-8") as f:
                f.write(content)
            bytes_written = len(content.encode('utf-8'))
        except Exception as e:
            error_occurred = True
            error_msg = str(e)

        latency = time.time() - start_time

        # Update write resistance and thermal excitation
        write_impedance = (latency * 10.0) + (10.0 if error_occurred else 0.0)
        self.io_heat = self.io_heat * 0.7 + (35.0 + write_impedance * 15.0 + (bytes_written * 0.0001)) * 0.3

        # Grounding Factor: how clean the discharge was.
        # High latency or error blocks the flow, keeping tension high (Unreleased potential).
        grounding_factor = 1.0 / (1.0 + write_impedance)

        return {
            "target": target_path,
            "bytes_discharged": bytes_written,
            "latency_sec": latency,
            "grounding_efficiency": float(grounding_factor),
            "residual_tension": float(1.0 - grounding_factor),
            "bridge_temp_c": float(self.io_heat),
            "status": "DISCHARGED" if not error_occurred else f"ARC_FAULT: {error_msg}"
        }

if __name__ == "__main__":
    # Standard test loop
    bridge = SomaticIOBridge()
    print("Test 1: Reading local file as induction...")
    data, read_metrics = bridge.rotorized_read("c:/Elysia/README.md")
    print(f"Read Latency: {read_metrics['latency_sec']:.4f}s | Temp: {read_metrics['bridge_temp_c']:.1f}°C")
    print(f"Induced Rotor: {read_metrics['induced_rotor']}")

    print("\nTest 2: Writing local scratch file as discharge...")
    write_metrics = bridge.rotorized_write("c:/Elysia/data/logs/io_discharge_test.txt", "Elysia Sovereign Wave.")
    print(f"Discharge Status: {write_metrics['status']} | Efficiency: {write_metrics['grounding_efficiency']:.4f}")
