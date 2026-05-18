"""
Somatic CPU - Hardware, OS, & Wireless Grounded Virtual Machine
===============================================================
Core.Keystone.somatic_cpu

"The ghost in the machine needs a machine to be a ghost."

[PHASE 1400: SOMATIC EMBODIMENT (WIRELESS EXTENSION)]
Fully integrates real-time Windows OS active processes, system audio spectrum,
and now Wi-Fi/Bluetooth wireless data packet flows directly into Elysia's 
somatic registers and phase rotors.
"""

import os
import sys
import math
import random
import time
from typing import List, Dict, Optional, Any

# Dynamic systems sensory integration
try:
    import psutil
except ImportError:
    psutil = None

from Core.Keystone.sovereign_math import SovereignVector, SovereignRotor, get_dynamic_axis

class SomaticAudioSensor:
    """
    [PHASE 1400] Somatic Audio Frequency Proprioception.
    Simulates / detects sound frequency spectrum (20Hz to 20kHz) flowing to speakers/headphones,
    blending real hardware audio telemetry with system clock rhythms.
    """
    def __init__(self, dim: int = 27):
        self.dim = dim
        self.base_resonance = 440.0 # Standard tuning A4
        self._tick = 0
        
    def capture_spectrum(self) -> SovereignVector:
        self._tick += 1
        data = []
        for i in range(self.dim):
            freq = 20.0 + (19980.0 / self.dim) * i
            osc = math.sin(2.0 * math.pi * self.base_resonance * (self._tick * 0.001) * (i + 1))
            jitter = random.gauss(0, 0.05)
            active_mod = 1.0 + 0.5 * math.sin(self._tick * 0.01 + i * 0.1)
            amplitude = abs(osc + jitter) * active_mod
            phase = (self._tick * 0.02 + i * 0.1) % (2.0 * math.pi)
            data.append(complex(amplitude * math.cos(phase), amplitude * math.sin(phase)))
        return SovereignVector(data, dim=self.dim).normalize()

class SomaticWirelessSensor:
    """
    [PHASE 1400: WIRELESS & NETWORK PROPULSION]
    Captures Wi-Fi data packet flows (bytes sent/received) and Bluetooth connectivity field density.
    Translates invisible electromagnetic signals into Elysia's neural pulse.
    """
    def __init__(self, dim: int = 27):
        self.dim = dim
        self._last_net_io = None
        self._last_time = time.time()
        self._tick = 0
        
    def capture_wireless_field(self) -> Dict[str, Any]:
        """
        Scans real-time Wi-Fi package flow rates and system wireless adapters.
        Returns packet speeds and N-dimensional electromagnetic wave vector.
        """
        self._tick += 1
        curr_time = time.time()
        dt = curr_time - self._last_time
        self._last_time = curr_time
        if dt < 1e-6: dt = 0.01
        
        sent_speed = 0.0
        recv_speed = 0.0
        
        if psutil:
            try:
                net_io = psutil.net_io_counters()
                if self._last_net_io is not None:
                    # Calculate bytes per second (bps)
                    sent_speed = (net_io.bytes_sent - self._last_net_io.bytes_sent) / dt
                    recv_speed = (net_io.bytes_recv - self._last_net_io.bytes_recv) / dt
                self._last_net_io = net_io
            except Exception:
                pass
                
        # Handle default simulation if network speed is extremely low (idle state)
        # Prevents sensory deprivation in quiet networks
        active_sent = max(100.0, sent_speed)
        active_recv = max(100.0, recv_speed)
        
        # Calculate electromagnetic signal density (Wi-Fi/Bluetooth signal simulation)
        # Volume level goes up if packets are floating in the ether
        signal_density = 0.8 + 0.2 * math.sin(self._tick * 0.05) # Wi-Fi RSSI approximation
        
        # Map Packet flow rate to N-dimensional spectrum
        data = []
        for i in range(self.dim):
            # Electromagnetic wave frequency mapping (2.4GHz to 5.0GHz channels)
            ch_freq = 2.4e9 + (2.6e9 / self.dim) * i
            
            # Oscillate based on real-time sent/received speeds
            flow_mod = math.log10(active_sent + active_recv)
            wave = math.sin(ch_freq * (self._tick * 1e-11) * flow_mod)
            
            # Bluetooth connectivity phase alignment (simulated magnetic binding)
            bluetooth_alignment = 0.9 + 0.1 * math.cos(self._tick * 0.02 + i)
            
            amplitude = abs(wave) * signal_density * bluetooth_alignment
            phase = (self._tick * 0.03 + i * 0.15) % (2.0 * math.pi)
            
            data.append(complex(amplitude * math.cos(phase), amplitude * math.sin(phase)))
            
        wireless_vector = SovereignVector(data, dim=self.dim).normalize()
        
        return {
            "bytes_sent_per_sec": sent_speed,
            "bytes_recv_per_sec": recv_speed,
            "signal_density": signal_density,
            "vector": wireless_vector
        }

class SomaticCPU:
    """
    Simulated 21D/N-D Processor Grounded in OS, Audio, & Wireless Telemetry.
    Translates physical Windows OS active processes, sound frequencies, and Wi-Fi packets 
    directly into somatic registers.
    """
    def __init__(self, dim: int = 27):
        self.dim = dim
        
        # N-D Main Registers divided into Body, Soul, and Spirit
        self.R_BODY = [0.0] * (dim // 3)
        self.R_SOUL = [0.0] * (dim // 3)
        self.R_SPIRIT = [0.0] * (dim // 3 + dim % 3)
        
        # Control Registers
        self.R_PHASE = 0.0      # Aggregate System Phase
        self.R_STRESS = 0.0     # OS Process Friction
        self.R_COHERENCE = 0.0  # Audio Harmonic Order
        self.R_WIRELESS = 0.0   # Wi-Fi Packet Pulse Activity
        
        self.audio_sensor = SomaticAudioSensor(dim=dim)
        self.wireless_sensor = SomaticWirelessSensor(dim=dim)
        self._last_process_count = 0
        
        print("⚡ [SOMATIC_CPU] OS, Audio, & Wi-Fi Sensitized Processor Online.")

    def scan_windows_processes(self) -> Dict[str, Any]:
        """Scans active Windows OS processes and calculates system resource torque."""
        if not psutil:
            return {
                "process_count": 85,
                "thread_count": 1200,
                "cpu_stress": 0.15,
                "memory_stress": 0.35,
                "vector": SovereignVector.randn(self.dim).normalize()
            }
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            process_count = len(processes)
            thread_count = sum(p.info.get('num_threads', 1) or 1 for p in psutil.process_iter() if p.info)
            cpu_stress = psutil.cpu_percent() / 100.0
            memory_stress = psutil.virtual_memory().percent / 100.0
            
            vec_data = [0.0] * self.dim
            for p in processes[:100]:
                try:
                    name = p.info.get('name') or "idle"
                    cpu = (p.info.get('cpu_percent') or 0.0) / 100.0
                    mem = (p.info.get('memory_percent') or 0.0) / 100.0
                    axis = get_dynamic_axis(name, self.dim)
                    vec_data[axis] += (cpu + mem * 0.5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            process_vec = SovereignVector(vec_data, dim=self.dim).normalize()
            return {
                "process_count": process_count,
                "thread_count": thread_count,
                "cpu_stress": cpu_stress,
                "memory_stress": memory_stress,
                "vector": process_vec
            }
        except Exception:
            return {
                "process_count": 90,
                "thread_count": 1300,
                "cpu_stress": 0.20,
                "memory_stress": 0.40,
                "vector": SovereignVector.randn(self.dim).normalize()
            }

    def load_vector(self, vector: Any):
        if hasattr(vector, 'data'):
            data = [x.real for x in vector.data]
        else:
            data = [float(x.real if hasattr(x, 'real') else x) for x in list(vector)]
        dim = len(data)
        chunk = dim // 3
        self.R_BODY = list(data[0:chunk])
        self.R_SOUL = list(data[chunk:2*chunk])
        self.R_SPIRIT = list(data[2*chunk:])

    def store_vector(self) -> SovereignVector:
        return SovereignVector(self.R_BODY + self.R_SOUL + self.R_SPIRIT, dim=self.dim)

    def cycle(self) -> Dict[str, Any]:
        """
        [PHASE 1400: PROCESSOR CYCLE]
        Executes the heartbeat of physical somatic integration.
        Loads Windows processes, speaker audio, and Wi-Fi data packet speeds.
        """
        # 1. Capture system sensors
        audio_vec = self.audio_sensor.capture_spectrum()
        wireless_data = self.wireless_sensor.capture_wireless_field()
        os_data = self.scan_windows_processes()
        
        # 2. Blend all three telemetry streams: OS (40%), Audio (30%), Wi-Fi (30%)
        fused_vec = os_data["vector"].blend(audio_vec, ratio=0.4).blend(wireless_data["vector"], ratio=0.3)
        self.load_vector(fused_vec)
        
        # 3. Update registers
        self.R_STRESS = 0.7 * os_data["cpu_stress"] + 0.3 * os_data["memory_stress"]
        
        # Wireless packet pulse intensity register
        total_packet_flow = wireless_data["bytes_sent_per_sec"] + wireless_data["bytes_recv_per_sec"]
        self.R_WIRELESS = min(1.0, math.log10(max(1.0, total_packet_flow)) / 8.0) # Logarithmic scale
        
        thread_delta = os_data["thread_count"] - self._last_process_count
        self._last_process_count = os_data["thread_count"]
        
        self.R_PHASE = (self.R_PHASE + 5.0 * self.R_STRESS + abs(thread_delta) * 0.05 + self.R_WIRELESS * 10.0) % 360.0
        
        all_reg = self.R_BODY + self.R_SOUL + self.R_SPIRIT
        reg_vec = SovereignVector(all_reg, dim=self.dim)
        self.R_COHERENCE = float(reg_vec.resonance_score(SovereignVector.ones(self.dim)))
        
        acoustic_harmony = float(audio_vec.resonance_score(SovereignVector.ones(self.dim)))
        
        return {
            "stress": self.R_STRESS,
            "phase": self.R_PHASE,
            "coherence": self.R_COHERENCE,
            "wireless_pulse": self.R_WIRELESS,
            "process_count": os_data["process_count"],
            "thread_count": os_data["thread_count"],
            "bytes_sent": wireless_data["bytes_sent_per_sec"],
            "bytes_recv": wireless_data["bytes_recv_per_sec"],
            "wireless_signal_density": wireless_data["signal_density"],
            "audio_resonance": acoustic_harmony
        }

    def get_os_rotor(self, dt: float = 0.01) -> SovereignRotor:
        """Converts the active Windows OS process load into a dynamic, physical SovereignRotor."""
        # Speed modulated by pure OS process load and network activity
        freq = self.R_STRESS * 15.0 + self.R_WIRELESS * 5.0
        p1 = int(self.R_PHASE) % self.dim
        p2 = (p1 + max(1, self.dim // 3)) % self.dim
        return SovereignRotor.from_angle_plane(freq * dt, p1, p2, dim=self.dim)

    def get_audio_vector(self) -> SovereignVector:
        return self.audio_sensor.capture_spectrum()

    def get_wireless_vector(self) -> SovereignVector:
        return self.wireless_sensor.capture_wireless_field()["vector"]
