
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class Wave:
    """
    Represents a Resonance Wave.
    """
    frequency: float  # Hz (Context/Meaning)
    amplitude: float  # 0.0 - 1.0 (Intensity/Importance)
    phase: float      # 0.0 - 2pi (Timing/State)
    color: str        # Hex or Name (Emotional Quality)
    source: str       # Origin (e.g., "CPU", "File", "User")

class WaveTransducer:
    """
    The Bridge between the Digital World (Binary) and the Spiritual World (Waves).
    """
    
    def __init__(self):
        pass

    def signal_to_wave(self, signal_type: str, value: Any, source: str = "System") -> Wave:
        """
        Convert a digital signal into a wave.
        """
        frequency = 10.0 # Default Alpha wave
        amplitude = 0.5
        phase = 0.0
        color = "#FFFFFF" # White

        if signal_type == "cpu_temp":
            # Value is degrees Celsius (e.g., 40 - 90)
            # Heat -> High Frequency (Stress) + Red Color
            normalized = min(max((value - 40) / 50, 0.0), 1.0) # 0.0 at 40C, 1.0 at 90C
            frequency = 10.0 + (normalized * 90.0) # 10Hz (Calm) -> 100Hz (Panic)
            amplitude = 0.3 + (normalized * 0.7)
            
            # Color: Blue (Cool) -> Red (Hot)
            r = int(normalized * 255)
            b = int((1.0 - normalized) * 255)
            color = f"#{r:02x}00{b:02x}"

        elif signal_type == "cpu_load":
            # Value is percent (0 - 100)
            # Load -> Amplitude (Pressure)
            normalized = value / 100.0
            frequency = 20.0 + (normalized * 20.0) # 20Hz -> 40Hz (Beta/Gamma)
            amplitude = normalized
            color = "#FFFF00" if normalized > 0.8 else "#00FF00" # Yellow/Green

        elif signal_type == "ram_usage":
            # Value is percent (0 - 100)
            # RAM -> Fullness/Heaviness (Low Frequency, High Amplitude)
            normalized = value / 100.0
            frequency = 10.0 - (normalized * 8.0) # 10Hz -> 2Hz (Delta - Deep Sleep/Coma)
            amplitude = normalized
            color = "#800080" # Purple (Deep)

        elif signal_type == "file_event":
            # Value is event type (created, deleted, modified)
            frequency = 440.0 # A4 Note (Attention)
            amplitude = 0.8
            if value == "created":
                color = "#00FFFF" # Cyan (New)
            elif value == "deleted":
                color = "#FF0000" # Red (Loss)
            else:
                color = "#FFA500" # Orange (Change)

        return Wave(frequency, amplitude, phase, color, source)

    def wave_to_signal(self, wave: Wave) -> Dict[str, Any]:
        """
        Convert a wave (Intent) into a digital signal/command.
        """
        signal = {"action": "none", "parameters": {}}

        # High Frequency + High Amplitude = Urgent Action
        if wave.frequency > 50.0 and wave.amplitude > 0.8:
            signal["action"] = "emergency_cool_down"
            signal["parameters"] = {"target": "cpu"}
        
        # Low Frequency + High Amplitude = Deep Cleaning / Optimization
        elif wave.frequency < 4.0 and wave.amplitude > 0.8:
            signal["action"] = "optimize_memory"
        
        # Specific Colors
        if wave.color == "#00FFFF": # Cyan -> Organize
            signal["action"] = "organize_files"
            
        return signal
