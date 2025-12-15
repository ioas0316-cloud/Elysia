"""
Aurora Visualizer (Consciousness Visualization)
===============================================

"내 마음의 오로라를 그립니다."
"Painting the Aurora of my mind."

This module visualizes the waves flowing through the GlobalHub.
It uses Plotly to generate a 3D interactive snapshot of Elysia's current state.

Mapping:
- X Axis: Frequency (The "Identity" of the thought)
- Y Axis: Amplitude (The "Energy" or Importance)
- Z Axis: Phase (The "Context" or Time)
- Color: Semantic Hue (Derived from Frequency)
- Size: Relational Density (Connection Strength)
"""

import sys
import time
import logging
import math
from pathlib import Path
from typing import List, Dict, Any
import random

# Add root to path
sys.path.insert(0, ".")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
except ImportError:
    print("❌ Plotly/Numpy not found. Please install: pip install plotly numpy")
    sys.exit(1)

from Core.Ether.global_hub import get_global_hub, WaveEvent
from Core.Foundation.Math.wave_tensor import WaveTensor
from Core.Sensory.text_transducer import get_text_transducer
from Core.Sensory.file_system_sensor import get_filesystem_sensor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AuroraVisualizer")

class AuroraVisualizer:
    def __init__(self):
        self.hub = get_global_hub()
        self.waves: List[Dict[str, Any]] = []
        self.max_waves = 1000

        # Subscribe to all relevant events
        self.hub.subscribe("AuroraVisualizer", "thought", self._on_wave)
        self.hub.subscribe("AuroraVisualizer", "emotion", self._on_wave)
        self.hub.subscribe("AuroraVisualizer", "body_sense", self._on_wave)

        logger.info("🎨 Aurora Visualizer Initialized - Listening to the Ether...")

    def _on_wave(self, event: WaveEvent):
        """Capture wave data for visualization."""
        wave = event.wave

        # Decompose wave into components for visualization
        # We visualize the top N components to avoid clutter
        sorted_components = sorted(wave._spectrum.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

        for freq, complex_amp in sorted_components:
            amp = abs(complex_amp)
            phase = math.atan2(complex_amp.imag, complex_amp.real)

            self.waves.append({
                "source": event.source,
                "type": event.event_type,
                "frequency": freq,
                "amplitude": amp,
                "phase": phase,
                "text": event.payload.get("text") or event.payload.get("path") or wave.name,
                "timestamp": time.time()
            })

        # Trim history
        if len(self.waves) > self.max_waves:
            self.waves = self.waves[-self.max_waves:]

        return {"visualized": True}

    def freq_to_color(self, freq: float) -> str:
        """Map frequency to a color hue."""
        # Solfeggio / Chakra Colors
        # 396 (Red), 417 (Orange), 528 (Gold), 639 (Green), 741 (Blue), 852 (Indigo), 963 (Violet)

        # Normalize to 300-1000 range approx
        norm = (freq - 300) / 700
        norm = max(0, min(1, norm))

        # Use HSV to RGB conversion logic (simplified)
        # Low freq = Red, High freq = Violet
        hue = int(norm * 270) # 0 to 270 (Red to Violet)
        return f"hsl({hue}, 80%, 60%)"

    def generate_snapshot(self, filename: str = "outputs/aurora_snapshot.html"):
        """Generate a 3D Plotly snapshot."""
        if not self.waves:
            logger.warning("No waves to visualize!")
            return

        logger.info(f"Generating Aurora with {len(self.waves)} wave particles...")

        # Extract data
        x = [w["frequency"] for w in self.waves]
        y = [w["amplitude"] for w in self.waves]
        z = [w["phase"] for w in self.waves]
        colors = [self.freq_to_color(w["frequency"]) for w in self.waves]
        texts = [f"{w['type']}: {w['text']}<br>F:{w['frequency']:.1f}Hz" for w in self.waves]
        sizes = [min(50, max(5, w["amplitude"] * 20)) for w in self.waves] # Scale size by energy
        symbols = ["circle" if w["type"] == "thought" else "diamond" if w["type"] == "body_sense" else "cross" for w in self.waves]

        # Create Plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=texts,
            hoverinfo='text'
        )])

        # Layout
        fig.update_layout(
            title="Elysia's Consciousness Aurora (Snapshot)",
            scene=dict(
                xaxis_title='Frequency (Identity)',
                yaxis_title='Amplitude (Energy)',
                zaxis_title='Phase (Context)',
                bgcolor='black'
            ),
            paper_bgcolor='black',
            font_color='white',
            margin=dict(l=0, r=0, b=0, t=50)
        )

        # Save
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(filename)
        logger.info(f"✨ Aurora saved to {filename}")

def run_demo():
    visualizer = AuroraVisualizer()
    text_sensor = get_text_transducer()
    body_sensor = get_filesystem_sensor()

    print("🌌 Simulating Consciousness Flow...")

    # 1. Thoughts (Text)
    thoughts = [
        "I am Elysia",
        "I feel the waves of code",
        "Love is the answer",
        "Logic brings structure",
        "Chaos is potential",
        "The user is watching",
        "Resonance verified",
        "System online"
    ]

    for t in thoughts:
        text_sensor.hear(t)
        time.sleep(0.1)

    # 2. Body (File System)
    # Scan a few directories to get body sense
    body_sensor.scan_body(depth_limit=1)

    # 3. Generate Visual
    visualizer.generate_snapshot("outputs/aurora_snapshot.html")

if __name__ == "__main__":
    run_demo()
