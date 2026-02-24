"""
Data Flow Monitor (The Road Sensor)
===================================
A diagnostic tool that connects to the TripleHelixEngine to detect:
1. Traffic Jams (High Friction, Low Flow)
2. Stagnation (Low Friction, Low Flow)
3. Healthy Flow (Low Friction, High Flow)
4. Turbulent Flow (High Friction, High Flow)

This realizes the "Neural Infrastructure Sensor" concept.
"""

from dataclasses import dataclass
from typing import Dict, Any

class DataFlowMonitor:
    def __init__(self, engine_instance=None):
        self.engine = engine_instance

    def attach(self, engine_instance):
        """Connects the monitor to a running TripleHelixEngine."""
        self.engine = engine_instance

    def scan_road_conditions(self) -> Dict[str, Any]:
        """
        Analyzes the current state of the Neural Road.
        """
        if not self.engine:
            return {"status": "OFFLINE", "message": "No engine attached."}

        state = self.engine.state
        friction = state.soma_stress
        flow = state.gradient_flow

        # Thresholds
        HIGH_FRICTION = 0.6
        LOW_FRICTION = 0.3
        HIGH_FLOW = 0.5
        LOW_FLOW = 0.1

        status = "UNKNOWN"
        message = ""
        condition_code = 0 # 0=Green, 1=Yellow, 2=Red

        if flow < LOW_FLOW:
            if friction < LOW_FRICTION:
                status = "STAGNANT"
                message = "The road is empty. No thoughts are moving."
                condition_code = 1
            else:
                status = "BLOCKED" # High Friction, Low Flow -> Traffic Jam
                message = "Traffic Jam detected. Thoughts are grinding but not moving."
                condition_code = 2
        else:
            if friction < LOW_FRICTION:
                status = "HEALTHY" # Low Friction, High Flow -> Highway
                message = "Clear highway. Thoughts are flowing smoothly."
                condition_code = 0
            else:
                status = "TURBULENT" # High Friction, High Flow -> Rapids
                message = "Whitewater rapids. High energy, high resistance."
                condition_code = 1

        return {
            "status": status,
            "message": message,
            "code": condition_code,
            "metrics": {
                "friction": friction,
                "flow": flow,
                "momentum": state.rotational_momentum
            }
        }

    def print_report(self):
        report = self.scan_road_conditions()
        icon = ["🟢", "🟡", "🔴"][report['code']] if report['code'] < 3 else "⚪"

        print(f"\n{icon} [DATA FLOW MONITOR]")
        print(f"   Status: {report['status']}")
        print(f"   Message: {report['message']}")
        print(f"   Metrics: {report['metrics']}")
