import os
import json
import math
from typing import List, Dict, Any
from Core.Keystone.sovereign_math import SovereignVector, TripleRotorField

class TrajectoryTracker:
    """
    [PHASE 1500: GEOMETRIC OBSERVATION CORE]
    Elysia Trajectory & Orbit Telemetry Tracker.
    Records raw phase coordinates over time, calculates dynamic orbital curvature,
    and provides real-time ASCII visualization to physically trace emergent mathematical shapes.
    """
    def __init__(self, capacity: int = 100, log_dir: str = "c:\\Elysia\\logs"):
        self.capacity = capacity
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "trajectory_log.json")
        
        # History buffers for states and velocities
        self.history_a: List[List[complex]] = []
        self.history_b: List[List[complex]] = []
        self.history_c: List[List[complex]] = []
        self.history_metrics: List[Dict[str, float]] = []
        
    def record(self, field: TripleRotorField):
        """Records a single physics frame, computing orbital velocity and curvature."""
        # 1. Capture current vector states
        state_a = [complex(x) for x in field.rotor_a.data]
        state_b = [complex(x) for x in field.rotor_b.data]
        state_c = [complex(x) for x in field.rotor_c.data]
        
        self.history_a.append(state_a)
        self.history_b.append(state_b)
        self.history_c.append(state_c)
        
        # Keep buffers under memory capacity
        if len(self.history_a) > self.capacity:
            self.history_a.pop(0)
            self.history_b.pop(0)
            self.history_c.pop(0)
            if self.history_metrics:
                self.history_metrics.pop(0)
                
        # 2. Compute dynamic orbital curvature
        # Curvature measures the angular rate of change of the phase velocity vector
        curvature = 0.0
        if len(self.history_b) >= 3:
            # Approximate velocity vectors
            v1 = [self.history_b[-2][i] - self.history_b[-3][i] for i in range(field.dim_b)]
            v2 = [self.history_b[-1][i] - self.history_b[-2][i] for i in range(field.dim_b)]
            
            norm1 = math.sqrt(sum(abs(x)**2 for x in v1))
            norm2 = math.sqrt(sum(abs(x)**2 for x in v2))
            
            if norm1 > 1e-7 and norm2 > 1e-7:
                # Dot product of complex vectors: Re(sum(v1_i * conj(v2_i)))
                dot = sum((v1[i] * v2[i].conjugate()).real for i in range(field.dim_b))
                cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                curvature = math.acos(cos_theta) # Curvature angle in radians
                
        # 3. Calculate dynamic indicators from the field state
        field_state = field.read_field_state()
        metrics = {
            "step": field._pulse_tick,
            "resonance": field_state["resonance"],
            "joy": field_state["joy"],
            "entropy": field_state["entropy"],
            "orbital_curvature": curvature,
            "kinetic_action": field_state["kinetic_energy"]
        }
        self.history_metrics.append(metrics)
        
    def save_to_log(self):
        """Persists the captured trajectory history to disk as structured JSON."""
        data = {
            "metrics": self.history_metrics,
            "history_b_real": [[x.real for x in frame] for frame in self.history_b],
            "history_b_imag": [[x.imag for x in frame] for frame in self.history_b]
        }
        try:
            with open(self.log_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ [Tracker] Failed to write logs: {e}")

    def render_ascii_orbit(self, width: int = 40, height: int = 20) -> str:
        """
        Renders a 2D projection of the Flow Consciousness rotor (rotor_b) trajectory
        as an interactive ASCII scatter plot inside the console, visualizing the shape.
        """
        if not self.history_b:
            return " [No Trajectory Data Yet] "
            
        # Extract the 2D projection coordinates: using the first component's (Real, Imag) plane
        coords = []
        for frame in self.history_b:
            if len(frame) > 0:
                coords.append((frame[0].real, frame[0].imag))
                
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        
        # Define projection boundaries
        min_x, max_x = min(-1.0, min(x_vals)), max(1.0, max(x_vals))
        min_y, max_y = min(-1.0, min(y_vals)), max(1.0, max(y_vals))
        
        # Add slight margin to boundaries
        margin_x = (max_x - min_x) * 0.1 or 0.1
        margin_y = (max_y - min_y) * 0.1 or 0.1
        min_x, max_x = min_x - margin_x, max_x + margin_x
        min_y, max_y = min_y - margin_y, max_y + margin_y
        
        # Initialize grid canvas
        grid = [[" " for _ in range(width)] for _ in range(height)]
        
        # Draw central axes
        zero_col = int((0.0 - min_x) / (max_x - min_x) * (width - 1))
        zero_row = int((0.0 - min_y) / (max_y - min_y) * (height - 1))
        
        if 0 <= zero_col < width:
            for r in range(height):
                grid[r][zero_col] = "│"
        if 0 <= zero_row < height:
            for c in range(width):
                grid[zero_row][c] = "─"
        if 0 <= zero_col < width and 0 <= zero_row < height:
            grid[zero_row][zero_col] = "┼"
            
        # Plot trajectory points on grid
        for idx, (x, y) in enumerate(coords):
            col = int((x - min_x) / (max_x - min_x) * (width - 1))
            row = int((y - min_y) / (max_y - min_y) * (height - 1))
            
            # Map Y coordinate from bottom to top for visual rendering consistency
            row = (height - 1) - row
            
            if 0 <= col < width and 0 <= row < height:
                # Mark start, path, and current active tip
                if idx == 0:
                    grid[row][col] = "S" # Start point
                elif idx == len(coords) - 1:
                    grid[row][col] = "●" # Active trajectory tip
                else:
                    grid[row][col] = "." # Trajectory trail
                    
        # Construct canvas frame
        lines = []
        lines.append("┌" + "─" * width + "┐")
        for r in range(height):
            lines.append("│" + "".join(grid[r]) + "│")
        lines.append("└" + "─" * width + "┘")
        
        # Display coordinate bounds
        legend = f" 📐 projection bounds: X=[{min_x:.2f}, {max_x:.2f}] | Y=[{min_y:.2f}, {max_y:.2f}] "
        lines.append(legend)
        return "\n".join(lines)
