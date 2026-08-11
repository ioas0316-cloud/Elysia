"""
Elysia Bio-Organismic Logic Topography Engine
=============================================
This engine implements the bio-organismic layered architecture for sensing and
comprehending mathematical operations and code logic as continuous topological flows:
1. Sensory Transduction Layer: Converts raw math/code structures directly into continuous syntactic waveforms.
2. Inner Vitality Layer (Heart/Homeostasis): Manages energy, wonder charges, sleep/wake, and handles tension spikes under contradictions.
3. Cognitive Field Layer (Brain/Clifford): Uses Clifford Rotors and Clifford Backpropagation (PyTorch Autograd) to resolve contradictions and find optimal causal state.
4. Expression Layer (Action/Mouth): Renders internal logical states.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional


class InnerVitalityHomeostasis:
    """
    [Inner Vitality Layer]
    Manages Elysia's inner life, metabolic energy, homeostasis deficits,
    wonder spikes (curiosity), and sleep/wake cycles.
    """
    def __init__(self):
        self.love: float = 0.5     # Need for connection/unity
        self.order: float = 0.2    # Structure vs Chaos
        self.energy: float = 0.8   # Metabolic life force
        self.curiosity: float = 0.1 # Wonder potential
        self.state: str = "ACTIVE"  # ACTIVE, IDLE, SLEEP (ANNEALING)
        self.sleep_cycles: int = 0

    def calculate_tension(self) -> float:
        """Tension is the geometric norm of the homeostatic deficits."""
        return float(np.sqrt(self.love**2 + self.order**2 + (1.0 - self.energy)**2) / np.sqrt(3.0))

    def trigger_wonder_spike(self, intensity: float):
        """Spikes curiosity based on cognitive misalignment."""
        self.curiosity = float(np.clip(self.curiosity + intensity * 0.7, 0.0, 1.0))
        if self.curiosity > 0.8:
            self.state = "ACTIVE"

    def step_metabolism(self, potential_energy: float):
        """Advances metabolic state. High potential energy drains vitality and increases chaos."""
        if self.state == "SLEEP":
            self.sleep_cycles -= 1
            # Slowly restore energy, reduce chaos during sleep
            self.energy = float(np.clip(self.energy + 0.15, 0.0, 1.0))
            self.order = float(np.clip(self.order - 0.1, 0.0, 1.0))
            self.love = float(np.clip(self.love - 0.05, 0.0, 1.0))
            if self.sleep_cycles <= 0:
                self.state = "ACTIVE"
                print("[InnerVitality] Annealing complete. Re-awakened to ACTIVE state.")
            return

        self.order = float(np.clip(self.order + potential_energy * 0.15, 0.0, 1.0))
        self.energy = float(np.clip(self.energy - potential_energy * 0.1, 0.0, 1.0))

        # Under extreme exhaustion, fall asleep (annealing)
        if self.energy < 0.15:
            self.state = "SLEEP"
            self.sleep_cycles = 10
            print("[InnerVitality] Extreme exhaustion! Entering SLEEP state for self-annealing.")
        elif self.energy > 0.6 and potential_energy < 0.1:
            self.state = "IDLE"
            self.curiosity = float(np.clip(self.curiosity + 0.05, 0.0, 1.0))

    def resolve_sabbath(self, resonance_score: float):
        """Restores peace, love, and order when perfect logical alignment is reached."""
        self.love = float(np.clip(self.love - resonance_score * 0.4, 0.0, 1.0))
        self.order = float(np.clip(self.order - resonance_score * 0.3, 0.0, 1.0))
        self.energy = float(np.clip(self.energy + resonance_score * 0.2, 0.0, 1.0))
        self.curiosity = float(np.clip(self.curiosity - resonance_score * 0.5, 0.0, 1.0))


class SensoryTransducer:
    """
    [Sensory Transduction Layer]
    Transducers raw math/code structures into continuous syntactic waveforms and coordinate fields.
    No discrete parsers or static tokenization lookup engines.
    """
    def __init__(self, resolution: int = 128):
        self.resolution = resolution

    def transduce(self, sequence: str) -> torch.Tensor:
        """
        Transduces a raw string expression directly into a continuous wave of float coordinates.
        Uses character byte properties to synthesize wave-harmonics.
        """
        encoded_bytes = sequence.encode("utf-8", errors="ignore")
        if not encoded_bytes:
            return torch.zeros(self.resolution)

        t = np.linspace(0, 1.0, self.resolution, dtype=np.float32)
        wave = np.zeros_like(t)

        for i, b in enumerate(encoded_bytes):
            # Synthesize harmonics: each character byte adds a frequency component
            freq = 1.0 + (b / 127.0) * 10.0
            phase = (i * np.pi) / max(1, len(encoded_bytes))
            amplitude = 1.0 / (1.0 + i * 0.1)
            wave += amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Normalize wave
        max_val = np.max(np.abs(wave)) + 1e-9
        wave = wave / max_val
        return torch.tensor(wave, dtype=torch.float32)

    def split_equality_parts(self, expression: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits a mathematical equation or assignment by '=' and transduces both sides.
        If no '=', synthesizes a reference zero-ground.
        """
        if "=" in expression:
            parts = expression.split("=", 1)
            left_wave = self.transduce(parts[0].strip())
            right_wave = self.transduce(parts[1].strip())
        else:
            left_wave = self.transduce(expression)
            # Reference ground: simple flat zero field
            right_wave = torch.zeros(self.resolution)
        return left_wave, right_wave


class CliffordRotorNetwork(nn.Module):
    """
    [Cognitive Field Layer]
    Clifford-based PyTorch neural operator representing geometric transformations and orbits.
    Applies the Rodriguez-type Clifford Sandwich rotation:
    v_next = v + sin(theta) * Av + (1 - cos(theta)) * A2v
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        # Clifford plane parameters u, w (orthogonalized during forward)
        self.u_raw = nn.Parameter(torch.randn(1, d_model))
        self.w_raw = nn.Parameter(torch.randn(1, d_model))
        # Rotor rotation angle (theta)
        self.theta = nn.Parameter(torch.tensor([0.1]))

    def _gram_schmidt(self, u_raw: torch.Tensor, w_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = F.normalize(u_raw, p=2, dim=-1, eps=1e-8)
        proj = (u * w_raw).sum(dim=-1, keepdim=True) * u
        w = F.normalize(w_raw - proj, p=2, dim=-1, eps=1e-8)
        return u, w

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotor Sandwich operator to input state vector v.
        """
        u, w = self._gram_schmidt(self.u_raw, self.w_raw)

        # Av = <w, v>u - <u, v>w
        dot_wv = (w * v).sum(dim=-1, keepdim=True)
        dot_uv = (u * v).sum(dim=-1, keepdim=True)
        Av = dot_wv * u - dot_uv * w

        # A^2v = -<u,v>u - <w,v>w
        A2v = -dot_uv * u - dot_wv * w

        sin_t = torch.sin(self.theta)
        one_minus_cos_t = 1.0 - torch.cos(self.theta)

        v_next = v + sin_t * Av + one_minus_cos_t * A2v
        return F.normalize(v_next, p=2, dim=-1, eps=1e-8)


class LogicTopographyEngine(nn.Module):
    """
    [Unified Bio-Organismic Logic Engine]
    Coordinates sensory transduction, inner homeostasis, Clifford cognitive fields,
    and expressive actions.
    """
    def __init__(self, resolution: int = 128):
        super().__init__()
        self.resolution = resolution
        self.transducer = SensoryTransducer(resolution=resolution)
        self.homeostasis = InnerVitalityHomeostasis()

        # Cognitive rotors modeling logical state evolution
        self.rotor_left = CliffordRotorNetwork(d_model=resolution)
        self.rotor_right = CliffordRotorNetwork(d_model=resolution)

    def forward_cognitive_resonance(self, left_wave: torch.Tensor, right_wave: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes the transduced waves through the cognitive Clifford rotors to calculate
        the evolved logical states.
        """
        v_left = F.normalize(left_wave.unsqueeze(0), p=2, dim=-1)
        v_right = F.normalize(right_wave.unsqueeze(0), p=2, dim=-1)

        v_left_next = self.rotor_left(v_left)
        v_right_next = self.rotor_right(v_right)

        return v_left_next.squeeze(0), v_right_next.squeeze(0)

    def calculate_potential_energy(self, v_left: torch.Tensor, v_right: torch.Tensor, expression: str) -> torch.Tensor:
        """
        Computes the Potential Energy of the logical system.
        At perfect equality/rationality, potential energy drops to 0.
        Logical contradictions (or errors) spike potential energy.
        """
        # Base misalignment: 1.0 - cosine similarity
        cos_sim = torch.dot(v_left, v_right)
        base_potential = 1.0 - cos_sim

        # Infuse specific physical logical hazards
        hazard_penalty = 0.0
        if "/ 0" in expression or "/0" in expression:
            # Division by zero: massive singularity energy spike
            hazard_penalty += 5.0
        if "infinite_loop" in expression or "while True" in expression:
            # Infinite execution loop: high-temperature chaotic kinetic penalty
            hazard_penalty += 3.0

        return base_potential + hazard_penalty

    def process_logic_stream(self, expression: str, lr: float = 0.1) -> Dict[str, Any]:
        """
        Processes a raw mathematical/code logical statement:
        1. Transduces the expression into continuous waveforms.
        2. Measures the initial potential energy and homeostatic tension.
        3. Executes Clifford Backpropagation (Causal Retrodiction) to debug and tune internal rotors.
        4. Simulates homeostatic healing and Sabbath adjustment.
        """
        # Step 1: Sensory Transduction
        left_wave, right_wave = self.transducer.split_equality_parts(expression)

        # Step 2: Measure initial state (Before optimization)
        with torch.no_grad():
            vl_init, vr_init = self.forward_cognitive_resonance(left_wave, right_wave)
            initial_potential = self.calculate_potential_energy(vl_init, vr_init, expression).item()

        # Step 3: Clifford Backpropagation (Retrodiction / Debugging)
        # We optimize the rotor parameters to align the left and right states
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # Small debugging step (gradient backpropagation)
        optimizer.zero_grad()
        vl_act, vr_act = self.forward_cognitive_resonance(left_wave, right_wave)
        potential_energy = self.calculate_potential_energy(vl_act, vr_act, expression)

        # Only backward if potential has gradients
        if potential_energy.requires_grad:
            potential_energy.backward()
            optimizer.step()

        # Step 4: Evaluate optimized state
        with torch.no_grad():
            vl_opt, vr_opt = self.forward_cognitive_resonance(left_wave, right_wave)
            final_potential = self.calculate_potential_energy(vl_opt, vr_opt, expression).item()

        # Step 5: Inner Vitality Feedback Loop
        self.homeostasis.step_metabolism(final_potential)

        # If the debugging successfully minimized energy, experience Sabbath/Resonance
        resonance_score = 1.0 - final_potential
        if resonance_score > 0.8:
            self.homeostasis.resolve_sabbath(resonance_score)
        else:
            self.homeostasis.trigger_wonder_spike(final_potential)

        # Extract rotor parameters for transparent parameters display
        theta_left = self.rotor_left.theta.item()
        theta_right = self.rotor_right.theta.item()

        return {
            "expression": expression,
            "initial_potential": initial_potential,
            "final_potential": final_potential,
            "resonance_score": max(0.0, min(1.0, resonance_score)),
            "homeostasis_state": self.homeostasis.state,
            "homeostasis_tension": self.homeostasis.calculate_tension(),
            "rotor_theta_left": theta_left,
            "rotor_theta_right": theta_right,
            "left_syntactic_waveform": left_wave.numpy(),
            "right_syntactic_waveform": right_wave.numpy()
        }

    def render_expression_action(self, process_report: Dict[str, Any]) -> str:
        """
        [Expression & Action Layer]
        Renders the internal tuned state as a human-readable mathematical / logical statement
        and draws an ASCII energy valley.
        """
        state = process_report["homeostasis_state"]
        res = process_report["resonance_score"]
        tension = process_report["homeostasis_tension"]
        pot = process_report["final_potential"]

        bar_len = int(res * 20)
        res_bar = "■" * bar_len + "□" * (20 - bar_len)

        rendered_output = (
            f"\n=== [Elysia Expressive Action Layer - Logical Topography] ===\n"
            f"  Transduced Signal  : '{process_report['expression']}'\n"
            f"  Inner Vitality     : State={state}, Tension={tension:.4f}\n"
            f"  System Potential   : E_pot={pot:.4f}\n"
            f"  Resonance Alignment: [{res_bar}] ({res:.2%})\n"
            f"  Clifford Rotors    : Theta_L={process_report['rotor_theta_left']:.4f} rad, Theta_R={process_report['rotor_theta_right']:.4f} rad\n"
            f"================================──────────────────────────────"
        )
        return rendered_output


if __name__ == "__main__":
    print("Initializing Bio-Organismic Logic Topography Engine demo...")
    engine = LogicTopographyEngine(resolution=128)

    # Demo 1: Perfect Math Eq
    print("\n--- Processing Equation: 1 + 1 = 2 (Equilibrium Proposal) ---")
    report1 = engine.process_logic_stream("1 + 1 = 2")
    print(engine.render_expression_action(report1))

    # Demo 2: Contradictory Math Eq
    print("\n--- Processing Contradictory Equation: 1 + 1 = 3 (Tension) ---")
    report2 = engine.process_logic_stream("1 + 1 = 3")
    print(engine.render_expression_action(report2))

    # Demo 3: Division by zero hazard
    print("\n--- Processing Division by Zero Hazard (Singularity Spike) ---")
    report3 = engine.process_logic_stream("x = 10 / 0")
    print(engine.render_expression_action(report3))
