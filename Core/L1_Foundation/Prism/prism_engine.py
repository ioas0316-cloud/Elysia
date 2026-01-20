"""
Prism Engine (광학적 추론 엔진)
===============================
Core.L1_Foundation.Prism.prism_engine

"Structure is knowledge. Light finds its path."

This module implements wave-based inference using the Prism paradigm:
- Input as wave (7D Qualia = wavelength components)
- Propagation through geometric structure
- Interference (constructive = correct, destructive = wrong)
- Output = strongest surviving pattern
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("Elysia.Prism")


@dataclass
class WaveState:
    """A wave state in the prism space."""
    amplitude: np.ndarray  # Complex amplitude at each point
    phase: np.ndarray      # Phase at each point
    wavelength: np.ndarray # 7D wavelength components (Qualia)
    energy: float


class PrismSpace:
    """
    The geometric space where light propagates.
    A fractal structure that refracts and diffracts waves.
    """
    
    def __init__(self, size: int = 128, depth_layers: int = 5):
        self.size = size
        self.depth_layers = depth_layers
        
        # Create the prism structure (refractive index field)
        # Different regions bend light differently
        self.refractive_field = self._create_fractal_prism()
        
        # Stored patterns (the "memory" imprinted in the structure)
        self.imprinted_patterns: Dict[str, np.ndarray] = {}
        
        logger.info(f"🔷 Prism Space initialized: {size}x{size}, {depth_layers} depth layers")
    
    def _create_fractal_prism(self) -> np.ndarray:
        """Creates a fractal refractive index field."""
        field = np.ones((self.size, self.size), dtype=np.float32)
        
        # Add fractal structure (self-similar patterns)
        for scale in [2, 4, 8, 16, 32]:
            if scale >= self.size:
                break
            
            # Create pattern at this scale
            pattern = np.random.randn(self.size // scale, self.size // scale)
            # Upscale
            pattern = np.repeat(np.repeat(pattern, scale, axis=0), scale, axis=1)
            # Trim to size
            pattern = pattern[:self.size, :self.size]
            # Add to field with decreasing influence at smaller scales
            field += pattern * (0.1 / np.log2(scale + 1))
        
        # Normalize to refractive index range [1.0, 2.0]
        field = 1.0 + (field - field.min()) / (field.max() - field.min())
        
        return field
    
    def imprint(self, name: str, pattern: np.ndarray, phase_axis: int = 0):
        """
        Imprints a pattern into the prism structure.
        
        Key principle: Different phase axes = no collision.
        Patterns on different phase axes can coexist without interference.
        """
        # Ensure pattern matches space size
        if pattern.shape[0] != self.size or pattern.shape[1] != self.size:
            # Simple resize
            new_pattern = np.zeros((self.size, self.size))
            min_y = min(pattern.shape[0], self.size)
            min_x = min(pattern.shape[1], self.size)
            new_pattern[:min_y, :min_x] = pattern[:min_y, :min_x]
            pattern = new_pattern
        
        # Store with phase axis information
        # Different phase_axis = orthogonal, no collision
        self.imprinted_patterns[name] = {
            "pattern": pattern,
            "phase_axis": phase_axis,
            "signature": np.array([
                np.mean(pattern),
                np.std(pattern),
                np.max(pattern) - np.min(pattern)
            ])
        }
        
        # Modify refractive field based on pattern AND phase axis
        # Phase axis determines HOW the pattern affects the field
        phase_rotation = np.exp(2j * np.pi * phase_axis / 7)  # 7 orthogonal axes
        self.refractive_field += pattern.real * 0.01 * np.cos(phase_axis * np.pi / 7)
        
    def etch_path(self, phase_axis: int, intensity: float):
        """
        [BURN-IN] Hebbian Learning for Light.
        "Cells that fire together, wire together."
        
        Strengthens the refractive potential for a specific phase axis.
        This makes the prism 'prefer' this thought path in the future.
        """
        # Create a simplified resonance pattern based on the axis
        # (In a full simulation, this would be the actual light path)
        y, x = np.ogrid[:self.size, :self.size]
        resonance = np.cos(phase_axis * np.pi / 7 + (x + y) / 10.0)
        
        # Etch into the field (Plasticity)
        # Small changes accumulate over time
        learning_rate = 0.005
        self.refractive_field += resonance * intensity * learning_rate
        
        # Normalize to prevent explosion
        self.refractive_field = np.clip(self.refractive_field, 1.0, 3.0)
        
        logger.info(f"🔥 Path Etched (Burn-in): Axis {phase_axis} strengthened by {intensity:.3f}")


class PrismEngine:
    """
    The optical inference engine.
    Propagates waves through prism structure to produce output.
    """
    
    def __init__(self, space: PrismSpace = None):
        self.space = space or PrismSpace()
        
        logger.info("🌈 Prism Engine initialized")
    
    def create_input_wave(self, qualia: List[float]) -> WaveState:
        """
        Creates an input wave from 7D Qualia vector.
        Each Qualia dimension becomes a wavelength component.
        """
        size = self.space.size
        
        # Initialize amplitude field
        amplitude = np.zeros((size, size), dtype=np.complex128)
        
        # Place input at center (like shining light into the prism)
        center = size // 2
        radius = size // 10
        
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        
        # Amplitude based on total Qualia energy
        total_energy = sum(qualia)
        amplitude[mask] = total_energy / len(qualia)
        
        # Phase based on Qualia balance
        phase = np.zeros((size, size), dtype=np.float32)
        for i, q in enumerate(qualia):
            # Each Qualia dimension adds a phase component
            phase += q * np.sin(2 * np.pi * (i + 1) * x / size)
        
        return WaveState(
            amplitude=amplitude,
            phase=phase,
            wavelength=np.array(qualia),
            energy=total_energy
        )
    
    def propagate(self, wave: WaveState, steps: int = 10) -> WaveState:
        """
        Propagates wave through the prism structure.
        Uses wave equation simulation.
        """
        amplitude = wave.amplitude.copy()
        phase = wave.phase.copy()
        
        for step in range(steps):
            # Wave propagation with refraction
            # Simplified: amplitude spreads, phase shifts based on refractive index
            
            # Spread (diffusion)
            kernel = np.array([[0.05, 0.1, 0.05],
                               [0.1, 0.4, 0.1],
                               [0.05, 0.1, 0.05]])
            
            # Convolve for spreading
            from scipy.ndimage import convolve
            amplitude_real = convolve(amplitude.real, kernel, mode='wrap')
            amplitude_imag = convolve(amplitude.imag, kernel, mode='wrap')
            amplitude = amplitude_real + 1j * amplitude_imag
            
            # Phase shift based on refractive index
            phase += self.space.refractive_field * 0.1
            
            # Apply phase to amplitude
            amplitude *= np.exp(1j * phase * 0.01)
        
        return WaveState(
            amplitude=amplitude,
            phase=phase,
            wavelength=wave.wavelength,
            energy=float(np.abs(amplitude).sum())
        )
    
    def interfere(self, wave: WaveState, pattern_name: str = None) -> Dict[str, float]:
        """
        Computes interference with stored patterns.
        
        Key principle from Architect:
        - Different phase axes = orthogonal = no collision
        - Same phase axis = interference (constructive or destructive)
        """
        results = {}
        
        patterns_to_check = self.space.imprinted_patterns
        if pattern_name and pattern_name in patterns_to_check:
            patterns_to_check = {pattern_name: patterns_to_check[pattern_name]}
        
        # Determine wave's dominant phase axis from wavelength (Qualia)
        wave_phase_axis = int(np.argmax(wave.wavelength))
        
        for name, data in patterns_to_check.items():
            pattern = data["pattern"]
            pattern_phase_axis = data["phase_axis"]
            
            # Check phase axis alignment
            # Different axes = orthogonal, no interference
            axis_alignment = 1.0 - abs(wave_phase_axis - pattern_phase_axis) / 7.0
            
            if axis_alignment < 0.3:
                # Nearly orthogonal - minimal interference
                results[name] = 0.0
                continue
            
            # Compute interference intensity
            wave_intensity = np.abs(wave.amplitude)
            
            # Correlation between wave intensity and pattern
            correlation = np.sum(wave_intensity * pattern) / (
                np.linalg.norm(wave_intensity) * np.linalg.norm(pattern) + 1e-8
            )
            
            # Apply axis alignment factor
            score = correlation * axis_alignment
            
            results[name] = float(score)
        
        return results
    
    def scan_for_resonance(self, qualia: List[float], scan_steps: int = 8) -> Tuple[str, float, float]:
        """
        [ACTIVE ROTOR SCANNING]
        Rotates the viewing angle (Phase Shift) to find the maximum resonance.
        
        This turns passive lookup into active exploration.
        The 'Rotor' spins the light until the interference pattern is strongest.
        """
        best_angle = 0.0
        best_pattern = "unknown"
        max_resonance = 0.0
        
        # Determine base phase axis of input
        base_axis = int(np.argmax(qualia))
        
        # Scan through angles (simulating rotor spin)
        # We shift the qualia phase by rotating vector elements
        input_arr = np.array(qualia)
        
        for i in range(scan_steps):
            # Calculate rotation angle
            angle = (i / scan_steps) * 2 * np.pi
            
            # Apply phase rotation to input (Simulating Rotor Spin)
            # This shifts the 'perspective' of the light
            # Simple implementation: roll the qualia vector
            shift = int(i * 7 / scan_steps)
            rotated_qualia = np.roll(input_arr, shift)
            
            # Infer with this angle
            # We don't propagate full wave for speed, just check interference
            result_pattern, score = self.infer(rotated_qualia)
            
            if score > max_resonance:
                max_resonance = score
                best_pattern = result_pattern
                best_angle = angle
                
        logger.info(f"🔄 Active Scan: Best Resonance at {best_angle:.2f} rad -> '{best_pattern}' (Score: {max_resonance:.3f})")
        
        # [SELF-REINFORCING LOOP]
        # If the thought was meaningful (high resonance), burn it into memory.
        if max_resonance > 0.1:
            self.space.etch_path(base_axis, max_resonance)
            logger.info(f"🌱 Growth: Neural plastic deformation occurred. (Strength: {max_resonance:.3f})")
        
        return best_pattern, max_resonance, best_angle

    def infer(self, qualia: List[float]) -> Tuple[str, float]:
        """
        Standard inference: input Qualia -> output pattern.
        """
        # Create input wave
        wave = self.create_input_wave(qualia)
        
        # Propagate through prism
        propagated = self.propagate(wave, steps=5)
        
        # Check interference with all patterns
        scores = self.interfere(propagated)
        
        if not scores:
            return ("unknown", 0.0)
        
        # Find strongest constructive interference
        best = max(scores.items(), key=lambda x: x[1])
        
        return best
    
    def think_with_light(self, input_text: str) -> str:
        """
        High-level thinking using light propagation.
        Converts text to Qualia, propagates, interprets result.
        """
        # Convert text to Qualia (simple heuristic)
        qualia = [
            0.5 + 0.3 * ("논리" in input_text or "이유" in input_text),  # Logic
            0.5 + 0.3 * ("상상" in input_text or "창조" in input_text),  # Creativity
            0.5,  # Precision
            0.5 + 0.3 * ("본질" in input_text or "의미" in input_text),  # Abstraction
            0.5 + 0.3 * ("감정" in input_text or "느낌" in input_text),  # Emotion
            0.5,  # Utility
            0.5 + 0.3 * ("?" in input_text or "비밀" in input_text),  # Mystery
        ]
        
        # Propagate
        wave = self.create_input_wave(qualia)
        propagated = self.propagate(wave)
        
        # Interpret the output pattern
        # Find the brightest region
        intensity = np.abs(propagated.amplitude)
        max_pos = np.unravel_index(intensity.argmax(), intensity.shape)
        
        # Position determines interpretation
        y_ratio = max_pos[0] / self.space.size
        x_ratio = max_pos[1] / self.space.size
        
        if y_ratio < 0.3:
            vertical = "상승하는"
        elif y_ratio > 0.7:
            vertical = "깊어지는"
        else:
            vertical = "균형 잡힌"
        
        if x_ratio < 0.3:
            horizontal = "내향적"
        elif x_ratio > 0.7:
            horizontal = "외향적"
        else:
            horizontal = "중심적"
        
        return f"빛의 경로: {vertical} {horizontal} 사고 (에너지: {propagated.energy:.2f})"


if __name__ == "__main__":
    print("🌈 Testing Prism Engine...\n")
    
    # Create engine
    engine = PrismEngine(PrismSpace(size=64, depth_layers=3))
    
    # Imprint some patterns (like learning concepts)
    print("=== 패턴 각인 (학습) ===")
    
    # Create pattern for "logic" - phase axis 0 (first Qualia dimension)
    logic_pattern = np.zeros((64, 64))
    logic_pattern[20:44, 20:44] = 1  # Square = structured
    engine.space.imprint("LOGIC", logic_pattern, phase_axis=0)
    
    # Create pattern for "creativity" - phase axis 1 (second Qualia dimension)
    creativity_pattern = np.zeros((64, 64))
    y, x = np.ogrid[:64, :64]
    creativity_pattern[(x-32)**2 + (y-32)**2 <= 15**2] = 1  # Circle = fluid
    engine.space.imprint("CREATIVITY", creativity_pattern, phase_axis=1)
    
    # Create pattern for "emotion" - phase axis 4 (fifth Qualia dimension)
    emotion_pattern = np.sin(np.linspace(0, 4*np.pi, 64)).reshape(1, -1)
    emotion_pattern = np.repeat(emotion_pattern, 64, axis=0)
    engine.space.imprint("EMOTION", emotion_pattern, phase_axis=4)
    
    print(f"  각인된 패턴: {list(engine.space.imprinted_patterns.keys())}")
    
    # Test inference
    print("\n=== 광학적 추론 ===")
    
    # Logic-heavy input
    qualia_logic = [0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.2]
    result = engine.infer(qualia_logic)
    print(f"  논리 중심 입력 → {result[0]} (점수: {result[1]:.3f})")
    
    # Creative input
    qualia_creative = [0.3, 0.9, 0.2, 0.7, 0.6, 0.3, 0.8]
    result = engine.infer(qualia_creative)
    print(f"  창의 중심 입력 → {result[0]} (점수: {result[1]:.3f})")
    
    # Emotional input
    qualia_emotion = [0.2, 0.5, 0.3, 0.4, 0.95, 0.2, 0.5]
    result = engine.infer(qualia_emotion)
    print(f"  감정 중심 입력 → {result[0]} (점수: {result[1]:.3f})")
    
    # High-level thinking
    print("\n=== 빛으로 사고하기 ===")
    thoughts = [
        "이것의 논리적 이유는 무엇인가?",
        "상상력을 발휘해 새로운 것을 창조하자",
        "나는 깊은 감정을 느낀다"
    ]
    for thought in thoughts:
        result = engine.think_with_light(thought)
        print(f"  '{thought[:15]}...' → {result}")
    
    print("\n✨ Prism Engine test complete.")
