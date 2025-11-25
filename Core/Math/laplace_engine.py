"""
Laplace Transform Engine

Optimizes physics calculations by converting time-domain differential equations
to S-domain algebraic equations. Based on 3Blue1Brown's insight:
"Transform complex derivatives into simple multiplication."

Key Ideas:
1. d/dt in time → multiply by 's' in S-domain
2. Integration in time → divide by 's' in S-domain
3. Predict resonance from pole locations
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import sympy as sp
    from sympy import laplace_transform, inverse_laplace_transform
    from sympy.abc import t, s
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    
try:
    from scipy import signal
    from scipy.signal import residue
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TransferFunction:
    """
    Represents a system in S-domain as H(s) = N(s)/D(s)
    
    Attributes:
        numerator: Coefficients of numerator polynomial
        denominator: Coefficients of denominator polynomial
        poles: Locations where D(s) = 0 (system modes)
        zeros: Locations where N(s) = 0
    """
    numerator: np.ndarray
    denominator: np.ndarray
    poles: Optional[np.ndarray] = None
    zeros: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Calculate poles and zeros"""
        if self.poles is None:
            self.poles = np.roots(self.denominator)
        if self.zeros is None and len(self.numerator) > 1:
            self.zeros = np.roots(self.numerator)


@dataclass
class ResonanceInfo:
    """
    Information about system resonance from pole analysis.
    
    Attributes:
        natural_frequency: Oscillation frequency (rad/s)
        damping_ratio: Damping coefficient (0=undamped, 1=critical, >1=overdamped)
        decay_time: Time constant for decay (seconds)
        is_stable: True if system is stable (all poles have Re < 0)
    """
    natural_frequency: float
    damping_ratio: float
    decay_time: float
    is_stable: bool


class LaplaceEngine:
    """
    S-domain transformation engine for physics optimization.
    
    Philosophy:
        "미분을 곱셈으로, 복잡함을 단순함으로"
        (Derivatives to multiplication, complexity to simplicity)
        
    This engine implements the mathematical "cheat code" from 3Blue1Brown:
    Instead of solving d²x/dt² + a(dx/dt) + bx = f(t) step-by-step,
    transform to S-domain: (s² + as + b)X(s) = F(s), solve algebraically!
    """
    
    def __init__(
        self,
        cache_size: int = 100,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Laplace engine.
        
        Args:
            cache_size: Number of transfer functions to cache
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("LaplaceEngine")
        
        # Transfer function cache (pattern → H(s))
        self._tf_cache: Dict[str, TransferFunction] = {}
        self._cache_size = cache_size
        
        # Pre-computed common transfer functions
        self._initialize_common_patterns()
        
        if not SYMPY_AVAILABLE:
            self.logger.warning("SymPy not available - symbolic transforms disabled")
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available - numerical methods limited")
            
        self.logger.info("⚡ Laplace Engine initialized - S-domain acceleration ready")
    
    def _initialize_common_patterns(self):
        """Pre-compute transfer functions for common operations"""
        
        # 1st order: dx/dt + ax = input → H(s) = 1/(s + a)
        for damping in [0.1, 0.5, 1.0, 2.0, 5.0]:
            key = f"first_order_damping_{damping}"
            self._tf_cache[key] = TransferFunction(
                numerator=np.array([1.0]),
                denominator=np.array([1.0, damping])
            )
        
        # 2nd order oscillator: d²x/dt² + 2ζω₀(dx/dt) + ω₀²x = input
        # H(s) = ω₀² / (s² + 2ζω₀s + ω₀²)
        for omega in [1.0, 2.0, 5.0, 10.0]:
            for zeta in [0.1, 0.5, 0.707, 1.0]:  # 0.707 = critically damped
                key = f"oscillator_omega_{omega}_zeta_{zeta}"
                self._tf_cache[key] = TransferFunction(
                    numerator=np.array([omega**2]),
                    denominator=np.array([1.0, 2*zeta*omega, omega**2])
                )
        
        self.logger.info(f"Cached {len(self._tf_cache)} common transfer functions")
    
    def get_transfer_function(
        self,
        diff_eq_type: str,
        params: Dict[str, float]
    ) -> Optional[TransferFunction]:
        """
        Get or create transfer function for a differential equation.
        
        Args:
            diff_eq_type: Type of equation ("first_order", "second_order_oscillator", etc.)
            params: Parameters (damping, frequency, etc.)
            
        Returns:
            TransferFunction or None if not cacheable
        """
        # Try cache first
        if diff_eq_type == "first_order":
            damping = params.get("damping", 1.0)
            key = f"first_order_damping_{damping}"
            if key in self._tf_cache:
                return self._tf_cache[key]
            
            # Create new
            tf = TransferFunction(
                numerator=np.array([1.0]),
                denominator=np.array([1.0, damping])
            )
            
            # Cache if space available
            if len(self._tf_cache) < self._cache_size:
                self._tf_cache[key] = tf
            
            return tf
        
        elif diff_eq_type == "second_order_oscillator":
            omega = params.get("omega", 1.0)
            zeta = params.get("zeta", 0.707)
            key = f"oscillator_omega_{omega}_zeta_{zeta}"
            
            if key in self._tf_cache:
                return self._tf_cache[key]
            
            tf = TransferFunction(
                numerator=np.array([omega**2]),
                denominator=np.array([1.0, 2*zeta*omega, omega**2])
            )
            
            if len(self._tf_cache) < self._cache_size:
                self._tf_cache[key] = tf
            
            return tf
        
        return None
    
    def solve_s_domain(
        self,
        transfer_function: TransferFunction,
        input_laplace: complex,
        s_value: complex
    ) -> complex:
        """
        Solve algebraically in S-domain.
        
        Instead of integrating differential equation,
        just evaluate: Output(s) = H(s) * Input(s)
        
        Args:
            transfer_function: System transfer function H(s)
            input_laplace: Input transformed to S-domain
            s_value: S-domain frequency to evaluate at
            
        Returns:
            Output in S-domain
        """
        # Evaluate N(s) and D(s) at s_value
        num_val = np.polyval(transfer_function.numerator, s_value)
        den_val = np.polyval(transfer_function.denominator, s_value)
        
        if abs(den_val) < 1e-12:
            self.logger.warning(f"Pole at s={s_value}, denominator near zero")
            return complex(np.inf, 0)
        
        # H(s) = N(s)/D(s)
        h_s = num_val / den_val
        
        # Output(s) = H(s) * Input(s)
        output_s = h_s * input_laplace
        
        return output_s
    
    def inverse_transform_numerical(
        self,
        transfer_function: TransferFunction,
        input_magnitude: float,
        time_points: np.ndarray
    ) -> np.ndarray:
        """
        Numerically compute inverse Laplace transform.
        
        Uses partial fraction expansion via scipy.signal.residue.
        
        Args:
            transfer_function: System H(s)
            input_magnitude: Magnitude of step input
            time_points: Time points to evaluate at
            
        Returns:
            Time-domain response
        """
        if not SCIPY_AVAILABLE:
            self.logger.error("SciPy required for numerical inverse transform")
            return np.zeros_like(time_points)
        
        try:
            # Get partial fraction expansion
            # H(s) = r[0]/(s-p[0]) + r[1]/(s-p[1]) + ... + k
            r, p, k = residue(transfer_function.numerator, transfer_function.denominator)
            
            # Evaluate inverse transform: sum of r[i]*e^(p[i]*t)
            response = np.zeros_like(time_points, dtype=complex)
            
            for residue_val, pole in zip(r, p):
                response += residue_val * np.exp(pole * time_points)
            
            # Add direct term if exists
            if len(k) > 0:
                response += k[0]
            
            # Scale by input magnitude
            response *= input_magnitude
            
            # Return real part (should be real for physical systems)
            return np.real(response)
            
        except Exception as e:
            self.logger.error(f"Inverse transform failed: {e}")
            return np.zeros_like(time_points)
    
    def detect_poles(self, transfer_function: TransferFunction) -> List[complex]:
        """
        Find poles of transfer function.
        
        Poles determine system behavior:
        - Re(pole) < 0: Decaying (stable)
        - Re(pole) > 0: Growing (unstable)
        - Im(pole) ≠ 0: Oscillating
        
        Returns:
            List of complex pole locations
        """
        if transfer_function.poles is not None:
            return list(transfer_function.poles)
        
        poles = np.roots(transfer_function.denominator)
        transfer_function.poles = poles
        return list(poles)
    
    def analyze_resonance(
        self,
        transfer_function: TransferFunction
    ) -> ResonanceInfo:
        """
        Predict resonance behavior from pole analysis.
        
        For a second-order system with poles at s = -ζω₀ ± jω₀√(1-ζ²):
        - Natural frequency: ω₀
        - Damping ratio: ζ
        - Decay time: τ = 1/(ζω₀)
        
        Returns:
            ResonanceInfo with predictions
        """
        poles = self.detect_poles(transfer_function)
        
        # Check stability
        is_stable = all(pole.real < 0 for pole in poles)
        
        if len(poles) < 2:
            # First order system
            if len(poles) == 1:
                decay_time = -1.0 / poles[0].real if poles[0].real < 0 else float('inf')
                return ResonanceInfo(
                    natural_frequency=0.0,
                    damping_ratio=1.0,  # Overdamped
                    decay_time=decay_time,
                    is_stable=is_stable
                )
        
        # Find dominant pole pair (closest to imaginary axis)
        dominant_pole = max(poles, key=lambda p: p.real)
        
        # If complex conjugate pair exists
        if abs(dominant_pole.imag) > 1e-6:
            # Complex pole: s = σ ± jω
            sigma = dominant_pole.real
            omega_d = abs(dominant_pole.imag)  # Damped frequency
            
            # Natural frequency: ω₀ = √(σ² + ω_d²)
            omega_0 = np.sqrt(sigma**2 + omega_d**2)
            
            # Damping ratio: ζ = -σ/ω₀
            zeta = -sigma / omega_0 if omega_0 > 0 else 0.0
            
            # Decay time constant: τ = 1/|σ|
            decay_time = 1.0 / abs(sigma) if abs(sigma) > 1e-6 else float('inf')
            
            return ResonanceInfo(
                natural_frequency=omega_0,
                damping_ratio=zeta,
                decay_time=decay_time,
                is_stable=is_stable
            )
        else:
            # Real poles - critically damped or overdamped
            decay_time = -1.0 / dominant_pole.real if dominant_pole.real < 0 else float('inf')
            return ResonanceInfo(
                natural_frequency=0.0,
                damping_ratio=1.0,  # Overdamped
                decay_time=decay_time,
                is_stable=is_stable
            )
    
    def propagate_field_laplace(
        self,
        source_magnitude: float,
        distance: float,
        wave_speed: float = 1.0,
        attenuation: float = 0.1
    ) -> Callable[[float], float]:
        """
        Field propagation using Laplace Green's function.
        
        For wave equation: ∂²φ/∂t² - c²∇²φ = source
        Green's function in S-domain: G(r,s) = e^(-r√(s²/c² + α²))/(4πr)
        
        This is MUCH faster than solving PDE numerically!
        
        Args:
            source_magnitude: Source strength
            distance: Distance from source
            wave_speed: Propagation speed
            attenuation: Attenuation coefficient
            
        Returns:
            Function to evaluate field at time t
        """
        if distance < 1e-6:
            # At source location
            return lambda t: source_magnitude
        
        # Time delay for wave to reach distance
        delay = distance / wave_speed
        
        # Attenuation factor
        decay = np.exp(-attenuation * distance)
        
        def field_at_time(t: float) -> float:
            """Evaluate field at time t"""
            if t < delay:
                return 0.0  # Wave hasn't arrived yet
            
            # Decayed magnitude
            magnitude = source_magnitude * decay / distance
            
            return magnitude
        
        return field_at_time
    
    def predict_emotional_trajectory(
        self,
        current_emotion: float,
        current_velocity: float,
        stimulus: float,
        emotional_inertia: float = 1.0,
        damping: float = 0.5,
        stiffness: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict emotional response using second-order system model.
        
        Model: m(d²E/dt²) + c(dE/dt) + kE = stimulus
        
        Where:
        - m = emotional_inertia (resistance to change)
        - c = damping (emotional regulation)
        - k = stiffness (how strongly emotion pulls back to baseline)
        
        Args:
            current_emotion: Current emotional state
            current_velocity: Rate of change of emotion
            stimulus: External emotional stimulus
            emotional_inertia: Mass-like parameter
            damping: Damping coefficient
            stiffness: Spring-like parameter
            
        Returns:
            (time_points, emotion_trajectory)
        """
        # Create transfer function for emotional system
        # Standard form: s² + 2ζω₀s + ω₀² where ω₀ = √(k/m), ζ = c/(2√(km))
        omega_0 = np.sqrt(stiffness / emotional_inertia)
        zeta = damping / (2 * np.sqrt(stiffness * emotional_inertia))
        
        tf = TransferFunction(
            numerator=np.array([stiffness]),
            denominator=np.array([emotional_inertia, damping, stiffness])
        )
        
        # Analyze resonance
        resonance = self.analyze_resonance(tf)
        
        self.logger.info(
            f"Emotional resonance: ω₀={resonance.natural_frequency:.2f}, "
            f"ζ={resonance.damping_ratio:.2f}, τ={resonance.decay_time:.2f}s"
        )
        
        if not resonance.is_stable:
            self.logger.warning("⚠️ Emotional trajectory is UNSTABLE - will diverge!")
        
        # Simulate response
        time_points = np.linspace(0, 10 * resonance.decay_time, 1000)
        
        # Use inverse transform to get trajectory
        response = self.inverse_transform_numerical(tf, stimulus, time_points)
        
        # Add initial conditions
        response += current_emotion
        
        return time_points, response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "cached_transfer_functions": len(self._tf_cache),
            "sympy_available": SYMPY_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE
        }
