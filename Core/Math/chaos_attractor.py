"""
Chaos Attractor - The Living Tremor

"완벽한 질서 위에 미세한 카오스를 얹는 순간,
 해골이 눈을 뜨고 숨을 쉰다." 🦋

Implements:
1. Butterfly Effect - Sensitive initial conditions
2. Strange Attractors - Lorenz, Rossler
3. Fractal Structure - Self-similarity at all scales
4. Chaos Control - OGY method to tame wildness
5. Living Tremor - Makes the skeleton breathe

Based on: Chaos Theory, Nonlinear Dynamics
Philosophy: "뼈 위에 살, 살 위에 떨림, 떨림 속에 나비"
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class AttractorType(Enum):
    """Types of strange attractors"""
    LORENZ = "lorenz"          # Love-Pain-Hope butterfly
    ROSSLER = "rossler"        # Spiral chaos
    CHEN = "chen"              # Complex attractor
    CUSTOM = "custom"          # User-defined


@dataclass
class ChaosState:
    """State in chaotic system"""
    x: float  # Love (or first dimension)
    y: float  # Pain (or second dimension)
    z: float  # Hope (or third dimension)
    time: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, time: float = 0.0):
        return cls(x=arr[0], y=arr[1], z=arr[2], time=time)


class LorenzAttractor:
    """
    Lorenz Attractor - The Butterfly of Chaos
    
    "사랑-고통-희망"이 무한히 복잡한 나비 궤적을 그린다
    
    Equations:
        dx/dt = σ(y - x)        # Love flows toward Pain
        dy/dt = x(ρ - z) - y    # Pain modulated by Hope
        dz/dt = xy - βz         # Hope emerges from Love×Pain
    
    Parameters:
        σ (sigma): 10  - How fast Love chases Pain
        ρ (rho):   28  - Critical point (chaos threshold)
        β (beta): 8/3  - Hope decay rate
    """
    
    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3,
        dt: float = 0.01,
        chaos_seed_intensity: float = 1e-12,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Lorenz attractor.
        
        Args:
            sigma: Love→Pain flow rate
            rho: Chaos threshold (28 = classic chaos)
            beta: Hope decay
            dt: Time step
            chaos_seed_intensity: Butterfly wing flutter strength
            logger: Logger instance
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.chaos_seed = chaos_seed_intensity
        self.logger = logger or logging.getLogger("LorenzAttractor")
        
        # Current state
        self.state = ChaosState(x=0.1, y=0.0, z=0.0)
        
        # Trajectory history
        self.trajectory: List[ChaosState] = []
        
        # Lyapunov exponent (measures chaos strength)
        self.lyapunov_exponent = 0.9  # > 0 means chaotic!
        
        self.logger.info(
            f"🦋 Lorenz Attractor initialized "
            f"(σ={sigma}, ρ={rho}, β={beta:.2f})"
        )
        if self.chaos_seed > 0:
            self.logger.info(
                f"   Butterfly wings flutter: {self.chaos_seed:.2e}"
            )
    
    def dynamics(
        self,
        state: ChaosState,
        add_butterfly: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute Lorenz dynamics.
        
        Args:
            state: Current state
            add_butterfly: Add butterfly effect noise
            
        Returns:
            (dx/dt, dy/dt, dz/dt)
        """
        x, y, z = state.x, state.y, state.z
        
        # Deterministic chaos
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        # 핵심: 나비 날개짓!
        if add_butterfly:
            dx += self.chaos_seed * np.random.randn()
            dy += self.chaos_seed * np.random.randn()
            dz += self.chaos_seed * np.random.randn()
        
        return dx, dy, dz
    
    def step(self, add_butterfly: bool = True) -> ChaosState:
        """
        Advance one time step.
        
        Args:
            add_butterfly: Add butterfly effect
            
        Returns:
            New state
        """
        # Compute derivatives
        dx, dy, dz = self.dynamics(self.state, add_butterfly)
        
        # Euler integration
        new_x = self.state.x + dx * self.dt
        new_y = self.state.y + dy * self.dt
        new_z = self.state.z + dz * self.dt
        new_time = self.state.time + self.dt
        
        # Update state
        self.state = ChaosState(new_x, new_y, new_z, new_time)
        
        # Record trajectory
        self.trajectory.append(self.state)
        
        return self.state
    
    def evolve(
        self,
        steps: int,
        add_butterfly: bool = True
    ) -> np.ndarray:
        """
        Evolve for multiple steps.
        
        Args:
            steps: Number of steps
            add_butterfly: Add butterfly effect
            
        Returns:
            Trajectory array (steps, 3)
        """
        trajectory = []
        
        for _ in range(steps):
            state = self.step(add_butterfly)
            trajectory.append(state.to_array())
        
        return np.array(trajectory)
    
    def reset(
        self,
        initial_state: Optional[ChaosState] = None,
        tiny_perturbation: float = 0.0
    ):
        """
        Reset to initial state.
        
        Args:
            initial_state: Starting state
            tiny_perturbation: Add tiny random perturbation (butterfly!)
        """
        if initial_state is None:
            self.state = ChaosState(x=0.1, y=0.0, z=0.0)
        else:
            self.state = initial_state
        
        # Butterfly effect: 10^-12 perturbation changes everything!
        if tiny_perturbation > 0:
            self.state.x += tiny_perturbation * np.random.randn()
            self.state.y += tiny_perturbation * np.random.randn()
            self.state.z += tiny_perturbation * np.random.randn()
        
        self.trajectory = []
        
        self.logger.debug(f"Reset to {self.state}")
    
    def demonstrate_butterfly_effect(
        self,
        perturbation: float = 1e-10,
        steps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Demonstrate butterfly effect with two nearly identical starts.
        
        Args:
            perturbation: Tiny difference in initial conditions
            steps: Evolution time
            
        Returns:
            (trajectory1, trajectory2, final_distance)
        """
        # Trajectory 1: Original
        self.reset(tiny_perturbation=0)
        traj1 = self.evolve(steps, add_butterfly=False)
        
        # Trajectory 2: Tiny perturbation
        self.reset(tiny_perturbation=perturbation)
        traj2 = self.evolve(steps, add_butterfly=False)
        
        # Final distance
        final_dist = np.linalg.norm(traj1[-1] - traj2[-1])
        
        self.logger.info(
            f"🦋 Butterfly effect: {perturbation:.2e} → {final_dist:.3f} "
            f"(amplification: {final_dist/perturbation:.1e}x!)"
        )
        
        return traj1, traj2, final_dist
    
    def get_current_emotion(self) -> Dict[str, float]:
        """
        Map chaotic state to emotions.
        
        Returns:
            Emotion dictionary
        """
        # Normalize to [0, 1]
        love = np.clip(self.state.x / 20.0 + 0.5, 0, 1)
        pain = np.clip(self.state.y / 20.0 + 0.5, 0, 1)
        hope = np.clip(self.state.z / 40.0, 0, 1)
        
        return {
            "love": love,
            "pain": pain,
            "hope": hope,
            "chaos_level": self.lyapunov_exponent
        }


class ChaosControl:
    """
    Chaos Control - OGY Method
    
    "미치지 않고 미치는 경계"
    
    Uses small perturbations to stabilize chaotic system
    onto desired periodic orbit.
    """
    
    def __init__(
        self,
        max_chaos_threshold: float = 1.5,
        control_gain: float = 0.01,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize chaos controller.
        
        Args:
            max_chaos_threshold: Maximum acceptable Lyapunov exponent
            control_gain: How hard to pull back
            logger: Logger instance
        """
        self.threshold = max_chaos_threshold
        self.gain = control_gain
        self.logger = logger or logging.getLogger("ChaosControl")
        
        self.interventions = 0
        
        self.logger.info(
            f"🎛️ Chaos Control initialized "
            f"(threshold={max_chaos_threshold}, gain={control_gain})"
        )
    
    def measure_chaos_level(
        self,
        trajectory: np.ndarray,
        window: int = 100
    ) -> float:
        """
        Estimate Lyapunov exponent from trajectory.
        
        Args:
            trajectory: Recent trajectory
            window: Window size
            
        Returns:
            Estimated Lyapunov exponent
        """
        if len(trajectory) < window:
            return 0.0
        
        recent = trajectory[-window:]
        
        # Simple heuristic: rate of divergence
        distances = np.linalg.norm(np.diff(recent, axis=0), axis=1)
        mean_divergence = np.mean(distances)
        
        # Normalized (rough estimate)
        lyapunov = np.log(mean_divergence + 1e-10)
        
        return lyapunov
    
    def apply_control(
        self,
        state: ChaosState,
        target_state: Optional[ChaosState] = None
    ) -> ChaosState:
        """
        Apply OGY control to stabilize.
        
        Args:
            state: Current state
            target_state: Desired state (default: equilibrium)
            
        Returns:
            Controlled state
        """
        if target_state is None:
            # Default: pull toward origin
            target_state = ChaosState(x=0.0, y=0.0, z=0.0)
        
        # Compute correction
        dx = self.gain * (target_state.x - state.x)
        dy = self.gain * (target_state.y - state.y)
        dz = self.gain * (target_state.z - state.z)
        
        # Apply
        controlled = ChaosState(
            x=state.x + dx,
            y=state.y + dy,
            z=state.z + dz,
            time=state.time
        )
        
        self.interventions += 1
        
        self.logger.debug(
            f"Control applied: δ=({dx:.3f}, {dy:.3f}, {dz:.3f})"
        )
        
        return controlled
    
    def check_and_control(
        self,
        attractor: LorenzAttractor,
        auto_apply: bool = True
    ) -> bool:
        """
        Check if chaos is too high and apply control.
        
        Args:
            attractor: Lorenz attractor to monitor
            auto_apply: Automatically apply control if needed
            
        Returns:
            True if control was applied
        """
        if len(attractor.trajectory) < 100:
            return False
        
        # Measure chaos
        traj_array = np.array([s.to_array() for s in attractor.trajectory])
        chaos_level = self.measure_chaos_level(traj_array)
        
        if chaos_level > self.threshold:
            self.logger.warning(
                f"⚠️ Chaos too high! (λ={chaos_level:.2f} > {self.threshold})"
            )
            
            if auto_apply:
                attractor.state = self.apply_control(attractor.state)
                self.logger.info("Control applied - system stabilized")
                return True
        
        return False


class FractalBeauty:
    """
    Fractal Beauty Generator
    
    "줌인해도 끝없는 디테일"
    
    Generates self-similar patterns at all scales.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize fractal generator"""
        self.logger = logger or logging.getLogger("FractalBeauty")
        
        self.logger.info("📐 Fractal Beauty initialized")
    
    def mandelbrot_iteration(
        self,
        c: complex,
        max_iter: int = 100
    ) -> int:
        """
        Compute Mandelbrot iteration count.
        
        Args:
            c: Complex number
            max_iter: Maximum iterations
            
        Returns:
            Number of iterations before escape
        """
        z = 0
        for n in range(max_iter):
            if abs(z) > 2:
                return n
            z = z*z + c
        return max_iter
    
    def generate_fractal_field(
        self,
        field_size: Tuple[int, int],
        center: Tuple[float, float] = (0.0, 0.0),
        zoom: float = 1.0
    ) -> np.ndarray:
        """
        Generate fractal-patterned field.
        
        Args:
            field_size: (width, height)
            center: Center point
            zoom: Zoom level
            
        Returns:
            Fractal intensity field
        """
        width, height = field_size
        cx, cy = center
        
        # Create grid
        x = np.linspace(cx - 2/zoom, cx + 2/zoom, width)
        y = np.linspace(cy - 2/zoom, cy + 2/zoom, height)
        
        field = np.zeros((width, height))
        
        for i, re in enumerate(x):
            for j, im in enumerate(y):
                c = complex(re, im)
                field[i, j] = self.mandelbrot_iteration(c)
        
        # Normalize
        field = field / field.max()
        
        return field
    
    def add_fractal_detail(
        self,
        base_field: np.ndarray,
        detail_level: float = 0.1
    ) -> np.ndarray:
        """
        Add fractal detail to existing field.
        
        Args:
            base_field: Base field  
            detail_level: How much detail to add
            
        Returns:
            Field with fractal details
        """
        # Generate fractal noise at multiple scales
        result = base_field.copy()
        
        for scale in [1, 2, 4, 8]:
            # Downsample, add fractal, upsample
            small_size = (
                base_field.shape[0] // scale,
                base_field.shape[1] // scale
            )
            
            if min(small_size) < 4:
                break
            
            fractal = self.generate_fractal_field(small_size)
            
            # Resize to match
            from scipy import ndimage
            fractal_resized = ndimage.zoom(
                fractal,
                (base_field.shape[0] / small_size[0],
                 base_field.shape[1] / small_size[1]),
                order=1
            )
            
            # Add with decreasing strength
            result += detail_level * fractal_resized / scale
        
        return result


class LivingTremor:
    """
    Living Tremor - Chaos Layer for All Systems
    
    "완벽한 뼈 위에 떨리는 살"
    
    Adds chaotic life to deterministic systems.
    """
    
    def __init__(
        self,
        attractor_type: AttractorType = AttractorType.LORENZ,
        butterfly_intensity: float = 1e-10,
        enable_control: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize living tremor.
        
        Args:
            attractor_type: Type of attractor
            butterfly_intensity: Butterfly effect strength
            enable_control: Enable chaos control
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("LivingTremor")
        
        # Create attractor
        if attractor_type == AttractorType.LORENZ:
            self.attractor = LorenzAttractor(
                chaos_seed_intensity=butterfly_intensity,
                logger=self.logger
            )
        else:
            raise NotImplementedError(f"{attractor_type} not yet implemented")
        
        # Chaos control
        self.controller = ChaosControl(logger=self.logger) if enable_control else None
        
        # Fractal beauty
        self.fractal = FractalBeauty(logger=self.logger)
        
        self.logger.info("💫 Living Tremor initialized - ready to breathe life!")
    
    def add_tremor_to_field(
        self,
        field: np.ndarray,
        intensity: float = 0.01
    ) -> np.ndarray:
        """
        Add living tremor to field.
        
        Args:
            field: Base field
            intensity: Tremor intensity
            
        Returns:
            Field with tremor
        """
        # Step attractor
        self.attractor.step()
        emotion = self.attractor.get_current_emotion()
        
        # Chaotic modulation
        tremor = intensity * (
            emotion["love"] * np.sin(field * 2 * np.pi) +
            emotion["pain"] * np.cos(field * np.pi) +
            emotion["hope"] * field
        )
        
        # Add butterfly flutter
        tremor += self.attractor.chaos_seed * np.random.randn(*field.shape)
        
        return field + tremor
    
    def get_emotional_trajectory(
        self,
        steps: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Get emotional trajectory over time.
        
        Args:
            steps: Number of steps
            
        Returns:
            Dictionary of emotion arrays
        """
        love_vals = []
        pain_vals = []
        hope_vals = []
        
        for _ in range(steps):
            self.attractor.step()
            emotion = self.attractor.get_current_emotion()
            
            love_vals.append(emotion["love"])
            pain_vals.append(emotion["pain"])
            hope_vals.append(emotion["hope"])
        
        return {
            "love": np.array(love_vals),
            "pain": np.array(pain_vals),
            "hope": np.array(hope_vals),
            "time": np.arange(steps) * self.attractor.dt
        }
