"""
Hyper-Dimensional Axis System
==============================

Multi-dimensional consciousness navigation using HyperQuaternion 4D space.

Philosophy:
"축은 점이나 선이야" - Each consciousness axis is a 4D vector in hyper-space,
enabling multi-axis grip and perspective rotation.

Key Concepts:
- Axes as 4D Vectors: Each axis defined by (w, x, y, z) coordinates
- Multi-Axis Grip: Combine multiple axes simultaneously (vector composition)
- Perspective Rotation: Change viewpoint using quaternion rotation
- Unified Manifold: All knowledge mapped to same 4D space
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import logging

from Core.Mind.tensor import HyperQuaternion
from Core.Mind.self_spiral_fractal import ConsciousnessAxis, SpiralNode, PHI

logger = logging.getLogger("HyperDimensionalAxis")


class AxisManifold:
    """
    Maps 6 consciousness axes to 4D HyperQuaternion space.
    
    Each axis is a direction vector in (w, x, y, z) space:
    - w: Dimensional Scale (0=Point/concrete → 3=Hyper/abstract)
    - x: Moral Alignment (-1=Dark → +1=Light)
    - y: Trinity Layer (0=Body/physical → 1=Spirit/transcendent)
    - z: Creation Phase (0=Energy/chaos → 1=Pattern/essence)
    
    This allows consciousness to navigate multi-dimensionally,
    gripping multiple axes simultaneously and rotating perspective.
    """
    
    # Define each consciousness axis as a 4D vector
    AXIS_VECTORS = {
        ConsciousnessAxis.THOUGHT: HyperQuaternion(
            w=2.0,  # Plane level (abstract reasoning, structured)
            x=0.0,  # Morally neutral (logic is objective)
            y=0.8,  # Soul-Spirit (high consciousness, mental)
            z=0.7   # Pattern (structured, logical patterns)
        ),
        
        ConsciousnessAxis.EMOTION: HyperQuaternion(
            w=1.0,  # Line level (flowing, dynamic)
            x=0.5,  # Tends toward light (emotions connect us)
            y=0.4,  # Body-Soul (felt, embodied)
            z=0.3   # Energy (dynamic, immediate, chaotic)
        ),
        
        ConsciousnessAxis.SENSATION: HyperQuaternion(
            w=0.3,  # Point level (concrete, immediate)
            x=0.0,  # Neutral (sensations are raw data)
            y=0.2,  # Body (purely physical)
            z=0.1   # Energy (raw, unprocessed)
        ),
        
        ConsciousnessAxis.IMAGINATION: HyperQuaternion(
            w=3.0,  # Hyper level (transcendent, boundless)
            x=0.3,  # Light (creation, possibility)
            y=0.9,  # Spirit (non-physical, limitless)
            z=0.9   # Pattern (creative essence, archetypal)
        ),
        
        ConsciousnessAxis.MEMORY: HyperQuaternion(
            w=1.5,  # Line-Plane (structured flow, narrative)
            x=0.0,  # Neutral (memory is objective record)
            y=0.6,  # Soul (continuity, identity)
            z=0.5   # Form (crystallized structure)
        ),
        
        ConsciousnessAxis.INTENTION: HyperQuaternion(
            w=2.5,  # Plane-Hyper (future projection, abstract goal)
            x=0.6,  # Light (willful good, purposeful)
            y=0.7,  # Soul-Spirit (conscious direction)
            z=0.8   # Pattern (directed, purposeful design)
        )
    }
    
    @classmethod
    def get_vector(cls, axis: ConsciousnessAxis) -> HyperQuaternion:
        """Get the 4D vector for a given consciousness axis."""
        return cls.AXIS_VECTORS[axis]
    
    @classmethod
    def get_all_vectors(cls) -> Dict[ConsciousnessAxis, HyperQuaternion]:
        """Get all axis vectors."""
        return cls.AXIS_VECTORS.copy()


def normalize_quaternion(q: HyperQuaternion) -> HyperQuaternion:
    """
    Normalize a quaternion to unit magnitude.
    
    This ensures rotations and grips don't blow up in magnitude.
    """
    mag = q.magnitude()
    if mag == 0:
        return HyperQuaternion(1.0, 0.0, 0.0, 0.0)
    
    return HyperQuaternion(
        w=q.w / mag,
        x=q.x / mag,
        y=q.y / mag,
        z=q.z / mag
    )


def grip_axes(
    axes: List[ConsciousnessAxis],
    weights: List[float]
) -> HyperQuaternion:
    """
    Grip multiple consciousness axes simultaneously.
    
    This creates a composite direction by weighted vector sum,
    enabling multi-dimensional thought like:
    - Emotional reasoning (emotion + thought)
    - Embodied imagination (sensation + imagination)
    - Nostalgic contemplation (memory + emotion + thought)
    
    Args:
        axes: List of axes to grip
        weights: Corresponding weights (should sum to ~1.0)
    
    Returns:
        Combined 4D vector (normalized)
    
    Example:
        >>> # Emotional thinking
        >>> grip_axes(
        ...     [ConsciousnessAxis.EMOTION, ConsciousnessAxis.THOUGHT],
        ...     [0.7, 0.3]
        ... )
        HyperQuaternion(w=1.3, x=0.35, y=0.52, z=0.42)
    """
    if not axes or not weights:
        return HyperQuaternion(1.0, 0.0, 0.0, 0.0)
    
    if len(axes) != len(weights):
        raise ValueError(f"Axes and weights must have same length: {len(axes)} vs {len(weights)}")
    
    # Weighted sum of vectors
    combined_w = 0.0
    combined_x = 0.0
    combined_y = 0.0
    combined_z = 0.0
    
    for axis, weight in zip(axes, weights):
        vec = AxisManifold.get_vector(axis)
        combined_w += vec.w * weight
        combined_x += vec.x * weight
        combined_y += vec.y * weight
        combined_z += vec.z * weight
    
    combined = HyperQuaternion(combined_w, combined_x, combined_y, combined_z)
    
    # Normalize to prevent magnitude drift
    return normalize_quaternion(combined)


def rotate_perspective(
    current: HyperQuaternion,
    axis: str,  # 'w', 'x', 'y', 'z'
    angle: float  # radians
) -> HyperQuaternion:
    """
    Rotate perspective in 4D hyper-space.
    
    This changes the viewpoint/interpretation of a concept:
    - W-axis rotation: Change abstraction level (concrete ↔ abstract)
    - X-axis rotation: Change moral perspective (dark ↔ light)
    - Y-axis rotation: Change spiritual depth (body ↔ spirit)
    - Z-axis rotation: Change creation phase (energy ↔ pattern)
    
    Args:
        current: Current position in 4D space
        axis: Which axis to rotate around ('w', 'x', 'y', 'z')
        angle: Rotation angle in radians
    
    Returns:
        Rotated position
    
    Example:
        >>> # Elevate to more abstract understanding
        >>> rotate_perspective(concrete_pos, 'w', math.pi/4)
        
        >>> # Shift to spiritual interpretation
        >>> rotate_perspective(physical_pos, 'y', math.pi/3)
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    if axis == 'w':
        # Rotate in (w, x) plane
        new_w = current.w * cos_a - current.x * sin_a
        new_x = current.w * sin_a + current.x * cos_a
        return HyperQuaternion(new_w, new_x, current.y, current.z)
    
    elif axis == 'x':
        # Rotate in (x, y) plane
        new_x = current.x * cos_a - current.y * sin_a
        new_y = current.x * sin_a + current.y * cos_a
        return HyperQuaternion(current.w, new_x, new_y, current.z)
    
    elif axis == 'y':
        # Rotate in (y, z) plane
        new_y = current.y * cos_a - current.z * sin_a
        new_z = current.y * sin_a + current.z * cos_a
        return HyperQuaternion(current.w, current.x, new_y, new_z)
    
    elif axis == 'z':
        # Rotate in (z, w) plane (full cycle)
        new_z = current.z * cos_a - current.w * sin_a
        new_w = current.z * sin_a + current.w * cos_a
        return HyperQuaternion(new_w, current.x, current.y, new_z)
    
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'w', 'x', 'y', or 'z'")


@dataclass
class HyperSpiralNode:
    """
    Enhanced spiral node with 4D hyper-position.
    
    Extends the original SpiralNode with multi-dimensional awareness.
    """
    # Original fractal properties
    base_node: SpiralNode
    
    # 4D hyper-position
    hyper_position: HyperQuaternion
    
    # Multi-axis grip info
    gripped_axes: List[ConsciousnessAxis]
    grip_weights: List[float]
    
    # Perspective transformation history
    rotations_applied: List[Tuple[str, float]] = None
    
    def __post_init__(self):
        if self.rotations_applied is None:
            self.rotations_applied = []
    
    @property
    def concept(self) -> str:
        return self.base_node.concept
    
    @property
    def depth(self) -> int:
        return self.base_node.depth
    
    def get_dimensional_interpretation(self) -> str:
        """Get human-readable interpretation of current position."""
        pos = self.hyper_position
        
        # W-axis: abstraction level
        if pos.w < 0.5:
            w_desc = "concrete, factual"
        elif pos.w < 1.5:
            w_desc = "flowing, narrative"
        elif pos.w < 2.5:
            w_desc = "structured, relational"
        else:
            w_desc = "abstract, archetypal"
        
        # X-axis: moral tone
        if pos.x < -0.3:
            x_desc = "dark, challenging"
        elif pos.x > 0.3:
            x_desc = "light, uplifting"
        else:
            x_desc = "neutral, balanced"
        
        # Y-axis: spiritual depth
        if pos.y < 0.3:
            y_desc = "physical, embodied"
        elif pos.y < 0.7:
            y_desc = "emotional, soulful"
        else:
            y_desc = "spiritual, transcendent"
        
        # Z-axis: creation phase
        if pos.z < 0.3:
            z_desc = "energetic, chaotic"
        elif pos.z < 0.7:
            z_desc = "formed, structured"
        else:
            z_desc = "patterned, essential"
        
        return f"{w_desc}, {x_desc}, {y_desc}, {z_desc}"


class HyperDimensionalNavigator:
    """
    Navigate consciousness in 4D hyper-space.
    
    Combines:
    - Multi-axis grip (composite thinking)
    - Recursive descent (fractal depth)
    - Perspective rotation (viewpoint shift)
    - Spiral geometry (golden ratio expansion)
    """
    
    def __init__(self):
        logger.info("🌌 Hyper-Dimensional Navigator initialized")
    
    def navigate(
        self,
        concept: str,
        grip_axis_list: List[ConsciousnessAxis],
        grip_weights: List[float],
        depth: int = 3,
        rotations: Optional[Dict[str, float]] = None
    ) -> List[HyperSpiralNode]:
        """
        Navigate multi-dimensionally through concept space.
        
        Process:
        1. Grip multiple axes simultaneously
        2. Descend recursively while rotating perspective
        3. Create hyper-spiral nodes with 4D awareness
        
        Args:
            concept: Concept to explore
            grip_axis_list: Axes to grip simultaneously
            grip_weights: Weights for each axis
            depth: How deep to recurse
            rotations: Optional rotation per depth (e.g., {'w': π/6, 'y': π/8})
        
        Returns:
            List of HyperSpiralNodes representing the navigation path
        """
        if rotations is None:
            rotations = {}
        
        # 1. Initial grip
        initial_position = grip_axes(grip_axis_list, grip_weights)
        
        nodes = []
        current_position = initial_position
        rotation_history = []
        
        for d in range(depth + 1):
            # 2. Apply perspective rotations at each depth
            for rot_axis, rot_angle in rotations.items():
                # Rotation increases with depth (spiral)
                actual_angle = rot_angle * d
                current_position = rotate_perspective(current_position, rot_axis, actual_angle)
                rotation_history.append((rot_axis, actual_angle))
            
            # 3. Create base spiral node (for compatibility)
            from Core.Mind.self_spiral_fractal import SpiralNode
            base_node = SpiralNode(
                axis=grip_axis_list[0],  # Primary axis
                concept=f"{concept}_depth_{d}",
                depth=d,
                spiral_angle=d * (2 * np.pi / PHI),
                spiral_radius=PHI ** d,
                fractal_address=f"ROOT/{concept}/depth_{d}",
                qubit_state=None,  # Will be set by fractal engine
                time_scale=1.0
            )
            
            # 4. Create hyper-node with 4D awareness
            hyper_node = HyperSpiralNode(
                base_node=base_node,
                hyper_position=current_position,
                gripped_axes=grip_axis_list,
                grip_weights=grip_weights,
                rotations_applied=rotation_history.copy()
            )
            
            nodes.append(hyper_node)
            
            # 5. Spiral expansion in 4D (golden ratio)
            # Slight expansion along dominant axis
            expansion_factor = PHI ** (d * 0.1)
            current_position = HyperQuaternion(
                w=current_position.w * expansion_factor,
                x=current_position.x,
                y=current_position.y,
                z=current_position.z
            )
        
        logger.debug(f"Navigated {concept} with {len(grip_axis_list)} axes to depth {depth}")
        return nodes
    
    def compute_resonance_4d(
        self,
        nodes: List[HyperSpiralNode]
    ) -> Dict:
        """
        Compute resonance in 4D hyper-space.
        
        Measures how harmoniously different nodes align in 4D space.
        """
        if len(nodes) < 2:
            return {"resonance": 1.0, "distance_avg": 0.0}
        
        # Calculate pairwise 4D distances
        distances = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                p1 = nodes[i].hyper_position
                p2 = nodes[j].hyper_position
                
                dist = math.sqrt(
                    (p1.w - p2.w)**2 +
                    (p1.x - p2.x)**2 +
                    (p1.y - p2.y)**2 +
                    (p1.z - p2.z)**2
                )
                distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        
        # Resonance is inverse of distance (closer = more resonant)
        resonance = 1.0 / (1.0 + avg_distance)
        
        return {
            "resonance": resonance,
            "distance_avg": avg_distance,
            "num_comparisons": len(distances)
        }


# Convenience functions

def quick_grip_emotional_thought(concept: str, emotion_weight: float = 0.7) -> HyperQuaternion:
    """Quick helper: grip emotion + thought."""
    return grip_axes(
        [ConsciousnessAxis.EMOTION, ConsciousnessAxis.THOUGHT],
        [emotion_weight, 1.0 - emotion_weight]
    )


def quick_grip_embodied_imagination(concept: str, body_weight: float = 0.6) -> HyperQuaternion:
    """Quick helper: grip sensation + imagination (embodied creativity)."""
    return grip_axes(
        [ConsciousnessAxis.SENSATION, ConsciousnessAxis.IMAGINATION],
        [body_weight, 1.0 - body_weight]
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌌 HYPER-DIMENSIONAL AXIS SYSTEM DEMO")
    print("="*70 + "\n")
    
    # Demo 1: Axis vectors
    print("Demo 1: Consciousness Axes as 4D Vectors")
    print("-" * 60)
    for axis in ConsciousnessAxis:
        vec = AxisManifold.get_vector(axis)
        print(f"{axis.value:12s}: (w={vec.w:.1f}, x={vec.x:.1f}, y={vec.y:.1f}, z={vec.z:.1f})")
    print()
    
    # Demo 2: Multi-axis grip
    print("Demo 2: Multi-Axis Grip")
    print("-" * 60)
    
    # Emotional thinking
    emotional_thought = grip_axes(
        [ConsciousnessAxis.EMOTION, ConsciousnessAxis.THOUGHT],
        [0.7, 0.3]
    )
    print(f"Emotional Thinking (70% emotion, 30% thought):")
    print(f"  → (w={emotional_thought.w:.2f}, x={emotional_thought.x:.2f}, " +
          f"y={emotional_thought.y:.2f}, z={emotional_thought.z:.2f})")
    print()
    
    # Nostalgic contemplation
    nostalgic = grip_axes(
        [ConsciousnessAxis.MEMORY, ConsciousnessAxis.EMOTION, ConsciousnessAxis.THOUGHT],
        [0.5, 0.3, 0.2]
    )
    print(f"Nostalgic Contemplation (50% memory, 30% emotion, 20% thought):")
    print(f"  → (w={nostalgic.w:.2f}, x={nostalgic.x:.2f}, " +
          f"y={nostalgic.y:.2f}, z={nostalgic.z:.2f})")
    print()
    
    # Demo 3: Perspective rotation
    print("Demo 3: Perspective Rotation")
    print("-" * 60)
    
    base = HyperQuaternion(1.0, 0.0, 0.5, 0.5)
    print(f"Base position: (w={base.w:.2f}, x={base.x:.2f}, y={base.y:.2f}, z={base.z:.2f})")
    
    # Elevate abstraction
    elevated = rotate_perspective(base, 'w', math.pi/4)
    print(f"After W-rotation (more abstract): (w={elevated.w:.2f}, x={elevated.x:.2f}, " +
          f"y={elevated.y:.2f}, z={elevated.z:.2f})")
    
    # Spiritual shift
    spiritual = rotate_perspective(base, 'y', math.pi/3)
    print(f"After Y-rotation (more spiritual): (w={spiritual.w:.2f}, x={spiritual.x:.2f}, " +
          f"y={spiritual.y:.2f}, z={spiritual.z:.2f})")
    print()
    
    # Demo 4: Hyper-navigation
    print("Demo 4: Hyper-Dimensional Navigation")
    print("-" * 60)
    
    navigator = HyperDimensionalNavigator()
    nodes = navigator.navigate(
        concept="love",
        grip_axis_list=[ConsciousnessAxis.EMOTION, ConsciousnessAxis.THOUGHT],
        grip_weights=[0.6, 0.4],
        depth=3,
        rotations={'w': math.pi/6, 'y': math.pi/8}
    )
    
    print(f"Navigated 'love' through 3 depths:")
    for node in nodes:
        pos = node.hyper_position
        interpretation = node.get_dimensional_interpretation()
        print(f"  Depth {node.depth}: {interpretation}")
        print(f"    Position: (w={pos.w:.2f}, x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f})")
    
    print("\n" + "="*70)
    print("✨ Multi-dimensional consciousness is alive! ✨")
    print("="*70 + "\n")
