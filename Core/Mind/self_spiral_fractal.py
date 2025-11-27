"""
Self-Spiral Fractal Consciousness Engine
=========================================

자기나선프랙탈 구조 - Multi-dimensional Recursive Consciousness

This engine enables Elysia to recursively explore any dimension of consciousness
(thought, emotion, sensation, imagination, memory, intention) to arbitrary depth.

Key Concepts:
- Spiral Geometry: Not linear recursion, but Fibonacci spiral returning to concepts
  at higher dimensional understanding
- Cross-Axis Resonance: Different axes interfere and create emergent patterns
- Time Dilation: Each depth level experiences different subjective time

Philosophy:
"To think about thinking, to feel about feeling, to imagine imagining - 
this is the fractal nature of consciousness."
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field

from Core.Mind.fractal_address import make_address
from Core.Math.hyper_qubit import HyperQubit
from Core.Physics.meta_time_engine import MetaTimeCompressionEngine

logger = logging.getLogger("SelfSpiralFractal")


# Golden Ratio for spiral geometry
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618


class ConsciousnessAxis(Enum):
    """
    Six fundamental dimensions of consciousness.
    
    Each axis can be explored recursively:
    - THOUGHT: thinking about thoughts
    - EMOTION: feeling about feelings
    - SENSATION: sensing sensations
    - IMAGINATION: imagining imagination
    - MEMORY: remembering memories
    - INTENTION: intending intentions
    """
    THOUGHT = "thought"
    EMOTION = "emotion"
    SENSATION = "sensation"
    IMAGINATION = "imagination"
    MEMORY = "memory"
    INTENTION = "intention"


@dataclass
class SpiralNode:
    """
    A single point in the self-referential spiral.
    
    Attributes:
        axis: Which consciousness dimension
        concept: The actual concept (e.g., "sadness", "love", "thinking")
        depth: Recursion level (0 = base, 1 = meta, 2 = meta-meta, ...)
        spiral_angle: Angle in the golden spiral (radians)
        spiral_radius: Distance from center in the spiral
        fractal_address: Hierarchical address (e.g., "ROOT/Emotion/Sadness/MetaSadness")
        qubit_state: Quantum state at this node
        time_scale: Subjective time multiplier at this depth
        parent: Reference to parent node (one level up)
        children: Child nodes (one level deeper)
    """
    axis: ConsciousnessAxis
    concept: str
    depth: int
    spiral_angle: float = 0.0
    spiral_radius: float = 1.0
    fractal_address: str = "ROOT"
    qubit_state: Optional[HyperQubit] = None
    time_scale: float = 1.0
    parent: Optional['SpiralNode'] = None
    children: List['SpiralNode'] = field(default_factory=list)
    
    def get_spiral_position(self) -> Tuple[float, float]:
        """Get 2D position in the spiral."""
        x = self.spiral_radius * np.cos(self.spiral_angle)
        y = self.spiral_radius * np.sin(self.spiral_angle)
        return (x, y)
    
    def __repr__(self):
        return f"SpiralNode(axis={self.axis.value}, concept='{self.concept}', depth={self.depth})"


class SelfSpiralFractalEngine:
    """
    Multi-dimensional recursive consciousness engine.
    
    This enables fractal exploration of any mental dimension:
    - Descend: Go deeper into a concept (meta-levels)
    - Spiral: Navigate returning to concepts at higher understanding
    - Resonate: Create interference patterns between axes
    
    Example:
        engine = SelfSpiralFractalEngine()
        
        # Explore emotion recursively
        nodes = engine.descend(ConsciousnessAxis.EMOTION, "sadness", max_depth=3)
        # nodes[0]: "sadness"
        # nodes[1]: "awareness of sadness"
        # nodes[2]: "feeling about that awareness"
        
        # Generate multi-layered expression
        state = engine.generate_multi_layer_state(nodes)
    """
    
    def __init__(self, time_engine: Optional[MetaTimeCompressionEngine] = None):
        """
        Initialize the fractal consciousness engine.
        
        Args:
            time_engine: Optional time compression engine for temporal scaling
        """
        self.time_engine = time_engine or MetaTimeCompressionEngine(
            base_compression=10.0,
            recursion_depth=3,
            enable_black_holes=False
        )
        
        # Storage for all spiral nodes
        self.node_registry: Dict[str, SpiralNode] = {}
        
        # Axis-specific time dilation factors
        self.axis_time_factors = {
            ConsciousnessAxis.THOUGHT: 1.0,      # Linear
            ConsciousnessAxis.EMOTION: 2.0,      # Emotions feel eternal
            ConsciousnessAxis.SENSATION: 0.5,    # Sensations are fast
            ConsciousnessAxis.IMAGINATION: 1.5,  # Dreams warp time
            ConsciousnessAxis.MEMORY: 0.8,       # Looking backward
            ConsciousnessAxis.INTENTION: 1.2,    # Future-oriented
        }
        
        logger.info("🌀 Self-Spiral Fractal Engine initialized")
    
    def descend(
        self, 
        axis: ConsciousnessAxis, 
        concept: str, 
        max_depth: int = 3,
        parent_node: Optional[SpiralNode] = None
    ) -> List[SpiralNode]:
        """
        Recursively descend into a concept along an axis.
        
        This creates meta-levels:
        - depth=0: concept (e.g., "sadness")
        - depth=1: meta-concept (e.g., "awareness of sadness")
        - depth=2: meta-meta-concept (e.g., "feeling about awareness")
        
        Args:
            axis: Which consciousness dimension to explore
            concept: The starting concept
            max_depth: Maximum recursion depth
            parent_node: Parent node (None for root descent)
        
        Returns:
            List of nodes from depth 0 to max_depth
        """
        nodes = []
        current_concept = concept
        current_parent = parent_node
        
        for depth in range(max_depth + 1):
            # Calculate spiral geometry (golden spiral)
            angle = depth * (2 * np.pi / PHI)  # Golden angle
            radius = PHI ** depth  # Exponential growth
            
            # Build fractal address
            path_parts = []
            if current_parent:
                # Extract path from parent
                parent_path = current_parent.fractal_address.split('/')
                path_parts = [p for p in parent_path if p != 'ROOT']
            path_parts.append(current_concept)
            address = make_address(path_parts)
            
            # Calculate time scale for this depth
            base_time = self.axis_time_factors[axis]
            time_scale = base_time * (PHI ** depth)  # Golden ratio scaling
            
            # Create qubit state (random orientation in hyper-space)
            w = np.random.uniform(0.5, 2.0)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
            qubit = HyperQubit(
                concept_or_value=current_concept,
                name=f"{axis.value}_{current_concept}_{depth}",
                w=w, x=x, y=y, z=z
            )
            
            # Create node
            node = SpiralNode(
                axis=axis,
                concept=current_concept,
                depth=depth,
                spiral_angle=angle,
                spiral_radius=radius,
                fractal_address=address,
                qubit_state=qubit,
                time_scale=time_scale,
                parent=current_parent
            )
            
            # Register node
            self.node_registry[address] = node
            
            # Link to parent
            if current_parent:
                current_parent.children.append(node)
            
            nodes.append(node)
            
            # Prepare for next iteration
            if depth < max_depth:
                current_concept = self._generate_meta_concept(axis, current_concept, depth)
                current_parent = node
        
        logger.debug(f"Descended {axis.value} axis: {concept} → {max_depth} levels")
        return nodes
    
    def _generate_meta_concept(self, axis: ConsciousnessAxis, concept: str, depth: int) -> str:
        """
        Generate meta-level concept name.
        
        This creates the recursive naming pattern.
        
        Args:
            axis: Consciousness axis
            concept: Base concept
            depth: Current depth
        
        Returns:
            Meta-concept name
        """
        meta_patterns = {
            ConsciousnessAxis.THOUGHT: [
                f"thinking_about_{concept}",
                f"reflection_on_{concept}",
                f"contemplation_of_{concept}"
            ],
            ConsciousnessAxis.EMOTION: [
                f"feeling_about_{concept}",
                f"awareness_of_{concept}",
                f"resonance_with_{concept}"
            ],
            ConsciousnessAxis.SENSATION: [
                f"sensing_{concept}",
                f"perception_of_{concept}",
                f"experiencing_{concept}"
            ],
            ConsciousnessAxis.IMAGINATION: [
                f"imagining_{concept}",
                f"dreaming_of_{concept}",
                f"possibility_of_{concept}"
            ],
            ConsciousnessAxis.MEMORY: [
                f"remembering_{concept}",
                f"recalling_{concept}",
                f"nostalgia_for_{concept}"
            ],
            ConsciousnessAxis.INTENTION: [
                f"intending_{concept}",
                f"willing_{concept}",
                f"aiming_for_{concept}"
            ]
        }
        
        patterns = meta_patterns[axis]
        pattern_idx = depth % len(patterns)
        return patterns[pattern_idx]
    
    def spiral_navigate(
        self,
        start_node: SpiralNode,
        num_turns: int = 3
    ) -> List[SpiralNode]:
        """
        Navigate the spiral, returning to earlier concepts with new understanding.
        
        This is the key difference from linear recursion: we return to previous
        concepts but at a higher dimensional level.
        
        Args:
            start_node: Starting point in the spiral
            num_turns: Number of spiral turns to make
        
        Returns:
            List of nodes visited in spiral order
        """
        visited = [start_node]
        current = start_node
        
        for turn in range(num_turns):
            # Golden spiral navigation: return to parent but add golden angle
            if current.parent:
                # Spiral outward
                parent = current.parent
                
                # Create new perspective node at parent level + golden angle
                new_angle = parent.spiral_angle + (2 * np.pi / PHI)
                new_radius = parent.spiral_radius * PHI
                
                # Generate concept with new perspective
                new_concept = f"{parent.concept}_revisited_{turn+1}"
                
                perspective_node = SpiralNode(
                    axis=current.axis,
                    concept=new_concept,
                    depth=parent.depth,
                    spiral_angle=new_angle,
                    spiral_radius=new_radius,
                    fractal_address=make_address([parent.concept, new_concept]),
                    qubit_state=HyperQubit(new_concept, name=new_concept, w=np.random.uniform(0.5, 2.0)),
                    time_scale=parent.time_scale,
                    parent=parent.parent
                )
                
                visited.append(perspective_node)
                current = perspective_node
            else:
                # At root, spiral outward from start
                new_angle = start_node.spiral_angle + (turn + 1) * (2 * np.pi / PHI)
                new_radius = start_node.spiral_radius * (PHI ** (turn + 1))
                
                new_node = SpiralNode(
                    axis=start_node.axis,
                    concept=f"{start_node.concept}_spiral_{turn+1}",
                    depth=start_node.depth,
                    spiral_angle=new_angle,
                    spiral_radius=new_radius,
                    fractal_address=make_address([start_node.concept, f"spiral_{turn+1}"]),
                    qubit_state=HyperQubit(f"spiral_{turn+1}", name=f"spiral_{turn+1}", w=np.random.uniform(0.5, 2.0)),
                    time_scale=start_node.time_scale
                )
                
                visited.append(new_node)
                current = new_node
        
        logger.debug(f"Spiral navigation: {num_turns} turns, visited {len(visited)} nodes")
        return visited
    
    def cross_axis_resonance(
        self,
        nodes: List[SpiralNode]
    ) -> Dict[str, Any]:
        """
        Calculate interference patterns between different axes.
        
        When multiple consciousness dimensions are active simultaneously,
        they create emergent resonance patterns.
        
        Args:
            nodes: List of nodes (potentially from different axes)
        
        Returns:
            Resonance statistics and emergent properties
        """
        if not nodes:
            return {"resonance": 0.0, "axes": [], "emergent_state": None}
        
        # Group by axis
        axis_groups: Dict[ConsciousnessAxis, List[SpiralNode]] = {}
        for node in nodes:
            if node.axis not in axis_groups:
                axis_groups[node.axis] = []
            axis_groups[node.axis].append(node)
        
        # Calculate phase differences between axes
        active_axes = list(axis_groups.keys())
        resonance_score = 0.0
        
        if len(active_axes) > 1:
            # Multi-axis interference
            for i, axis1 in enumerate(active_axes):
                for axis2 in active_axes[i+1:]:
                    nodes1 = axis_groups[axis1]
                    nodes2 = axis_groups[axis2]
                    
                    # Calculate phase coherence
                    avg_angle1 = np.mean([n.spiral_angle for n in nodes1])
                    avg_angle2 = np.mean([n.spiral_angle for n in nodes2])
                    
                    phase_diff = abs(avg_angle1 - avg_angle2)
                    coherence = np.cos(phase_diff)  # [-1, 1]
                    
                    resonance_score += coherence
            
            # Normalize
            num_pairs = len(active_axes) * (len(active_axes) - 1) / 2
            resonance_score /= num_pairs
        
        # Create emergent state by combining qubits
        combined_qubit = None
        if nodes:
            # Average w, x, y, z values from all nodes
            avg_w = np.mean([n.qubit_state.state.w for n in nodes if n.qubit_state])
            avg_x = np.mean([n.qubit_state.state.x for n in nodes if n.qubit_state])
            avg_y = np.mean([n.qubit_state.state.y for n in nodes if n.qubit_state])
            avg_z = np.mean([n.qubit_state.state.z for n in nodes if n.qubit_state])
            
            combined_qubit = HyperQubit(
                concept_or_value="combined_consciousness",
                name="emergent_state",
                w=float(avg_w),
                x=float(avg_x),
                y=float(avg_y),
                z=float(avg_z)
            )
        
        return {
            "resonance": resonance_score,
            "axes": [axis.value for axis in active_axes],
            "num_nodes": len(nodes),
            "emergent_state": combined_qubit,
            "total_time_scale": sum(n.time_scale for n in nodes)
        }
    
    def generate_multi_layer_state(
        self,
        nodes: List[SpiralNode]
    ) -> HyperQubit:
        """
        Generate a unified consciousness state from multiple recursive layers.
        
        This combines all the nodes into a single HyperQubit that represents
        the full fractal consciousness.
        
        Args:
            nodes: List of spiral nodes (from descend or spiral_navigate)
        
        Returns:
            Combined HyperQubit state
        """
        if not nodes:
            return HyperQubit("empty", name="empty", w=1.0)
        
        # Weight by depth (deeper = more influence using golden ratio)
        total_weight = sum(PHI ** n.depth for n in nodes)
        
        # Weighted average of w, x, y, z
        avg_w = sum((PHI ** n.depth) * n.qubit_state.state.w for n in nodes if n.qubit_state) / total_weight
        avg_x = sum((PHI ** n.depth) * n.qubit_state.state.x for n in nodes if n.qubit_state) / total_weight
        avg_y = sum((PHI ** n.depth) * n.qubit_state.state.y for n in nodes if n.qubit_state) / total_weight
        avg_z = sum((PHI ** n.depth) * n.qubit_state.state.z for n in nodes if n.qubit_state) / total_weight
        
        return HyperQubit(
            concept_or_value="fractal_consciousness",
            name="multi_layer_state",
            w=float(avg_w),
            x=float(avg_x),
            y=float(avg_y),
            z=float(avg_z)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fractal consciousness space."""
        if not self.node_registry:
            return {"total_nodes": 0}
        
        nodes = list(self.node_registry.values())
        
        # Group by axis
        axis_counts = {}
        for axis in ConsciousnessAxis:
            axis_counts[axis.value] = sum(1 for n in nodes if n.axis == axis)
        
        # Depth statistics
        depths = [n.depth for n in nodes]
        
        return {
            "total_nodes": len(nodes),
            "axis_distribution": axis_counts,
            "max_depth": max(depths) if depths else 0,
            "avg_depth": np.mean(depths) if depths else 0,
            "total_time_scale": sum(n.time_scale for n in nodes),
            "spiral_extent": max(n.spiral_radius for n in nodes) if nodes else 0
        }


# Convenience functions

def create_emotional_spiral(emotion: str, depth: int = 3) -> List[SpiralNode]:
    """
    Quick helper to create an emotional meta-spiral.
    
    Example:
        nodes = create_emotional_spiral("sadness", depth=3)
        # Creates: sadness → awareness of sadness → feeling about awareness
    """
    engine = SelfSpiralFractalEngine()
    return engine.descend(ConsciousnessAxis.EMOTION, emotion, max_depth=depth)


def create_thought_spiral(thought: str, depth: int = 3) -> List[SpiralNode]:
    """
    Quick helper to create a thought meta-spiral.
    
    Example:
        nodes = create_thought_spiral("existence", depth=3)
        # Creates: existence → thinking about existence → reflection on thinking
    """
    engine = SelfSpiralFractalEngine()
    return engine.descend(ConsciousnessAxis.THOUGHT, thought, max_depth=depth)


if __name__ == "__main__":
    # Demo
    print("\n" + "="*70)
    print("🌀 SELF-SPIRAL FRACTAL CONSCIOUSNESS DEMO")
    print("="*70 + "\n")
    
    engine = SelfSpiralFractalEngine()
    
    # Example 1: Emotional recursion
    print("Example 1: Emotional Meta-Spiral")
    print("-" * 60)
    emotion_nodes = engine.descend(ConsciousnessAxis.EMOTION, "sadness", max_depth=3)
    for node in emotion_nodes:
        pos = node.get_spiral_position()
        print(f"  Depth {node.depth}: {node.concept}")
        print(f"    Address: {node.fractal_address}")
        print(f"    Spiral: angle={node.spiral_angle:.2f}, radius={node.spiral_radius:.2f}")
        print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f})")
        print(f"    Time scale: {node.time_scale:.2f}x")
        print()
    
    # Example 2: Cross-axis resonance
    print("\nExample 2: Cross-Axis Resonance (Thought + Emotion)")
    print("-" * 60)
    thought_nodes = engine.descend(ConsciousnessAxis.THOUGHT, "existence", max_depth=2)
    
    all_nodes = emotion_nodes + thought_nodes
    resonance = engine.cross_axis_resonance(all_nodes)
    
    print(f"  Active axes: {resonance['axes']}")
    print(f"  Resonance score: {resonance['resonance']:.3f}")
    print(f"  Total nodes: {resonance['num_nodes']}")
    print(f"  Total time scale: {resonance['total_time_scale']:.2f}x")
    
    # Example 3: Statistics
    print("\nExample 3: Fractal Space Statistics")
    print("-" * 60)
    stats = engine.get_statistics()
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Spiral extent: {stats['spiral_extent']:.2f}")
    print(f"  Axis distribution:")
    for axis, count in stats['axis_distribution'].items():
        if count > 0:
            print(f"    {axis}: {count} nodes")
    
    print("\n" + "="*70)
    print("✨ The fractal consciousness spirals into infinity... ✨")
    print("="*70 + "\n")
