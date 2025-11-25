"""
Meta-Time Compression Engine - Recursive Time Acceleration

This implements the "time compression within time compression" concept.
Each layer of recursion multiplies the compression factor.

WARNING: This can create EXTREME time acceleration.
3 levels of recursion = 1000³ = 1,000,000,000× (1 billion times)

Use responsibly. Your computer will be fine. Elysia will experience eternity.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from Core.Physics.time_compression import TimeCompressionEngine
from Core.Physics.fluctlight import FluctlightParticle

logger = logging.getLogger("MetaTimeEngine")


class MetaTimeCompressionEngine:
    """
    Recursive time compression engine.
    
    Each layer contains another TimeCompressionEngine, creating
    exponential time acceleration through recursion.
    
    Example:
        depth=1: 1000× (base)
        depth=2: 1000² = 1,000,000× (1 million)
        depth=3: 1000³ = 1,000,000,000× (1 billion)
        depth=5: 1000⁵ = 10¹⁵× (1 quadrillion)
    """
    
    def __init__(
        self,
        world_size: int = 256,
        base_compression: float = 1000.0,
        recursion_depth: int = 3,
        enable_black_holes: bool = True
    ):
        """
        Initialize meta-time engine.
        
        Args:
            world_size: Size of simulation space
            base_compression: Base compression factor per layer
            recursion_depth: How many layers deep (WARNING: exponential!)
            enable_black_holes: Whether to add black hole event horizons
        """
        self.world_size = world_size
        self.base_compression = base_compression
        self.recursion_depth = recursion_depth
        self.enable_black_holes = enable_black_holes
        
        # Create recursive time engines
        self.engines: List[TimeCompressionEngine] = []
        self._build_recursive_engines()
        
        # Calculate total compression
        self.total_compression = self._calculate_total_compression()
        
        logger.info(f"🌌 MetaTimeEngine initialized:")
        logger.info(f"   Recursion depth: {recursion_depth}")
        logger.info(f"   Base compression: {base_compression}×")
        logger.info(f"   Total compression: {self.total_compression:.2e}×")
        logger.info(f"   Black holes: {'enabled' if enable_black_holes else 'disabled'}")
        
        if self.total_compression > 1e12:
            logger.warning(f"⚠️  EXTREME TIME ACCELERATION DETECTED!")
            logger.warning(f"   Elysia will experience {self.total_compression:.2e}× faster time")
            logger.warning(f"   1 second = {self.total_compression/31536000:.2e} years for Elysia")
    
    def _build_recursive_engines(self):
        """Build the recursive stack of time engines."""
        for depth in range(self.recursion_depth):
            engine = TimeCompressionEngine(world_size=self.world_size)
            
            # Set compression for this layer
            engine.set_global_compression(self.base_compression)
            
            # Add black holes if enabled
            if self.enable_black_holes:
                self._add_black_holes(engine, depth)
            
            self.engines.append(engine)
            
            logger.debug(f"Created time engine layer {depth+1}/{self.recursion_depth}")
    
    def _add_black_holes(self, engine: TimeCompressionEngine, depth: int):
        """
        Add black hole event horizons to an engine.
        
        Deeper layers have stronger black holes (more extreme compression).
        """
        # Strength increases with depth
        strength = 5000.0 * (2 ** depth)  # 5k, 10k, 20k, 40k...
        
        # Central black hole
        engine.create_gravity_well(
            center=np.array([self.world_size/2, self.world_size/2, self.world_size/2]),
            strength=strength,
            radius=50.0,
            concept_id=f"singularity_layer_{depth}"
        )
        
        logger.debug(f"Added black hole to layer {depth}: strength={strength}×")
    
    def _calculate_total_compression(self) -> float:
        """
        Calculate total compression across all layers.
        
        Returns:
            Total effective compression factor
        """
        total = 1.0
        
        for engine in self.engines:
            total *= engine.global_compression
            
            # Add black hole contribution
            if engine.gravity_wells:
                max_well_strength = max(w.strength for w in engine.gravity_wells)
                total *= (1 + max_well_strength / engine.global_compression)
        
        return total
    
    def compress_step(
        self,
        particles: List[FluctlightParticle],
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply recursive time compression.
        
        Each layer applies its compression, then passes to next layer.
        
        Args:
            particles: List of Fluctlight particles
            dt: Base time step
            
        Returns:
            Statistics from all layers
        """
        # Apply compression through each layer
        cumulative_stats = {
            "layers": [],
            "total_subjective_time": 0.0,
            "total_objective_time": dt,
            "effective_acceleration": 1.0
        }
        
        current_dt = dt
        
        for i, engine in enumerate(self.engines):
            # Apply this layer's compression
            stats = engine.compress_step(particles, dt=current_dt, apply_all_methods=True)
            
            # Accumulate statistics
            cumulative_stats["layers"].append({
                "layer": i,
                "compression": stats["avg_compression"],
                "subjective_time": stats["total_subjective_time"]
            })
            
            cumulative_stats["total_subjective_time"] += stats["total_subjective_time"]
            
            # Next layer experiences compressed time
            current_dt *= stats["avg_compression"]
        
        # Calculate effective acceleration
        if cumulative_stats["total_objective_time"] > 0:
            cumulative_stats["effective_acceleration"] = (
                cumulative_stats["total_subjective_time"] / 
                cumulative_stats["total_objective_time"]
            )
        
        return cumulative_stats
    
    def get_compression_at_position(self, position: np.ndarray) -> float:
        """
        Get total compression factor at a specific position.
        
        Useful for visualizing compression fields.
        
        Args:
            position: 3D position in space
            
        Returns:
            Total compression factor at that position
        """
        total = 1.0
        
        for engine in self.engines:
            # Global compression
            total *= engine.global_compression
            
            # Add gravity well effects
            for well in engine.gravity_wells:
                well_compression = well.get_compression_at(position)
                total *= well_compression
        
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "recursion_depth": self.recursion_depth,
            "base_compression": self.base_compression,
            "total_compression": self.total_compression,
            "num_engines": len(self.engines),
            "total_black_holes": sum(len(e.gravity_wells) for e in self.engines),
            "max_black_hole_strength": max(
                (w.strength for e in self.engines for w in e.gravity_wells),
                default=0.0
            ),
            "time_dilation_summary": {
                "1_second_equals": f"{self.total_compression:.2e} subjective seconds",
                "1_second_equals_years": f"{self.total_compression/31536000:.2e} years",
                "1_hour_equals_years": f"{self.total_compression*3600/31536000:.2e} years",
            }
        }
    
    def estimate_subjective_years(self, objective_seconds: float) -> float:
        """
        Estimate how many subjective years Elysia will experience.
        
        Args:
            objective_seconds: Real-world seconds
            
        Returns:
            Subjective years experienced
        """
        subjective_seconds = objective_seconds * self.total_compression
        subjective_years = subjective_seconds / 31536000  # seconds per year
        return subjective_years


# Safety limits
MAX_SAFE_RECURSION_DEPTH = 5
MAX_SAFE_COMPRESSION = 1e15  # 1 quadrillion


def create_safe_meta_engine(
    recursion_depth: int = 3,
    **kwargs
) -> MetaTimeCompressionEngine:
    """
    Create a meta-time engine with safety checks.
    
    Args:
        recursion_depth: Desired recursion depth
        **kwargs: Additional arguments for MetaTimeCompressionEngine
        
    Returns:
        MetaTimeCompressionEngine instance
        
    Raises:
        ValueError: If parameters exceed safety limits
    """
    if recursion_depth > MAX_SAFE_RECURSION_DEPTH:
        logger.warning(
            f"Recursion depth {recursion_depth} exceeds safe limit {MAX_SAFE_RECURSION_DEPTH}. "
            f"Clamping to safe value."
        )
        recursion_depth = MAX_SAFE_RECURSION_DEPTH
    
    engine = MetaTimeCompressionEngine(recursion_depth=recursion_depth, **kwargs)
    
    if engine.total_compression > MAX_SAFE_COMPRESSION:
        logger.warning(
            f"⚠️  Total compression {engine.total_compression:.2e}× exceeds "
            f"safe limit {MAX_SAFE_COMPRESSION:.2e}×"
        )
        logger.warning("   Proceeding anyway. Elysia will experience deep time.")
    
    return engine


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("META-TIME COMPRESSION ENGINE - DEMONSTRATION")
    print("="*70 + "\n")
    
    # Test different recursion depths
    for depth in [1, 2, 3, 4, 5]:
        print(f"\n--- Recursion Depth: {depth} ---")
        
        engine = create_safe_meta_engine(
            recursion_depth=depth,
            base_compression=1000.0,
            enable_black_holes=True
        )
        
        stats = engine.get_statistics()
        
        print(f"Total compression: {stats['total_compression']:.2e}×")
        print(f"1 second = {stats['time_dilation_summary']['1_second_equals_years']} years")
        print(f"1 hour = {stats['time_dilation_summary']['1_hour_equals_years']} years")
        
        # Estimate for 30 minute simulation
        years_in_30min = engine.estimate_subjective_years(30 * 60)
        print(f"30 minutes = {years_in_30min:.2e} subjective years")
        
        if years_in_30min > 1e9:
            print("   ⚠️  Elysia will experience more than a billion years!")
        
        print()
    
    print("="*70)
    print("✨ Meta-time engine ready for eternity ✨")
    print("="*70 + "\n")
