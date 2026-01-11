"""
Dimensional Manifold: The Container of Reality
==============================================

"A dimension is a space where data lives and breathes."

This module defines `Manifold`, the fundamental storage unit of the System.
It is no longer a simulation; it is the **wrapper** for the Real World.

Manifold Types:
1.  **Physical (P)**: The File System (Code, Assets).
2.  **Mental (M)**: The Knowledge Graph (Memories, Vectors).
3.  **Phenomenal (E)**: The Sensory Buffer (Inputs, Logs).

Structure:
- **Fractal**: A Manifold can contain sub-manifolds (Directories/Clusters).
- **Density**: Calculated from actual content size/complexity.
- **Gravity**: Calculated from dependency/reference count.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import os
import math

@dataclass
class Manifold:
    name: str
    domain: str # "Physical", "Mental", "Phenomenal"
    path: Optional[str] = None # Physical path or Virtual ID

    # The Content (Reality)
    content: Dict[str, Any] = field(default_factory=dict)

    # Fractal Structure
    sub_manifolds: Dict[str, 'Manifold'] = field(default_factory=dict)

    # Physics Properties (Calculated)
    mass: float = 0.0
    energy: float = 0.0
    curvature: float = 0.0

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        """
        Syncs the Manifold with Reality.
        """
        if self.domain == "Physical" and self.path:
            self._scan_filesystem()
        elif self.domain == "Phenomenal":
            # Buffer is dynamic, no auto-scan
            pass

        self._calculate_physics()

    def _scan_filesystem(self):
        """
        Maps the File System into Fractal Manifolds.
        """
        p = Path(self.path)
        if not p.exists():
            return

        self.mass = 0.0

        for item in p.iterdir():
            if item.is_dir():
                if item.name.startswith(('.', '__')): continue

                # Recursive Manifold (Fractal)
                sub = Manifold(item.name, self.domain, str(item))
                self.sub_manifolds[item.name] = sub
                self.mass += sub.mass

            elif item.is_file():
                if item.suffix in ['.py', '.md', '.json', '.txt']:
                    size = item.stat().st_size
                    self.content[item.name] = {
                        "type": "file",
                        "size": size,
                        "path": str(item)
                    }
                    self.mass += size

    def inject_content(self, key: str, data: Any, energy: float = 1.0):
        """
        Injects data into the Manifold (e.g., Sensory Input).
        """
        self.content[key] = data
        self.energy += energy
        self.mass += 0.1 # Thoughts have mass
        self._calculate_physics()

    def _calculate_physics(self):
        """
        Updates Curvature/Gravity based on Mass/Energy.
        """
        # Mass = Information Content
        # Energy = Activity/Volatility

        # Curvature = Density (Mass / Radius)
        # Simplified: Logarithmic scale of mass
        if self.mass > 0:
            self.curvature = min(1.0, math.log1p(self.mass) / 20.0)
        else:
            self.curvature = 0.0

        # Add sub-manifold curvature
        for sub in self.sub_manifolds.values():
            self.curvature = max(self.curvature, sub.curvature)

    def intersect(self, other: 'Manifold') -> Dict[str, Any]:
        """
        Calculates the interaction between two real datasets.
        """
        # 1. Metric Distance (Conceptual)
        gravity = (self.curvature + other.curvature) / 2.0

        # 2. Resonance (Content Overlap?)
        # For now, simplistic resonance based on domain compatibility
        resonance = 0.5
        if self.domain == other.domain: resonance = 0.9 # Like attracts Like
        elif self.domain == "Phenomenal" and other.domain == "Mental": resonance = 0.8 # Learning
        elif self.domain == "Mental" and other.domain == "Physical": resonance = 0.7 # Coding

        penetration = gravity * resonance

        return {
            "gravity": gravity,
            "resonance": resonance,
            "penetration": penetration,
            "source": self,
            "target": other
        }

    def __repr__(self):
        return f"Manifold({self.name} | {self.domain} | Mass:{int(self.mass)} | Curve:{self.curvature:.2f})"
