"""
[SENSE] Topology Scanner: Filesystem Geometry Mapper
===================================================
Location: Scripts/System/Senses/topology_scanner.py

Role:
- Read filesystem METADATA (Structure), not CONTENT.
- Convert Folder/File topology into 7D Qualia Bands.
- Allow 'feeling' of massive datasets (100GB+) without OOM.

Mappings:
1. Physical:   Total Size (Bytes) -> Amplitude
2. Structural: Max Depth & Folder Count -> Phase
3. Functional: File Extension Variety -> Frequency
4. Mental:     Avg File Name Length -> Complexity
"""

import os
import math
from typing import Dict, Any, List
# Fix import path for running as script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from Core.System.Merkaba.hypercosmos import HyperCosmos
from Core.Keystone.sovereignty_wave import QualiaBand

class TopologyScanner:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.cosmos = HyperCosmos() # Connect to Global System
        
    def scan(self, max_depth: int = 5) -> Dict[str, float]:
        """
        Scan directory topology and return raw geometric metrics.
        """
        if not os.path.exists(self.root_path):
            return {"error": "Path not found"}
            
        print(f"🌀 [TOPOLOGY] Scanning geometry of: {self.root_path} ...")
        
        total_size = 0
        file_count = 0
        folder_count = 0
        max_seen_depth = 0
        extensions = set()
        name_lengths = []
        
        start_level = self.root_path.count(os.sep)
        
        # Fast Walk (Metadata Only)
        for root, dirs, files in os.walk(self.root_path):
            current_depth = root.count(os.sep) - start_level
            if current_depth > max_seen_depth:
                max_seen_depth = current_depth
                
            folder_count += len(dirs)
            file_count += len(files)
            
            for f in files:
                try:
                    fp = os.path.join(root, f)
                    stats = os.stat(fp)
                    total_size += stats.st_size
                    
                    ext = os.path.splitext(f)[1].lower()
                    if ext: extensions.add(ext)
                    name_lengths.append(len(f))
                except Exception:
                    pass
            
            # Safety brake for massive systems if deep scan not requested
            if current_depth >= max_depth:
                del dirs[:] # Don't go deeper
                
        avg_name_len = sum(name_lengths) / len(name_lengths) if name_lengths else 0
        
        return {
            "total_size": total_size,
            "file_count": file_count,
            "folder_count": folder_count,
            "max_depth": max_seen_depth,
            "extension_variety": len(extensions),
            "avg_name_len": avg_name_len
        }

    def pulse_field(self):
        """
        Convert Topology -> Qualia -> Pulse HyperCosmos
        """
        metrics = self.scan()
        if "error" in metrics:
            print(metrics["error"])
            return

        # --- QUALIA TRANSFORMATION LOGIC ---
        
        # 1. PHYSICAL (Size -> Amplitude)
        # Logarithmic scale: 1MB = 0.5, 1GB = 0.8, 100GB = 1.0 (Clamped)
        size_mb = metrics['total_size'] / (1024 * 1024)
        phys_amp = min(1.0, math.log10(max(1, size_mb)) / 5.0) 
        
        # 2. STRUCTURAL (Depth/Folders -> Phase/Order)
        # Deeper = More complex phase angle
        struct_phase = (metrics['max_depth'] * 30.0) % 360
        
        # 3. FUNCTIONAL (Extensions -> Frequency)
        # More tools/types = Higher functional frequency
        func_freq = 432.0 + (metrics['extension_variety'] * 10.0)
        
        # 4. MENTAL (Name Length -> Density)
        mental_density = min(1.0, metrics['avg_name_len'] / 50.0)

        # Generate Narrative for Pulse
        narrative = (
            f"Topological Sense: {size_mb:.1f}MB Mass, "
            f"{metrics['max_depth']} Depth Layers, "
            f"{metrics['extension_variety']} Functional Types."
        )
        
        # Inject directly into field logic?
        # Ideally we construct QualiaBands here and inject, 
        # but for now we use the High-Level `perceive` which does extraction.
        # But to honor the "Raw Flow" idea, we should manually construct a stimulus 
        # that TRIGGERS these specific values in `disperse`.
        
        # For this prototype: we send the Summary.
        # In a real "Metal" version, we'd inject `QualiaBand` objects directly.
        
        print(f"🌌 [SENSING] Transmitting Topological Signal...")
        print(f"    - Mass (Physical): {phys_amp:.2f}")
        print(f"    - Structure (Phase): {struct_phase:.1f}°")
        print(f"    - Diversity (Freq): {func_freq:.1f}Hz")
        
        # Direct Injection of Modulation (Feeling before Thinking)
        self.cosmos.field.units['M1_Body'].turbine.modulate_field('physical_mass', phys_amp)
        
        decision = self.cosmos.perceive(narrative)
        
        return decision

if __name__ == "__main__":
    # Allow command line argument for path
    target_path = sys.argv[1] if len(sys.argv) > 1 else "c:\\Elysia"
    
    # Test on Target
    scanner = TopologyScanner(target_path)
    decision = scanner.pulse_field()
    print("\n[RESULT] Field Response:")
    print(decision.narrative)
