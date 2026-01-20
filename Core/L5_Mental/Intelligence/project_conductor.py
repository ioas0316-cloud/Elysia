"""
Project Conductor (The Overseer)
================================
Core.L5_Mental.Intelligence.project_conductor

"A conductor does not play an instrument. He aligns the resonance."

This module scans the entire codebase, treating every file as a Rotor.
It generates the "Galaxy Map" of the Project.

Capabilities:
1. Recursive Rotor Scanning.
2. System Health Aggregation.
3. Frequency Analysis (Is the project more "Mental" or "Physical"?).
"""

import os
import sys

# Ensure Core is visible (Fix for ModuleNotFoundError)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import List, Dict
from Core.L1_Foundation.Foundation.Code.code_rotor import CodeRotor
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA

class ProjectConductor:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.rotors: List[CodeRotor] = []
        self.system_dna = WaveDNA(label="SystemAverage")
        
    def scan_project(self):
        """Recursively loads all .py files as Rotors."""
        print(f"🔭 [Conductor] Scanning Galaxy at {self.root_path}...")
        self.rotors = []
        
        for root, dirs, files in os.walk(self.root_path):
            if "Sandbox" in root: continue # Skip Sandbox (Chaos Zone)
            if "__pycache__" in root: continue
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rotor = CodeRotor(full_path)
                    print(f"   Detected: {rotor.name} ({rotor.health})")
                    self.rotors.append(rotor)
                    
        self._calculate_system_dna()
        
    def _calculate_system_dna(self):
        """Averages the DNA of all Rotors."""
        if not self.rotors: return
        
        count = len(self.rotors)
        avg_dna = WaveDNA()
        for r in self.rotors:
            avg_dna.physical += r.dna.physical
            avg_dna.functional += r.dna.functional
            avg_dna.phenomenal += r.dna.phenomenal
            avg_dna.causal += r.dna.causal
            avg_dna.mental += r.dna.mental
            avg_dna.structural += r.dna.structural
            avg_dna.spiritual += r.dna.spiritual
            
        # Normalize
        avg_dna.physical /= count
        avg_dna.functional /= count
        avg_dna.phenomenal /= count
        avg_dna.causal /= count
        avg_dna.mental /= count
        avg_dna.structural /= count
        avg_dna.spiritual /= count
        
        avg_dna.normalize()
        self.system_dna = avg_dna

    def report(self):
        """Prints the God's Eye View."""
        print("\n" + "="*40)
        print(f"🌌 SYSTEM STATUS REPORT: {os.path.basename(self.root_path)}")
        print("="*40)
        print(f"Rotors Active: {len(self.rotors)}")
        
        # Health Check
        healthy = sum(1 for r in self.rotors if r.health == "Healthy")
        print(f"System Health: {healthy}/{len(self.rotors)} ({(healthy/len(self.rotors))*100:.1f}%)")
        
        # DNA Analysis
        print(f"\n🧠 System Soul (Average DNA):")
        print(f"{self.system_dna}")
        
        # Top 3 Rotors by Complexity (Frequency)
        sorted_rotors = sorted(self.rotors, key=lambda r: r.dna.frequency, reverse=True)
        print("\n🔥 Highest Energy Centers (Complexity):")
        for i, r in enumerate(sorted_rotors[:3]):
            print(f"  {i+1}. {r.name} ({r.dna.frequency:.1f} Hz) - {r.diagnose()}")
            
        print("\n" + "="*40)

if __name__ == "__main__":
    # Scan Elysia
    root = "c:/Elysia"
    conductor = ProjectConductor(root)
    conductor.scan_project()
    conductor.report()
