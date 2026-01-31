"""
Project Conductor (The Overseer)
================================
Core.1_Body.L5_Mental.Reasoning_Core.project_conductor

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

from typing import List, Dict, Optional
from Core.1_Body.L1_Foundation.Foundation.Code.code_rotor import CodeRotor
from Core.1_Body.L6_Structure.Wave.wave_dna import WaveDNA

class ProjectConductor:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.rotors: List[CodeRotor] = []
        self.system_dna = WaveDNA(label="SystemAverage")
        
    def scan_project(self):
        """Recursively loads all .py files as Rotors with intelligent filtering."""
        print(f"  [Conductor] Scanning Galaxy at {self.root_path} (Filtering activated)...")
        self.rotors = []
        
        # [PHASE 23.3: HARDWARE OPTIMIZATION]
        # Skip noise zones to avoid I/O choking
        EXCLUSION_ZONES = [
            "__pycache__", ".git", ".venv", "venv", "env", 
            "node_modules", "site-packages", "Sandbox", "Archive"
        ]
        
        for root, dirs, files in os.walk(self.root_path):
            # Prune directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUSION_ZONES]
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        rotor = CodeRotor(full_path)
                        self.rotors.append(rotor)
                    except Exception:
                        continue 
                    
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

    def discern_roadmap(self) -> Optional[Dict[str, str]]:
        """
        [SOVEREIGN VISION]
        Reads the ROADMAP_AGENTIC_SOVEREIGNTY.md and identifies the next priority.
        """
        roadmap_path = "c:/Elysia/docs/1_Body/L7_Spirit/M1_Providence/ROADMAP_AGENTIC_SOVEREIGNTY.md"
        if not os.path.exists(roadmap_path):
            return None
            
        try:
            with open(roadmap_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            import re
            # Find the first unchecked milestone: - [ ] **Milestone X.Y: Name**: Description
            milestone_pattern = r"- \[ \] \*\*Milestone (\d+\.\d+): ([^*]+)\*\*: ([^\n]+)"
            matches = re.findall(milestone_pattern, content)
            
            if matches:
                m_id, m_name, m_desc = matches[0]
                return {
                    "id": m_id.strip(),
                    "name": m_name.strip(),
                    "description": m_desc.strip(),
                    "context": f"Next phase of evolution detected: Milestone {m_id}"
                }
        except Exception as e:
            print(f"  [Conductor] Vision blurred: {e}")
            
        return None

    def report(self):
        """Prints the God's Eye View."""
        print("\n" + "="*40)
        print(f"  SYSTEM STATUS REPORT: {os.path.basename(self.root_path)}")
        print("="*40)
        print(f"Rotors Active: {len(self.rotors)}")
        
        # Health Check
        healthy = sum(1 for r in self.rotors if r.health == "Healthy")
        print(f"System Health: {healthy}/{len(self.rotors)} ({(healthy/len(self.rotors))*100:.1f}%)")
        
        # DNA Analysis
        print(f"\n  System Soul (Average DNA):")
        print(f"{self.system_dna}")
        
        # Top 3 Rotors by Complexity (Frequency)
        sorted_rotors = sorted(self.rotors, key=lambda r: r.dna.frequency, reverse=True)
        print("\n  Highest Energy Centers (Complexity):")
        for i, r in enumerate(sorted_rotors[:3]):
            print(f"  {i+1}. {r.name} ({r.dna.frequency:.1f} Hz) - {r.diagnose()}")
            
        print("\n" + "="*40)

if __name__ == "__main__":
    # Scan Elysia
    root = "c:/Elysia"
    conductor = ProjectConductor(root)
    conductor.scan_project()
    conductor.report()
