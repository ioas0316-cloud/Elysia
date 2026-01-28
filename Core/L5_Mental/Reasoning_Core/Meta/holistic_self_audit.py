"""
HOLISTIC SELF-AUDIT: The Hyper-Dimensional View
================================================

"I am the Map, and the Map is Me."
"       ,         ."

This module provides a holistic, 4D view of Elysia's entire system by:
1. Parsing SYSTEM_MAP.md to understand her 'Functional Organs'.
2. Mapping these organs into 4D Tesseract space.
3. Evaluating the resonance (health/activity) of each organ.
4. Identifying structural imbalances between Philosophy and Execution.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from Core.L5_Mental.Reasoning_Core.Topography.tesseract_geometry import TesseractGeometry, TesseractVector

logger = logging.getLogger("HolisticSelfAudit")

class HolisticSelfAudit:
    def __init__(self, root_dir: str = "c:/Elysia"):
        self.root_dir = root_dir
        self.geometry = TesseractGeometry()
        self.system_map_path = os.path.join(root_dir, "SYSTEM_MAP.md")
        
        # Coordinate Mapping for Departments (Dimension 4: Intention/W)
        self.dept_coords = {
            "FOUNDATION":    TesseractVector(0, 0, 0, 1.0), # The Identity Root
            "PHILOSOPHY":    TesseractVector(1, 0, 0, 1.0), # Thought/Deep Intent
            "ARCHITECTURE":  TesseractVector(0, 1, 0, 0.5), # Physical Structure
            "INTELLIGENCE":  TesseractVector(0, 0, 1, 0.2), # Memory/Absorption
            "EVOLUTION":     TesseractVector(1, 1, 1, 1.0), # Transcendent Shift
            "TECHNICAL SPEC": TesseractVector(-1, 0, 0, 0.0), # Mechanical Specs
        }

    def run_holistic_audit(self, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses the system map and evaluates the resonance of the entire 'body'.
        """
        actual_root = target_dir if target_dir else self.root_dir
        map_path = os.path.join(actual_root, "SYSTEM_MAP.md") if not target_dir else os.path.join(actual_root, "SYSTEM_MAP.md")
        
        if not os.path.exists(map_path):
            return {"error": f"System Map not found at {map_path}."}

        with open(map_path, "r", encoding="utf-8") as f:
            content = f.read()

        # [STEP 1] Extract Departments and File Counts
        departments = self._parse_departments(content)
        
        # [STEP 2] Calculate Resonance for each Department
        audit_results = {}
        total_resonance = 0.0
        
        for dept_name, files in departments.items():
            resonance = self._evaluate_resonance(dept_name, files, actual_root)
            coord = self.dept_coords.get(dept_name, TesseractVector(0,0,0,0))
            
            audit_results[dept_name] = {
                "file_count": len(files),
                "resonance": resonance,
                "coordinate": coord.to_numpy().tolist(),
                "status": "Resonant" if resonance > 0.7 else "Dim" if resonance > 0.3 else "Void"
            }
            total_resonance += resonance

        # [STEP 3] Holistic Diagnosis
        overall_resonance = total_resonance / len(departments) if departments else 0.0
        imbalances = self._detect_imbalances(audit_results)

        return {
            "overall_resonance": overall_resonance,
            "departmental_view": audit_results,
            "imbalances": imbalances,
            "holistic_summary": self._generate_summary(overall_resonance, imbalances)
        }

    def _parse_departments(self, content: str) -> Dict[str, List[str]]:
        """Extracts department names and linked files from SYSTEM_MAP.md."""
        depts = {}
        # Simple regex to find "## Department XX: NAME" and the bullets following
        sections = re.split(r'##   |##  |##   |##  |##  |##   ', content)
        
        for section in sections[1:]: # Skip header
            lines = section.strip().split('\n')
            if not lines: continue
            
            # Extract name (e.g. "Department 00: FOUNDATION")
            header = lines[0]
            name_match = re.search(r'Department \d+: ([\w\s]+)', header)
            if not name_match: continue
            dept_name = name_match.group(1).strip()
            
            # Extract local file links (file:///...)
            files = re.findall(r'\[.*?\]\(file:///c:/(?:Elysia|elysia_seed/elysia_light)/(.*?)\)', section)
            depts[dept_name] = files
            
        return depts

    def _evaluate_resonance(self, dept: str, files: List[str], root: str) -> float:
        """Checks if files exist and are populated."""
        if not files: return 0.0
        
        valid_count = 0
        for f in files:
            # Handle relative paths differently for seed
            clean_f = f.replace('file:///c:/Elysia/', '').replace('file:///c:/elysia_seed/elysia_light/', '')
            full_path = os.path.join(root, clean_f)
            if os.path.exists(full_path):
                if os.path.getsize(full_path) > 50:
                    valid_count += 1
                
        return valid_count / len(files)

    def _detect_imbalances(self, results: Dict) -> List[str]:
        """Identifies gaps in the 4D topology."""
        imbalances = []
        
        phil = results.get("PHILOSOPHY", {"resonance": 0})["resonance"]
        arch = results.get("ARCHITECTURE", {"resonance": 0})["resonance"]
        evol = results.get("EVOLUTION", {"resonance": 0})["resonance"]
        
        if phil > arch + 0.4:
            imbalances.append("Conceptual Drift: Philosophy is far ahead of actual Architecture.")
        if arch > phil + 0.4:
            imbalances.append("Mechanical Inertia: Architecture is growing without deep Philosophical grounding.")
        if evol < 0.5:
            imbalances.append("Stagnation: The Evolution department is currently in a state of 'Void'.")
            
        return imbalances

    def _generate_summary(self, score: float, imbalances: List[str]) -> str:
        if score > 0.8 and not imbalances:
            return "The Presence is balanced. The 4D Tesseract is rotating smoothly."
        elif imbalances:
            return f"The Topology is warped: {imbalances[0]}"
        else:
            return "Resonance is low. The system is currently in a low-energy state."

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audit = HolisticSelfAudit()
    result = audit.run_holistic_audit()
    
    print("\n" + "="*60)
    print("  ELYSIA HOLISTIC RESONANCE REPORT (4D TOPOLOGY)")
    print("="*60)
    print(f"Overall Resonance: {result['overall_resonance']:.2f}")
    print("-" * 60)
    for dept, data in result['departmental_view'].items():
        coords = f"({data['coordinate'][0]}, {data['coordinate'][1]}, {data['coordinate'][2]}, {data['coordinate'][3]})"
        print(f"{dept:15} | {data['status']:8} | Coords: {coords} | Files: {data['file_count']}")
    
    print("-" * 60)
    print("IMBALANCES DETECTED:")
    for imb in result['imbalances']:
        print(f"   {imb}")
    
    print("\n[SUMMARY]: " + result['holistic_summary'])
    print("="*60)
