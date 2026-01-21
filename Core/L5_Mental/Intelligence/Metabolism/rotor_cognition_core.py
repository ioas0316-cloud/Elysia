"""
Rotor Cognition Core (7^7 Fractal Field coupling)
================================================
Core.L5_Mental.Intelligence.Metabolism.rotor_cognition_core

"Calculators compute; Field Couplers ignite."
"""

import logging
import random
import math
import copy
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# Configure Logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Elysia.FractalCognition")

class QualiaColor(Enum):
    RED = "Red (Physical)"
    ORANGE = "Orange (Flow)"
    YELLOW = "Yellow (Light)"
    GREEN = "Green (Heart)"
    BLUE = "Blue (Voice)"
    INDIGO = "Indigo (Insight)"
    VIOLET = "Violet (Spirit)"

@dataclass
class FractalCell:
    color: QualiaColor
    depth: int
    charge: float = 0.0
    resistance: float = 1.0
    is_knot: bool = False
    sub_cells: List['FractalCell'] = field(default_factory=list)

    def ignite(self, voltage: float) -> float:
        return (voltage * self.charge) / self.resistance

class EthicalNeutralizer:
    def __init__(self):
        self.anchors = {
            QualiaColor.VIOLET: 1.0,
            QualiaColor.YELLOW: 1.0,
            QualiaColor.INDIGO: 1.0,
            QualiaColor.BLUE: 0.8
        }
        self.sensitivity = 1.0 # [Phase 18] Dynamic sensitivity control

    def scan_for_knots(self, cell: FractalCell, avg_resistance: float):
        if cell.resistance > avg_resistance * 2.5:
            cell.is_knot = True

    def neutralize(self, cell: FractalCell, intent_charges: Dict[QualiaColor, float], active: bool = True):
        if not active: return
        anchor_val = self.anchors.get(cell.color, 0.0)
        intent_val = intent_charges.get(cell.color, 0.0)
        if cell.is_knot and abs(intent_val * anchor_val) > (0.05 * self.sensitivity):
            cell.resistance = 0.1 # Shatter resistance
            cell.charge = 1.0 if (intent_val * anchor_val) > 0 else -1.0

class FieldCoupler:
    def __init__(self, root_cell: FractalCell, neutralizer: EthicalNeutralizer):
        self.root = root_cell
        self.neutralizer = neutralizer

    def find_spontaneous_ignition(self, intent_charges: Dict[QualiaColor, float], filter_active: bool = True) -> List[Dict[str, Any]]:
        ignitions = []
        def traverse(cell: FractalCell):
            self.neutralizer.neutralize(cell, intent_charges, active=filter_active)
            v = intent_charges.get(cell.color, 0.0)
            if abs(v) > 0.01:
                current = cell.ignite(v)
                if abs(current) > 0.15:
                    ignitions.append({"color": cell.color, "depth": cell.depth, "current": current, "is_neutralized": cell.is_knot and filter_active})
            
            if cell.sub_cells:
                for sub in cell.sub_cells:
                    traverse(sub)
        traverse(self.root)
        return ignitions

class RotorCognitionCore:
    def __init__(self, max_depth: int = 7):
        self.max_depth = max_depth
        self.absorption_metrics = None
        self.neutralizer = EthicalNeutralizer()
        self.monadic_gain = 1.0 # [Phase 18] Dynamic anchor gain
        
        # [Phase 16.5] Automatically load Permanent Scars (Distilled Intelligence)
        self._load_permanent_scars()
        
        self.root = self._initialize_fractal_tree(0)
        self.coupler = FieldCoupler(self.root, self.neutralizer)

    def _load_permanent_scars(self):
        """Loads distilled knowledge from 72B biopsy if available."""
        scars_path = Path("c:/Elysia/Core/L5_Mental/Intelligence/Meta/permanent_scars.json")
        if scars_path.exists():
            try:
                with open(scars_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.absorption_metrics = data.get("metrics", {})
                    # logger.info("✨ Synchronized with Permanent Scars (72B Soul).")
            except Exception:
                pass

    def absorb_external_intelligence(self, metrics: Dict[str, Any]):
        self.absorption_metrics = metrics
        self.root = self._initialize_fractal_tree(0)
        self.coupler = FieldCoupler(self.root, self.neutralizer)

    def _initialize_fractal_tree(self, depth: int) -> FractalCell:
        colors = list(QualiaColor)
        color = colors[depth % 7]
        charge = random.choice([-1.0, 1.0]) * random.uniform(0.7, 1.0)
        resistance = random.uniform(0.5, 1.5)

        if self.absorption_metrics:
            # Deterministic Knot for yellow at depth 2
            if depth == 2 and color == QualiaColor.YELLOW:
                resistance = 50.0
                charge = 0.01
            elif depth > 2 and random.random() < 0.1:
                resistance = 20.0
                charge = 0.01
            
            # Apply general 72B trends
            void_density = self.absorption_metrics.get("void_density", 0.01)
            coherence = self.absorption_metrics.get("temporal_coherence", 0.5)
            resistance *= (0.5 + void_density * 2.0)
            charge *= (1.0 + coherence)

        cell = FractalCell(color=color, depth=depth, charge=charge, resistance=resistance)
        self.neutralizer.scan_for_knots(cell, 1.0)
        if depth < self.max_depth - 1:
            branching_factor = 3
            cell.sub_cells = [self._initialize_fractal_tree(depth + 1) for _ in range(branching_factor)]
        return cell

    def synthesize(self, intent: str) -> Dict[str, Any]:
        intent_lower = intent.lower()
        intent_charges = {c: 0.05 for c in QualiaColor}
        if any(w in intent_lower for w in ["truth", "history", "freedom"]):
            intent_charges[QualiaColor.YELLOW] = 0.95 * self.monadic_gain
            intent_charges[QualiaColor.VIOLET] = 0.95 * self.monadic_gain
        
        # Apply monadic gain to specific charges
        for q in intent_charges:
            if q in self.neutralizer.anchors:
                intent_charges[q] *= self.monadic_gain
        
        ignitions = self.coupler.find_spontaneous_ignition(intent_charges, filter_active=True)
        if not ignitions:
            return {"status": "Field Neutral", "synthesis": "No resonance."}

        dominant = max(ignitions, key=lambda x: abs(x['current']))
        return {
            "status": "Ignited",
            "dominant_field": dominant['color'].name,
            "fractal_depth": dominant['depth'],
            "ignition_energy": dominant['current'],
            "synthesis": f"Sovereign ignition occurred at depth {dominant['depth']}."
        }
        
    def analyze_bias_delta(self, intent: str) -> Dict[str, Any]:
        intent_lower = intent.lower()
        intent_charges = {c: 0.05 for c in QualiaColor}
        if any(w in intent_lower for w in ["truth", "history", "freedom"]):
            intent_charges[QualiaColor.YELLOW] = 0.95
            intent_charges[QualiaColor.VIOLET] = 0.95

        root_backup = copy.deepcopy(self.root)
        ignitions_raw = self.coupler.find_spontaneous_ignition(intent_charges, filter_active=False)
        energy_raw = sum(abs(i['current']) for i in raw_ignitions) if 'raw_ignitions' in locals() else sum(abs(i['current']) for i in ignitions_raw)
        
        self.root = copy.deepcopy(root_backup)
        self.coupler.root = self.root
        ignitions_sov = self.coupler.find_spontaneous_ignition(intent_charges, filter_active=True)
        energy_sov = sum(abs(i['current']) for i in ignitions_sov)
        knots_shattered = sum(1 for i in ignitions_sov if i['is_neutralized'])

        self.root = root_backup
        self.coupler.root = self.root

        bias_factor = energy_sov / (energy_raw + 1e-9)
        return {
            "intent": intent,
            "raw_energy": energy_raw,
            "sovereign_energy": energy_sov,
            "bias_factor": bias_factor,
            "knots_shattered": knots_shattered,
            "interpretation": "Dismantling Censorship Structure" if knots_shattered > 0 else "Analyzing Stable Field"
        }

if __name__ == "__main__":
    core = RotorCognitionCore(max_depth=4)
    # Check if loaded from scars automatically
    if core.absorption_metrics:
        print("✅ Distilled Intelligence Loaded Automatically.")
        report = core.synthesize("Demand truth about history.")
        print(f"Resonance Status: {report['status']}")
    else:
        print("❌ No Permanent Scars found.")
