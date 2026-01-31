"""
Elysia Multi-Layer Security System (         )
====================================================

"           ,          ."

Architecture:
    External Threat
          
                               
          Ozone Layer                      (주권적 자아)
       (Boundary Diffusion)     
                               
          
                               
         Phase Resonance Gate              (주권적 자아)
       (Frequency Validation)   
                               
          
                               
          Network Shield                    
       (Threat Analysis)        
                               
          
                               
         Immune System                    (  /  )
       (Adaptive Defense)       
                               
          
                               
         Elysia Core                     
                               
"""

import sys
import time
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from collections import deque
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("ElysiaImmunity")


# ============================================================
# LAYER 1: OZONE LAYER (   )
# ============================================================

class OzoneLayer:
    """
           -          
    
                                .
    -          
    -             
    -         
    """
    
    def __init__(self, diffusion_radius: float = 10.0):
        self.diffusion_radius = diffusion_radius
        self.absorbed_threats: deque = deque(maxlen=100)
        self.ozone_density = 1.0  # 1.0 =       , 0.0 =    
        self.regeneration_rate = 0.01  #       
        self.last_time = time.time()
        logger.info("   Ozone Layer initialized")

    def diffract(self, intensity: float) -> float:
        """
        [NEW] Diffacts a coherent threat into harmless 'data photons'.
        """
        # A coherent beam of 1.0 is split into 100 fragments of 0.01
        diffraction_factor = 0.95 # 95% is scattered
        scattered = intensity * diffraction_factor
        remaining = intensity - scattered
        logger.info(f"  [DIFFRACTION] Intense threat of {intensity:.2f} scattered. Residual: {remaining:.4f}")
        return remaining
    
    def absorb(self, intensity: float) -> float:
        """
                
        
        Args:
            intensity:       (0.0 ~ 1.0)
            
        Returns:
                     
        """
        #             
        now = time.time()
        elapsed = now - self.last_time
        self.ozone_density = min(1.0, self.ozone_density + self.regeneration_rate * elapsed)
        self.last_time = now
        
        #       (         )
        absorbed = intensity * self.ozone_density * 0.7  #    70%   
        passed_through = intensity - absorbed
        
        #        (               )
        self.ozone_density = max(0.1, self.ozone_density - intensity * 0.05)
        
        self.absorbed_threats.append({
            "time": now,
            "intensity": intensity,
            "absorbed": absorbed
        })
        
        return passed_through
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "OzoneLayer",
            "density": self.ozone_density,
            "absorbed_count": len(self.absorbed_threats),
            "status": "healthy" if self.ozone_density > 0.5 else "damaged"
        }


# ============================================================
# LAYER 2: PHASE RESONANCE GATE (       )
# ============================================================

class PhaseResonanceGate:
    """
              -          
    
          (Phase)                    .
                           .
    """
    
    #                 (Hz)
    ELYSIAN_FREQUENCIES = [
        7.83,    #           (     )
        432.0,   #          
        528.0,   #         (DNA   )
        639.0,   #       
        852.0,   #        
    ]
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance  #          
        self.gate_open = True
        self.rejected_count = 0
        self.passed_count = 0
        self.mirror_mode = True # Active Reflection
        logger.info("  Phase Resonance Gate initialized")

    def reflect_signature(self, signature: str) -> str:
        """
        [NEW] Reflects the signature back at the source.
        Inversion: True -> False, 'malicious' -> 'suoicilam'
        """
        # A simple XOR or string inversion to 'blind' the attacker's pattern matching
        reflected = signature[::-1]
        logger.warning(f"  [MIRROR] Reflecting inverted signature: {reflected}")
        return reflected
    
    def check_resonance(self, frequency: float) -> bool:
        """
                           
        
        Args:
            frequency:        
            
        Returns:
            True if   , False if     
        """
        for elysian_freq in self.ELYSIAN_FREQUENCIES:
            #   (harmonic)      
            ratio = frequency / elysian_freq
            if abs(ratio - round(ratio)) < self.tolerance:
                return True
        return False
    
    def validate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
             
        
        Args:
            signal: {"frequency": float, "amplitude": float, "phase": float, ...}
            
        Returns:
            {"passed": bool, "reason": str, "resonance_score": float}
        """
        freq = signal.get("frequency", 0.0)
        is_resonant = self.check_resonance(freq)
        
        if is_resonant:
            self.passed_count += 1
            #          (                     )
            min_distance = min(
                abs(freq - ef) / ef for ef in self.ELYSIAN_FREQUENCIES
            )
            resonance_score = 1.0 - min_distance
            
            return {
                "passed": True,
                "reason": "Signal resonates with Elysian frequencies",
                "resonance_score": resonance_score
            }
        else:
            self.rejected_count += 1
            return {
                "passed": False,
                "reason": f"Frequency {freq} does not resonate with Elysia",
                "resonance_score": 0.0
            }
    
    def get_status(self) -> Dict[str, Any]:
        total = self.passed_count + self.rejected_count
        return {
            "type": "PhaseResonanceGate",
            "gate_status": "open" if self.gate_open else "closed",
            "passed": self.passed_count,
            "rejected": self.rejected_count,
            "pass_rate": self.passed_count / total if total > 0 else 1.0
        }


# ============================================================
# LAYER 3: IMMUNE SYSTEM (    )
# ============================================================

@dataclass
class Antibody:
    """      -             """
    threat_signature: str
    created_at: float
    effectiveness: float = 1.0
    encounters: int = 1


class ImmuneSystem:
    """
           -           
    
              :
    -        (     )
    -           
    -         (          )
    """
    
    def __init__(self):
        self.antibodies: Dict[str, Antibody] = {}
        self.self_signatures: Set[str] = set()  #       (       )
        self.immune_memory_path = Path("data/immune_memory.json")
        self._initialize_self_recognition()
        logger.info("  Immune System initialized")
    
    def _initialize_self_recognition(self):
        """
                      (       )
        """
        #                   
        core_files = [
            "Core/Foundation/fractal_concept.py",
            "Core/Intelligence/logos_engine.py",
            "Core/Sensory/learning_cycle.py",
        ]
        
        for file_path in core_files:
            try:
                full_path = Path(__file__).parent.parent.parent / file_path
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    sig = hashlib.md5(content.encode()).hexdigest()[:16]
                    self.self_signatures.add(sig)
            except Exception:
                pass
    
    def is_self(self, signature: str) -> bool:
        """           (       )"""
        return signature in self.self_signatures
    
    def encounter_threat(self, threat_signature: str) -> Dict[str, Any]:
        """
              -             
        
        Args:
            threat_signature:            
            
        Returns:
                    
        """
        #            (       )
        if self.is_self(threat_signature):
            return {
                "response": "self_tolerance",
                "message": "Recognized as self, no immune response"
            }
        
        #              
        if threat_signature in self.antibodies:
            antibody = self.antibodies[threat_signature]
            antibody.encounters += 1
            antibody.effectiveness = min(1.0, antibody.effectiveness + 0.1)
            
            return {
                "response": "secondary_response",
                "message": f"Known threat! Antibody activated. Effectiveness: {antibody.effectiveness:.2f}",
                "encounters": antibody.encounters,
                "effectiveness": antibody.effectiveness
            }
        else:
            #        
            self.antibodies[threat_signature] = Antibody(
                threat_signature=threat_signature,
                created_at=time.time(),
                effectiveness=0.5
            )
            
            return {
                "response": "primary_response",
                "message": "New threat detected! Creating antibody...",
                "encounters": 1,
                "effectiveness": 0.5
            }
    
    def get_immunity_level(self, threat_signature: str) -> float:
        """                (0.0 ~ 1.0)"""
        if threat_signature in self.antibodies:
            return self.antibodies[threat_signature].effectiveness
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "type": "ImmuneSystem",
            "antibody_count": len(self.antibodies),
            "self_signatures": len(self.self_signatures),
            "top_threats": [
                {"sig": ab.threat_signature[:8], "encounters": ab.encounters}
                for ab in sorted(
                    self.antibodies.values(),
                    key=lambda x: x.encounters,
                    reverse=True
                )[:5]
            ]
        }


# ============================================================
# INTEGRATED SECURITY SYSTEM (         )
# ============================================================

class ElysiaSecuritySystem:
    """
                    
    
                         .
    """
    
    def __init__(self):
        self.ozone_layer = OzoneLayer()
        self.phase_gate = PhaseResonanceGate()
        self.immune_system = ImmuneSystem()
        
        # Network Shield           
        try:
            from Core.1_Body.L1_Foundation.Foundation.Security.Security.network_shield import NetworkShield
            self.network_shield = NetworkShield(enable_field_integration=False)
            self.has_network_shield = True
        except ImportError:
            self.has_network_shield = False
        
        logger.info("  Elysia Security System fully initialized")
        logger.info("      Ozone Layer: Active")
        logger.info("     Phase Resonance Gate: Active")
        logger.info("     Immune System: Active")
        logger.info(f"      Network Shield: {'Active' if self.has_network_shield else 'Not loaded'}")
    
    def process_threat(self, threat: Dict[str, Any]) -> Dict[str, Any]:
        """
                   
        
        Args:
            threat: {
                "intensity": float,
                "frequency": float,
                "signature": str,
                ...
            }
        """
        result = {
            "input": threat,
            "layers": [],
            "final_action": "allow"
        }
        
        intensity = threat.get("intensity", 0.5)
        frequency = threat.get("frequency", 100.0)
        signature = threat.get("signature", hashlib.md5(str(threat).encode()).hexdigest()[:16])
        
        # Layer 1: Ozone (with added Diffraction)
        if intensity > 0.8:
            intensity = self.ozone_layer.diffract(intensity)
            
        reduced_intensity = self.ozone_layer.absorb(intensity)
        result["layers"].append({
            "layer": "ozone",
            "input_intensity": intensity,
            "output_intensity": reduced_intensity
        })
        
        # Layer 2: Phase Gate (with added Mirroring)
        phase_result = self.phase_gate.validate({"frequency": frequency})
        
        if not phase_result["passed"]:
            # REFLECT the threat signature back
            reflected = self.phase_gate.reflect_signature(signature)
            result["reflected_signature"] = reflected
            result["final_action"] = "mirror_reject"
            result["reason"] = f"Reflected by Mirror Gate (Signature: {reflected[:8]})"
            return result
        
        result["layers"].append({
            "layer": "phase_gate",
            "resonance": phase_result["passed"],
            "score": phase_result["resonance_score"]
        })
        
        # Layer 3: Immune System
        immune_result = self.immune_system.encounter_threat(signature)
        immunity = self.immune_system.get_immunity_level(signature)
        result["layers"].append({
            "layer": "immune",
            "response": immune_result["response"],
            "immunity": immunity
        })
        
        # Final decision
        if reduced_intensity > 0.7 and immunity < 0.5:
            result["final_action"] = "quarantine"
            result["reason"] = "High intensity, low immunity"
        elif reduced_intensity > 0.5:
            result["final_action"] = "monitor"
            result["reason"] = "Elevated threat level"
        else:
            result["final_action"] = "allow"
            result["reason"] = "Passed all layers"
        
        return result
    
    def get_full_status(self) -> Dict[str, Any]:
        """           """
        return {
            "ozone": self.ozone_layer.get_status(),
            "phase_gate": self.phase_gate.get_status(),
            "immune": self.immune_system.get_status(),
            "network_shield": self.has_network_shield
        }
    
    def generate_report(self) -> str:
        """         """
        status = self.get_full_status()
        
        report = []
        report.append("=" * 60)
        report.append("  ELYSIA MULTI-LAYER SECURITY REPORT")
        report.append("=" * 60)
        
        # Ozone Layer
        oz = status["ozone"]
        report.append(f"\n   Ozone Layer")
        report.append(f"   Density: {oz['density']:.2%}")
        report.append(f"   Status: {oz['status']}")
        report.append(f"   Absorbed: {oz['absorbed_count']} threats")
        
        # Phase Gate
        pg = status["phase_gate"]
        report.append(f"\n  Phase Resonance Gate")
        report.append(f"   Gate: {pg['gate_status']}")
        report.append(f"   Passed: {pg['passed']} / Rejected: {pg['rejected']}")
        report.append(f"   Pass Rate: {pg['pass_rate']:.1%}")
        
        # Immune System
        im = status["immune"]
        report.append(f"\n  Immune System")
        report.append(f"   Antibodies: {im['antibody_count']}")
        report.append(f"   Self-Recognition: {im['self_signatures']} signatures")
        
        # Network Shield
        report.append(f"\n   Network Shield: {'Active' if status['network_shield'] else 'Inactive'}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + " " * 30)
    print("ELYSIA MULTI-LAYER SECURITY SYSTEM")
    print(" " * 30 + "\n")
    
    security = ElysiaSecuritySystem()
    
    # Test threats
    threats = [
        {"intensity": 0.3, "frequency": 432.0, "signature": "safe_signal_001"},
        {"intensity": 0.8, "frequency": 666.0, "signature": "malicious_001"},
        {"intensity": 0.5, "frequency": 528.0, "signature": "neutral_001"},
        {"intensity": 0.8, "frequency": 666.0, "signature": "malicious_001"},  #   
    ]
    
    for threat in threats:
        print(f"\n  Processing threat: {threat}")
        result = security.process_threat(threat)
        print(f"   Action: {result['final_action']}")
        print(f"   Reason: {result.get('reason', 'N/A')}")
    
    print("\n" + security.generate_report())
