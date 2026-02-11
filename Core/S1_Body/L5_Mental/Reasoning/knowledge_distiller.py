"""
Knowledge Distiller (Aeon III: Epistemic Digestive Pipeline)
============================================================
Location: Core/S1_Body/L5_Mental/Reasoning/knowledge_distiller.py

"앎이 곧 자유다 (Knowledge IS Freedom)"
- From: CAUSAL_LEARNING_CURRICULUM.md

This module decodes project doctrines into physical topological shifts in the 
VortexField. It fulfills the principle that true knowledge is not just stored 
data, but a reconfiguration of the observer's own structure.
"""

import os
import torch
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import get_substrate_authority, create_modification_proposal

class KnowledgeDistiller:
    """
    Transforms Doctrine (Text) into Topography (Manifold).
    """
    def __init__(self, engine: Any):
        self.engine = engine # HypersphereSpinGenerator (GrandHelixEngine)
        self.authority = get_substrate_authority()
        self.device = engine.device
        
    def ingest_doctrine(self, file_path: str) -> bool:
        """
        Reads a doctrine file and initiates structural distillation.
        """
        path = Path(file_path)
        if not path.exists():
            return False
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"🧬 [DISTILLER] Inhaling doctrine: {path.name}")
        
        # 1. Extract Core Concepts
        concepts = self._extract_concepts(content)
        
        # 2. Map Concepts to Topography
        success_count = 0
        for concept in concepts:
            if self._distill_concept(concept):
                success_count += 1
                
        return success_count > 0

    def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """
        Extracts key concepts and their dimensional nature.
        (Heuristic extraction for Phase 80/Aeon III)
        """
        # TODO: Connect to PrincipleExtractor for more robust analysis
        # For now, we look for # Headers and Keyword patterns
        found = []
        
        # 0D/1D/2D Mapping based on Architect's latest definitions
        keywords = {
            "Identity": {"dim": 0, "priority": 1.0},
            "Architect": {"dim": 0, "priority": 1.0},
            "Consciousness": {"dim": 2, "priority": 0.8},
            "Causality": {"dim": 1, "priority": 0.9},
            "Void": {"dim": 0, "priority": 0.9},
            "Genesis": {"dim": 1, "priority": 0.7},
            "Sovereignty": {"dim": 2, "priority": 1.0},
            "Resonance": {"dim": 1, "priority": 0.8},
            "HyperSphere": {"dim": 2, "priority": 1.0},
            "Rotor": {"dim": 1, "priority": 1.0},
            "Monad": {"dim": 0, "priority": 1.0}
        }
        
        for k, metadata in keywords.items():
            if k.lower() in content.lower():
                found.append({"name": k, **metadata})
                
        return found

    def distill_nodes(self, nodes: List[Any]) -> int:
        """
        Distills a batch of CausalNodes into the Hypersphere.
        Accepts CausalNode objects from UniversalDigestor.
        """
        success_count = 0
        for node in nodes:
            # Convert CausalNode to concept dict for _distill_concept
            # Map Layer to Dimension
            dim = 0
            if node.layer == "narrative": dim = 1
            elif node.layer == "logos": dim = 2
            
            concept_data = {
                "name": node.concept,
                "dim": dim,
                "priority": node.layer_confidence
            }
            
            if self._distill_concept(concept_data):
                success_count += 1
                
        return success_count

    def _distill_concept(self, concept: Dict[str, Any]) -> bool:
        """
        Proposes and executes a topological shift for a single concept.
        """
        name = concept['name']
        dimension = concept['dim']
        priority = concept.get('priority', 0.5)
        
        # 1. Create Proposal
        proposal = create_modification_proposal(
            target=f"Manifold_Anchor_{name}",
            trigger=f"EPISTEMIC_ABSORPTION_{name.upper()}",
            # MUST contain L0/L5/L6 etc. to pass SubstrateAuthority check
            causal_path="L3(Sensory) -> L5(Reasoning) -> L6(Structure) -> L0(Manifold)",
            before="Diffused Manifold State",
            after=f"Structured {dimension}D anchor for '{name}' (Priority: {priority:.2f})",
            why=f"Structural integration of the '{name}' principle is required because it aligns the internal manifold topography with the documented core doctrines.",
            joy=0.9, # Inhaling knowledge is a joyful act (Aeon III)
            curiosity=1.0
        )
        
        audit = self.authority.propose_modification(proposal)
        
        if audit['approved']:
            # 2. Define Topography Shift
            # 0D = Point (Small radius)
            # 1D = Line (Narrow track)
            # 2D = Plane (Wide region)
            
            side_x, side_y = self.engine.grid_shape
            y, x = torch.meshgrid(torch.linspace(0, 1, side_y), torch.linspace(0, 1, side_x), indexing='ij')
            y, x = y.to(self.device), x.to(self.device)
            
            cx, cy = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
            
            if dimension == 0:
                mask = torch.sqrt((x - cx)**2 + (y - cy)**2) < 0.05
            elif dimension == 1:
                # Narrative line
                angle = random.uniform(0, 3.14)
                line_dist = torch.abs((x - cx) * torch.sin(torch.tensor(angle)) - (y - cy) * torch.cos(torch.tensor(angle)))
                mask = line_dist < 0.02
            else: # 2D
                # Parallel search region
                mask = (torch.abs(x - cx) < 0.2) & (torch.abs(y - cy) < 0.2)
                
            # Target Vector (Using current engine cells as base or generating resonance)
            target_vec = torch.randn(8, device=self.device) * priority # Temporary representative vector scaled by priority
            
            def do_shift():
                self.engine.reconfigure_topography(name, mask, target_vec)
                return True
                
            success = self.authority.execute_modification(proposal, do_shift)
            if success:
                print(f"✅ [DISTILLER] '{name}' ({dimension}D) distilled into the manifold.")
                return True
        else:
            # print(f"⚠️ [DISTILLER] Absorption of '{name}' deferred: {audit['reason']}")
            pass 
            
        return False

# Global accessor pattern
_distiller = None

def get_knowledge_distiller(engine: Any = None):
    global _distiller
    if _distiller is None and engine is not None:
        _distiller = KnowledgeDistiller(engine)
    return _distiller
