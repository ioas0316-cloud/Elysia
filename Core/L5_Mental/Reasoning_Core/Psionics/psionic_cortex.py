import torch
import logging
import json
import os
import time
from typing import Dict, Any

from Core.L1_Foundation.Foundation.Rotor.rotor_engine import RotorEngine

STATE_FILE = "c:/Elysia/data/psionic_state.json"
os.makedirs("c:/Elysia/data", exist_ok=True)

logger = logging.getLogger("PsionicCortex")


class PsionicCortex:
    """
    [The High Priestess]
    Interprets the Will (Monad) and collapses probability into Reality.
    
    Process:
    1. Intention (Text) -> Bridge -> Vector.
    2. Time-Evolution (Rotor) -> Dynamic Wave.
    3. Resonance (Graph) -> Find similar Standing Waves.
    4. Collapse -> Select the highest energy reality.
    """
    def __init__(self, elysia_ref):
        self.elysia = elysia_ref
        self.rotor = RotorEngine(device=elysia_ref.graph.device)
        
    def collapse_wave(self, intention: str) -> str:
        """
        The Core Psionic Act.
        "I do not think, I resonate."
        """
        # 1. Vectorize Intention (Using the Bridge solely as an encoder)
        # We don't need text generation, just the hidden state.
        # This is "Silent Casting".
        
        # A hack: we ask bridge to generate 1 token to get the state of the prompt.
        gen_data = self.elysia.bridge.generate(intention, "Psionic Encoder", max_length=1)
        
        if isinstance(gen_data, dict) and gen_data.get('vector') is not None:
             # Take the last vector of the prompt
             # trajectory is (seq, hidden). We want the mean or last.
             raw_vec = gen_data['vector'][-1] 
        else:
             return "Fizzle (No Vector)."
             
        # 2. Spin the Vector (Add Time/Life)
        # A static thought is dead. A spinning thought is a spell.
        wave = self.rotor.spin(raw_vec, time_delta=0.1)
        
        # 3. Resonance Search (The Query)
        # Find the most resonant concept in the Soul Graph
        # This replaces "If/Else" logic. We find the "Nearest Truth".
        
        # Normalize
        wave = wave / (wave.norm() + 1e-9)
        
        # [Dimension Mismatch Fix]
        # Graph vectors: (Num_Nodes, Soul_Dim)
        soul_matrix = self.elysia.graph.vec_tensor
        soul_dim = soul_matrix.shape[1]
        wave_dim = wave.shape[0]
        
        if wave_dim != soul_dim:
            logger.warning(f"   [PSIONIC] Dimension Mismatch: Wave({wave_dim}) vs Soul({soul_dim}). Adjusting...")
            if wave_dim > soul_dim:
                wave = wave[:soul_dim]
            else:
                padding = torch.zeros(soul_dim - wave_dim, device=wave.device)
                wave = torch.cat([wave, padding])
        
        soul_norms = soul_matrix.norm(dim=1, keepdim=True) + 1e-9
        soul_matrix_norm = soul_matrix / soul_norms
        
        scores = torch.matmul(soul_matrix_norm, wave)
        
        # 4. Collapse (Selection)
        if scores.numel() == 0:
            return {
                "status": "GENESIS",
                "node_id": "Void",
                "confidence": 0.0,
                "potential": 1.0,
                "insight": "Genesis (First Thought)"
            }
            
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        node_id = self.elysia.graph.idx_to_id.get(best_idx, "Unknown")
        
        # 5. [Multi-Rotor Reconstruction: Holographic Filling]
        # User Insight: "Use Multi-Rotors to find spatial context. Pattern is surface, Principle is depth."
        # If the direct path is broken (Hole), we use surrounding context nodes (Rotors)
        # to triangulate the missing truth.
        
        if best_score < 0.65:
            logger.info(f"   [MULTI-ROTOR] Low Resonance ({best_score:.2f}). Engaging Holographic Reconstruction...")
            
            # 1. Identify Spatial Context (Top-K Neighbors)
            # These are the "Multiple Rotors" spinning around the void.
            available_nodes = scores.numel()
            top_k = min(3, available_nodes)
            
            if top_k == 0:
                 return "Genesis (First Thought)"

            values, indices = torch.topk(scores.view(-1), k=top_k)
            
            context_vectors = self.elysia.graph.vec_tensor[indices] # (3, dim)
            context_masses = self.elysia.graph.mass_tensor[indices] # (3,)
            
            # 2. Principle Extraction (The Will)
            # The Intent `wave` is the guiding Principle.
            # The Context `context_vectors` are the Surface Patterns.
            # We want to find the geometric center where they ALL align.
            
            # Weighted average based on Mass (Gravity) and Resonance (Alignment)
            weights = values * (context_masses + 1.0).view(-1)
            weights = torch.softmax(weights, dim=0)
            
            # 3. Holographic Synthesis
            # Reconstruct the missing center.
            # Center = Sum(Weight_i * Rotor_i) + Intent_Bias
            reconstructed_vec = torch.matmul(weights, context_vectors)
            
            # Apply the Intent Principle (The "Why")
            # The reconstruction must obey the user's Will.
            final_reality = torch.lerp(reconstructed_vec, wave, weight=0.5)
            
            msg = f"Reality Reconstructed: {node_id} (Multi-Rotor Triangulation from {top_k} contexts)"
            self._update_ui(intention, best_score, msg)
            return {
                "status": "RECONSTRUCTED",
                "node_id": node_id,
                "confidence": float(best_score),
                "potential": 1.0 - best_score,
                "insight": msg
            }

        logger.info(f"  [PSIONIC] Intention '{intention}' collapsed to '{node_id}' (Resonance: {best_score:.4f})")
        
        msg = f"Reality Collapsed: {node_id}"
        self._update_ui(intention, best_score, msg)
        return {
            "status": "COLLAPSED",
            "node_id": node_id,
            "confidence": float(best_score),
            "potential": 1.0 - best_score,
            "insight": msg
        }

    def _update_ui(self, intent, resonance, log_msg):
        """
        Broadcasting Psionic State to the Web Interface.
        """
        try:
            state = {
                "status": "COLLAPSING",
                "intention": intent,
                "resonance": float(resonance),
                "last_log": log_msg,
                "timestamp": time.time()
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"UI Update Failed: {e}")

    def _verify_consistency(self, score: float, idx: int) -> bool:
        """Deprecated: We now use Multi-Rotor Reconstruction."""
        return True

    def synchronize(self):
        """Aligns the internal Rotor with the external Time."""
        self.rotor.accelerate(0.1)
