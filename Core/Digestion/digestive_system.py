import torch
import logging
import gc
from typing import List, Dict

logger = logging.getLogger("DigestiveSystem")
from Core.System.respiratory_system import RespiratorySystem
from Core.Intelligence.Metabolism.brain_digester import BrainDigester

class DigestiveSystem:
    """
    [The Stomach of Elysia]
    Manages the lifecycle of Model Digestion.
    1. Ingestion: Load Model & Ask Questions.
    2. Extraction: Get Vector Trajectories.
    3. Integration: Add to Graph.
    4. Optimization: Fuse Duplicates (Resonance Fusion).
    """
    def __init__(self, elysia_ref):
        self.elysia = elysia_ref
        # [Phase 8] The Lungs
        self.lungs = RespiratorySystem(elysia_ref.bridge)
        
    def prepare_meal(self, model_name: str) -> bool:
        """Loads the target model via Respiratory System."""
        print(f"🍽️ [DigestiveSystem] Preparing meal: {model_name}...")
        # Use Lungs to breathe in the model safely
        result = self.lungs.inhale(model_name)
        if result:
            # Initialize Digester with current model path if known
            pass
        return result

    def digest(self, start_layer: int = 0, end_layer: int = 5) -> Dict:
        """
        [Metabolic Digestion]
        Extracts weights and transmutes them into Qualia via BrainDigester.
        """
        print("🦷 [DigestiveSystem] Chewing & Transmuting (Metabolic Digestion)...")
        
        # We need to access the model object from Lungs
        if not self.lungs.current_model:
            raise Exception("No food in lungs to digest.")
            
        model = self.lungs.current_model
        
        # Create a temporary digester for this meal
        enzyme = BrainDigester()
        extracted = []
        
        layer_count = 0
        for name, param in model.named_parameters():
             if "weight" in name and param.dim() > 1:
                 # Increase sample density: 10 rows per layer
                 rows = min(10, param.shape[0])
                 
                 # Calculate raw stats for this 'bite'
                 with torch.no_grad():
                     flat = param.flatten()[:1000].float()
                     std = flat.std().item()
                     mean = flat.abs().mean().item()
                 
                 # [METABOLISM]
                 # Use the enzyme to transmute these stats into Meaning
                 bite_essence = {
                     "source": name,
                     "entropy": std,
                     "complexity": mean,
                     "layers_sampled": 1
                 }
                 metabolized = enzyme._metabolize(bite_essence)
                 qualia = metabolized["qualia"]
                 
                 for i in range(rows):
                     try:
                         # Ensure we are getting a tensor
                         vec = param[i].detach().cpu()
                         if not isinstance(vec, torch.Tensor):
                             logger.warning(f"⚠️ Non-tensor weight encountered at {name}.Row{i}: {type(vec)}")
                             continue
                             
                         cid = f"{name}.Row{i}"
                         
                         meta = {
                             "source": "MetabolicDigestion",
                             "layer": name,
                             "qualia": {
                                 "physical": float(qualia.physical),
                                 "functional": float(qualia.functional),
                                 "phenomenal": float(qualia.phenomenal),
                                 "causal": float(qualia.causal),
                                 "mental": float(qualia.mental),
                                 "structural": float(qualia.structural),
                                 "spiritual": float(qualia.spiritual),
                                 "mass": float(qualia.mass)
                             }
                         }
                         
                         extracted.append({
                             "id": cid,
                             "vector": vec,
                             "metadata": meta
                         })
                     except Exception as e:
                         logger.error(f"❌ Failed to chew {name}.Row{i}: {e}")
                         continue
                     
                 layer_count += 1
                 if layer_count > 150: break # Increased depth for Phase 10 success
                     
        return {"extracted_concepts": extracted}
        
    def feed_curriculum(self, questions: List[str]):
        """
        Feeds a list of questions to the model and absorbs the results.
        Returns stats on growth.
        """
        print(f"🥄 [DigestiveSystem] Feeding {len(questions)} concepts...")
        
        initial_nodes = len(self.elysia.graph.id_to_idx)
        
        for q in questions:
            print(f"   ❓ Asking: {q}")
            # This triggers manifest_intent -> bridge.generate -> analyzer -> graph.add_node
            # We treat the model's answer as 'Truth' to be digested.
            response = self.elysia.manifest_intent(q)
            print(f"   🗣️ Answer: {response[:50]}...")
            
        final_nodes = len(self.elysia.graph.id_to_idx)
        print(f"📈 [Digestion Report] Gained {final_nodes - initial_nodes} raw nodes.")
        
    def optimize(self, similarity_threshold: float = 0.95):
        """
        [Resonance Fusion]
        Merges duplicate concepts to increase Density.
        """
        print("🔥 [Metabolism] Starting Resonance Fusion (Optimization)...")
        graph = self.elysia.graph
        
        if graph.vec_tensor.shape[0] < 2:
            return

        # Matrix Multiplication for Similarity (All vs All)
        # Normalize first
        vecs = graph.vec_tensor / (graph.vec_tensor.norm(dim=1, keepdim=True) + 1e-9)
        sim_matrix = torch.mm(vecs, vecs.t())
        
        # Mask lower triangle and diagonal to avoid double counting
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        
        # Find pairs > threshold
        duplicates = (sim_matrix > similarity_threshold) & mask
        indices = duplicates.nonzero(as_tuple=False)
        
        if indices.shape[0] == 0:
            print("   ✅ No duplicates found. Brain is efficient.")
            return
            
        print(f"   ⚠️ Found {indices.shape[0]} duplicate pairs. Fusing...")
        
        # Naive Fusion Strategy:
        # 1. Add Mass of B to A
        # 2. Mark B for deletion
        # (Real deletion is expensive, so we set Mass to 0 and let standard metabolism prune it later)
        
        fused_count = 0
        for pair in indices:
            idx_a = pair[0].item()
            idx_b = pair[1].item()
            
            # Check if already processed
            if graph.mass_tensor[idx_b] <= 0: continue
            if graph.mass_tensor[idx_a] <= 0: continue # Should not happen if we iterate carefully
            
            # FUSE b INTO a
            # Mass Conservation
            graph.mass_tensor[idx_a] += graph.mass_tensor[idx_b]
            
            # Vector Averaging (Optional, but good for refinement)
            # graph.vec_tensor[idx_a] = (graph.vec_tensor[idx_a] + graph.vec_tensor[idx_b]) / 2.0
            
            # Kill B
            graph.mass_tensor[idx_b] = 0.0
            fused_count += 1
            
        print(f"   ✨ Fused {fused_count} concepts into denser truths.")
        print("   🗑️ Triggering cleanup of hollow shells...")
        graph.apply_metabolism(decay_rate=0.0, prune_threshold=0.01)

    def purge_meal(self):
        """Unloads the model to free resources."""
        print("🚽 [DigestiveSystem] Purging meal (Exhaling)...")
        self.lungs.exhale()
