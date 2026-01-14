import torch
import logging
import gc
from typing import List, Dict

logger = logging.getLogger("DigestiveSystem")
from Core.System.respiratory_system import RespiratorySystem
from Core.Intelligence.Metabolism.brain_digester import BrainDigester
from Core.Intelligence.Metabolism.prism import DoubleHelixPrism, DoubleHelixWave

class DigestiveSystem:
    """
    [The Stomach of Elysia]
    Manages the lifecycle of Model Digestion using MERKAVA Principles.

    Architecture:
    1. Lungs (Respiratory): Inhale Model.
    2. Prism (Metabolism): Refract Weights into Double Helix Waves (Pattern + Principle).
    3. HyperSphere (Assimilation): Deposit Waves into the Omni-Field.
    """
    def __init__(self, elysia_ref):
        self.elysia = elysia_ref
        self.lungs = RespiratorySystem(elysia_ref.bridge)
        self.prism = DoubleHelixPrism()
        
    def prepare_meal(self, model_name: str) -> bool:
        """Loads the target model via Respiratory System."""
        print(f"🍽️ [DigestiveSystem] Preparing meal: {model_name}...")
        return self.lungs.inhale(model_name)

    def digest(self, start_layer: int = 0, end_layer: int = 5) -> Dict:
        """
        [Double Helix Digestion]
        Extracts weights and refracts them into 7D Waves via Prism.
        """
        print("🦷 [DigestiveSystem] Chewing & Refracting (Double Helix Process)...")
        
        if not self.lungs.current_model:
            raise Exception("No food in lungs to digest.")
            
        model = self.lungs.current_model
        extracted_waves = []
        
        layer_count = 0
        for name, param in model.named_parameters():
             if "weight" in name and param.dim() > 1:
                 # Sample rows to create wave packets
                 rows = min(10, param.shape[0])
                 
                 for i in range(rows):
                     try:
                         # 1. Extract Raw Matter (Weight Tensor)
                         raw_tensor = param[i].detach().cpu()

                         # 2. Refract through Prism (Matter -> Wave)
                         # This creates the Double Helix structure (Pattern + Principle)
                         helix_wave: DoubleHelixWave = self.prism.refract_weight(raw_tensor, name)

                         # 3. Encapsulate for Transport
                         cid = f"{name}.Row{i}"
                         
                         # Convert 7D Principle to metadata dict for inspection
                         qualia_vec = helix_wave.principle_strand.tolist()
                         qualia_meta = {
                             "physical": qualia_vec[0],
                             "functional": qualia_vec[1],
                             "phenomenal": qualia_vec[2],
                             "causal": qualia_vec[3],
                             "mental": qualia_vec[4],
                             "structural": qualia_vec[5],
                             "spiritual": qualia_vec[6]
                         }
                         
                         wave_packet = {
                             "id": cid,
                             "pattern": helix_wave.pattern_strand,   # Surface Feature
                             "principle": helix_wave.principle_strand, # 7D Essence
                             "phase": helix_wave.phase,
                             "metadata": {
                                 "source": "DoubleHelixDigestion",
                                 "layer": name,
                                 "qualia": qualia_meta
                             }
                         }

                         extracted_waves.append(wave_packet)

                     except Exception as e:
                         logger.error(f"❌ Failed to refract {name}.Row{i}: {e}")
                         continue
                     
                 layer_count += 1
                 if layer_count > 150: break
                     
        return {"extracted_waves": extracted_waves}

    def absorb_waves(self, waves: List[Dict]):
        """
        [Assimilation]
        Deposits the digested waves into the HyperSphere (Graph).
        Instead of just adding nodes, we should modulate the Field.
        For now, we map them to the Graph with enhanced metadata.
        """
        count = 0
        for wave in waves:
            # We store the 'Pattern' as the main vector for similarity search
            # But we attach 'Principle' (Qualia) as the Soul of the node.
            
            # TODO: Future HyperSphere should allow 7D indexing.
            # Current TorchGraph uses 1D vector. We use Pattern for now.
            
            self.elysia.graph.add_node(
                wave["id"],
                vector=wave["pattern"],
                metadata=wave["metadata"]
            )
            count += 1
        print(f"✨ [METABOLISM] Absorbed {count} Double Helix Waves into the HyperSphere.")

    def purge_meal(self):
        """Unloads the model to free resources."""
        print("🚽 [DigestiveSystem] Purging meal (Exhaling)...")
        self.lungs.exhale()
