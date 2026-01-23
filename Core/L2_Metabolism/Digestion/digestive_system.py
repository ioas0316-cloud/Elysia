import torch
import logging
import gc
from typing import List, Dict

logger = logging.getLogger("DigestiveSystem")
from Core.L6_Structure.System.respiratory_system import RespiratorySystem
from Core.L5_Mental.Intelligence.Metabolism.brain_digester import BrainDigester
from Core.L5_Mental.Intelligence.Metabolism.prism import DoubleHelixPrism, DoubleHelixWave
from Core.L1_Foundation.Foundation.Memory.fractal_causality import FractalCausalityEngine, CausalRole

class DigestiveSystem:
    """
    [The Stomach of Elysia]
    Manages the lifecycle of Model Digestion using MERKAVA Principles.

    Architecture:
    1. Lungs (Respiratory): Inhale Model.
    2. Prism (Metabolism): Refract Weights into Double Helix Waves (Pattern + Principle).
    3. HyperSphere (Assimilation): Deposit Waves into the Omni-Field.
    4. Feedback (Reinforcement): Update FreeWillEngine needs.
    """
    def __init__(self, elysia_ref):
        self.elysia = elysia_ref
        self.lungs = RespiratorySystem(elysia_ref.bridge)
        self.prism = DoubleHelixPrism()
        self.causality = FractalCausalityEngine("DigestiveCausality")
        
    def eat(self, model_name: str) -> str:
        """
        [PHASE 12] Convenience wrapper for the full digestion cycle.
        """
        try:
            if self.prepare_meal(model_name):
                result = self.digest()
                self.absorb_waves(result["extracted_waves"])

                # [Phase 2: True Digestion Feedback]
                # Consuming knowledge satisfies the Meaning Need.
                if self.elysia and hasattr(self.elysia, 'will_engine'):
                    self.elysia.will_engine.satisfy("Meaning", 20.0)
                    print("  [REWARD] Hunger for Meaning satisfied by digestion.")

                self.purge_meal()
                return f"Successfully digested and assimilated {model_name}."
            else:
                return f"Failed to prepare {model_name}. Check logs."
        except Exception as e:
            logger.error(f"Digestion failed: {e}")
            return f"Error during digestion: {e}"
        
    def prepare_meal(self, model_name: str) -> bool:
        """Loads the target model via Respiratory System."""
        print(f"   [DigestiveSystem] Preparing meal: {model_name}...")
        return self.lungs.inhale(model_name)

    def active_probe(self, param: torch.Tensor, name: str) -> Dict:
        """
        [MERKAVA Phase 4-A: Active Probing]
        Stimulates the weights with dummy input and traces activation pattern.
        """
        # 1. Cause: Origin/Intent of this weight
        cause_desc = f"Origin of {name}"
        
        # 2. Structure: Geometry of the weight
        structure_desc = f"Geometric layout: {param.shape}"
        
        # 3. Function: Stimulate and observe activation
        # Create a unit stimulus (Normal distribution)
        stimulus = torch.randn(1, param.shape[-1])
        with torch.no_grad():
            # Simulated forward pass for this layer slice
            response = torch.matmul(stimulus, param[0:1].T)
            sparsity = (response < 0.01).float().mean().item()
            entropy = response.std().item()
        
        function_desc = f"Activation Response: S={sparsity:.2f}, E={entropy:.2f}"
        
        # 4. Reality: The final node classification
        reality_desc = f"Manifested Node: {name}"
        
        # Build Fractal Chain
        # chain = self.causality.create_chain(cause_desc, "Processing...", reality_desc)
        
        return {
            "cause": cause_desc,
            "structure": structure_desc,
            "function": function_desc,
            "reality": reality_desc
        }

    def digest(self, start_layer: int = 0, end_layer: int = 5) -> Dict:
        """
        [Double Helix Digestion]
        Extracts weights and refracts them into 7D Waves via Prism.
        Includes Active Probing logic.
        """
        print("  [DigestiveSystem] Dynamic Digestion (Double Helix + Active Probing)...")
        
        if not self.lungs.current_model:
            raise Exception("No food in lungs to digest.")
            
        model = self.lungs.current_model
        extracted_waves = []
        
        layer_count = 0
        for name, param in model.named_parameters():
             if "weight" in name and param.dim() > 1:
                 # Sample rows
                 rows = min(5, param.shape[0])
                 
                 # Active Probe for the entire layer logic
                 probe_data = self.active_probe(param, name)
                 
                 for i in range(rows):
                     try:
                         # 1. Extract Raw Matter
                         raw_tensor = param[i].detach().cpu()

                         # 2. Refract through Prism
                         helix_wave: DoubleHelixWave = self.prism.refract_weight(raw_tensor, name)

                         # 3. Encapsulate for Transport
                         cid = f"{name}.Row{i}"
                         
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
                             "pattern": helix_wave.pattern_strand,   
                             "principle": helix_wave.principle_strand, 
                             "phase": helix_wave.phase,
                             "metadata": {
                                 "source": "DoubleHelixDigestion",
                                 "layer": name,
                                 "qualia": qualia_meta,
                                 "causal_chain": probe_data
                             }
                         }

                         extracted_waves.append(wave_packet)

                     except Exception as e:
                         logger.error(f"  Failed to refract {name}.Row{i}: {e}")
                         continue
                     
                 layer_count += 1
                 if layer_count > 50: break
                     
        return {"extracted_waves": extracted_waves}

    def absorb_waves(self, waves: List[Dict]):
        """
        [Assimilation]
        Deposits the digested waves into the HyperSphere (Graph).
        """
        count = 0
        for wave in waves:
            self.elysia.graph.add_node(
                wave["id"],
                vector=wave["pattern"],
                metadata=wave["metadata"]
            )
            count += 1
        print(f"  [METABOLISM] Absorbed {count} Double Helix Waves into the HyperSphere.")

    def purge_meal(self):
        """Unloads the model to free resources."""
        print("  [DigestiveSystem] Purging meal (Exhaling)...")
        self.lungs.exhale()