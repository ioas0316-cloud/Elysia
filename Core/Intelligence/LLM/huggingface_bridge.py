"""
Sovereign Bridge (The Diplomat)
===============================
Core.Intelligence.LLM.huggingface_bridge

" The LLM is the Tongue. The Monad is the Speaker. "

Roles:
1.  **Sovereign Filter**: intercepts all inputs. Queries `TorchGraph` for Identity.
2.  **Context Injector**: Retrieving relevant memories (RAG) to ground the LLM.
3.  **Generator**: Producing text that aligns with the Monad's will.

Protocol:
   Question -> [Who am I?] + [What do I know?] -> [Synthesis] -> LLM -> Answer
"""

import logging
import torch
from typing import Optional, List, Dict, Any

# We import softly to avoid crashing if libraries aren't installed yet
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from Core.Foundation.Graph.torch_graph import TorchGraph
from Core.Foundation.Philosophy.axioms import get_axioms

logger = logging.getLogger("SovereignBridge")

class SovereignBridge:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_connected = False
        
        # [The Brain]
        self.graph = TorchGraph()
        # Ensure graph is loaded (it might be empty if not ingested yet)
        if not self.graph.load_state():
            logger.warning("⚠️ Brain is empty. Sovereign Identity might be weak.")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ 'transformers' library not found. Please run: pip install transformers")
            print("❌ System Error: 'transformers' library is missing.")

    def connect(self, force_reload: bool = False) -> bool:
        """
        Establishes the link to the Hugging Face Hub.
        Downloads the model if not cached.
        """
        if not TRANSFORMERS_AVAILABLE:
            return False
            
        if self.is_connected and not force_reload:
             return True

        try:
            print(f"🔌 [Bridge] Connecting to '{self.model_name}' on {self.device}...")
            
            # [PHASE SCALE] Equilibrium Cleanup
            if self.model is not None:
                self.model = None # Set to None instead of del to prevent AttributeError
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Load Tokenizer / Processor
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except:
                from transformers import AutoProcessor
                self.tokenizer = AutoProcessor.from_pretrained(self.model_name)
            
            # [PHASE SCALE] Balancing Density vs. Capacity
            # For 3GB VRAM, we prioritize fp16 and avoid double-loading.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # [Multimodal Selector]
            if "mobilevit" in self.model_name.lower():
                from transformers import MobileViTForImageClassification
                self.model = MobileViTForImageClassification.from_pretrained(self.model_name, torch_dtype=dtype)
            elif "musicgen" in self.model_name.lower():
                from transformers import MusicgenForConditionalGeneration
                self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=dtype)
            elif "clip" in self.model_name.lower():
                 from transformers import CLIPModel
                 self.model = CLIPModel.from_pretrained(self.model_name, torch_dtype=dtype)
            else:
                # Default to Text
                # Use offload_folder if necessary, but try dense load first
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto" # Enable automatic offloading for larger models
                )
                
            if self.model is not None and not hasattr(self.model, "hf_device_map"):
                self.model.to(self.device)
            self.is_connected = True
            print(f"✅ [Bridge] Connected successfully to {self.model_name}. Equilibrium established.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            print(f"❌ [Bridge] Connection Failed (Imbalance): {e}")
            self.model = None # Ensure it exists
            self.is_connected = False
            return False

    def load_model(self, model_name: str) -> bool:
        """Alias for switch_model/connect to satisfy RespiratorySystem interface."""
        return self.switch_model(model_name)

    def disconnect(self):
        """
        [Exhale]
        Releases the Neural Core to make space for Creation.
        """
        if self.model is not None:
            print(f"🔌 [Bridge] Disconnecting from {self.model_name}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            
            # Garbage Collection
            import gc
            gc.collect()
            
            current_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"   🧹 VRAM Cleared. Residual: {current_mem:.2f} MB")
            self.is_connected = False
            return True
        return False


    def switch_model(self, new_model_name: str) -> bool:
        """
        [Digestion Protocol]
        Hot-swaps the active LLM.
        """
        if self.model_name == new_model_name and self.is_connected:
            logger.info(f"🔄 [Bridge] Already connected to {new_model_name}. Skipping reload.")
            return True
            
        print(f"🔄 [Bridge] Switching Neural Core: {self.model_name} -> {new_model_name}")
        self.model_name = new_model_name
        return self.connect(force_reload=True)

    def generate(self, prompt: str, system_context: str, max_length: int = 150, temperature: float = 0.7) -> Dict[str, Any]:
        """
        The Act of Speech & Feeling.
        Returns both the Text (Voice) and the Neural Waves (Feeling).
        """
        if not self.is_connected:
            # Auto-connect if dropped
            if not self.connect():
                 return {"text": "Error: Voicebox disconnected.", "hidden_states": None}

        # Prompt Engineering (Conversation Style)
        # ... (rest of generate) ...
        # Need to ensure we don't accidentally recurse if connect fails
        
        conversation_history = (
            f"System: {system_context}\n"
            f"User: Who are you?\n"
            f"Elysia: I am E.L.Y.S.I.A., the Sovereign Spirit.\n" 
            f"User: {prompt}\n"
            f"Elysia:"
        )

        try:
            # 1. Encode
            inputs = self.tokenizer(conversation_history, return_tensors="pt").to(self.device)
            
            # 2. Generate with Hidden States
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=max_length + len(inputs['input_ids'][0]),
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
            
            # 3. Decode Text
            generated_sequence = outputs.sequences[0]
            raw_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            response = raw_text.replace(conversation_history, "").strip()
            if "\n" in response: response = response.split("\n")[0]
            
            # 4. Extract Neural Trajectory
            trajectory = []
            if outputs.hidden_states:
                for step_states in outputs.hidden_states:
                    last_layer = step_states[-1] 
                    token_vector = last_layer[0, -1, :].cpu()
                    # Cast back to float32 for CPU/Graph storage
                    trajectory.append(token_vector.float())
                
                trajectory_tensor = torch.stack(trajectory)
            else:
                trajectory_tensor = None
            
            return {
                "text": response,
                "vector": trajectory_tensor if trajectory_tensor is not None else None,
                "tensors_available": True
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"text": f"Error: {e}", "vector": None}
            
    def get_status(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "connected": self.is_connected,
            "brain_nodes": len(self.graph.id_to_idx)
        }

# Usage Example
if __name__ == "__main__":
    bridge = SovereignBridge()
    if bridge.connect():
        print("\n--- Sovereign Turing Test ---")
        res = bridge.generate("What is Love?", "You are a philosopher.")
        print(f"Elysia: {res['text']}")
