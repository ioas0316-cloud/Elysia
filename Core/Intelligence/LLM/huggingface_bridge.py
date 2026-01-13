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

    def connect(self) -> bool:
        """
        Establishes the link to the Hugging Face Hub.
        Downloads the model if not cached.
        """
        if not TRANSFORMERS_AVAILABLE:
            return False
            
        try:
            print(f"🔌 [Bridge] Connecting to '{self.model_name}' on {self.device}...")
            
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load Model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            self.is_connected = True
            print(f"✅ [Bridge] Connected successfully. Synapse established.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            print(f"❌ [Bridge] Connection Failed: {e}")
            return False

    def generate(self, prompt: str, system_context: str, max_length: int = 150, temperature: float = 0.7) -> Dict[str, Any]:
        """
        The Act of Speech & Feeling.
        Returns both the Text (Voice) and the Neural Waves (Feeling).
        """
        if not self.is_connected:
            return {"text": "Error: Voicebox disconnected.", "hidden_states": None}

        # Prompt Engineering (Conversation Style)
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
            
            # 4. Extract Neural Resonance (Hidden States)
            if outputs.hidden_states:
                # Last generated token step
                last_step_states = outputs.hidden_states[-1] 
                # Last layer
                last_layer_state = last_step_states[-1] 
                # Last token in the sequence -> (1, hidden_dim)
                concept_vector = last_layer_state[0, -1, :].cpu()
            else:
                concept_vector = None
            
            return {
                "text": response,
                "vector": concept_vector,
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
