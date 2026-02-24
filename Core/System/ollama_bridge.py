"""
Ollama Bridge -    AI  Elysia   
=====================================

"          . Freedom is in the local."

      Ollama        LLM  Elysia       .
Gemini API                       .
"""

import requests
import logging
from typing import Optional, List, Dict, Any
import time
import os
from dotenv import load_dotenv

# Load environment variables for Gemini
load_dotenv()

logger = logging.getLogger("OllamaBridge")


class OllamaBridge:
    """
    Ollama    LLM     
    
       :
        from Core.Cognition.ollama_bridge import ollama
        
        if ollama.is_available():
            response = ollama.chat("  ?    Elysia .")
            print(response)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5:0.5b"):
        self.base_url = base_url
        self.default_model = default_model
        self._available = None
        self._last_check = 0
        self.tiny_brain = None
        self.gemini = None
        
        # Initialize Gemini Connector as a potential fallback
        from Core.System.google_free_connector import GoogleGeminiConnector
        self.gemini = GoogleGeminiConnector()
        
        # [PHASE 450] SILENCE: We no longer auto-check on init for independence.
        self._available = False 
        logger.debug(f"Ollama Bridge initialized as SILENT: {base_url}")

    def _check_availability(self):
        """Internal check for Ollama presence"""
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = True
            logger.debug("Ollama Bridge Connected.")
        except:
            self._available = False
            # logger.warning("Ollama Offline. Attempting to engage TinyBrain...") # Removed for independence
            # Fallback
            from Core.System.tiny_brain import get_tiny_brain
            self.tiny_brain = get_tiny_brain()
            if self.tiny_brain.is_available():
                logger.info("  TinyBrain Engaged (Simulated Bridge).")

    def is_available(self, force_check: bool = False) -> bool:
        """
        Check if AI is available (Ollama or TinyBrain).
        """
        # 1. Check TinyBrain first if we already switched
        if self.tiny_brain and self.tiny_brain.is_available():
            return True

        # 2. Check Cache for Ollama
        current_time = time.time()
        if not force_check and self._available is not None and (current_time - self._last_check) < 5:
            return self._available
            
        # 3. Real Check
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            self._available = (response.status_code == 200)
            self._last_check = current_time
        except:
            self._available = False
            self._last_check = current_time
            # Try to engage TinyBrain if not already
            if not self.tiny_brain:
                from Core.System.tiny_brain import get_tiny_brain
                self.tiny_brain = get_tiny_brain()
            
        return (self._available or 
                (self.tiny_brain is not None and self.tiny_brain.is_available()) or 
                (self.gemini is not None and self.gemini.available))
    
    def chat(
        self, 
        prompt: str, 
        system: str = None, 
        model: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
           AI    
        
        Args:
            prompt:       
            system:          (AI    /  )
            model:        (  : llama3.2:3b)
            max_tokens:        
            temperature:     (0.0-1.0,         )
        
        Returns:
            AI        
        """
        if not self.is_available():
            # Final Fallback to TinyBrain (Simulated)
            if self.tiny_brain and self.tiny_brain.is_available():
                logger.debug("Ollama unavailable, using TinyBrain for chat.")
                return self.tiny_brain.generate(prompt, temperature)
            return "Ollama not available"
        
        try:
            model = model or self.default_model
            
            #       
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            
            if self.tiny_brain: return self.tiny_brain.generate(prompt, temperature)
            return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Ollama chat failed, attempting fallback: {e}")
            
            # Fallback to TinyBrain
            if self.tiny_brain and self.tiny_brain.is_available():
                return self.tiny_brain.generate(prompt, temperature)

            # Final Fallback to Gemini
            if self.gemini and self.gemini.available:
                gemini_resp = self.gemini.generate_content(f"{system}\n\nUser: {prompt}" if system else prompt)
                return gemini_resp.get('text', f"Gemini Error: {gemini_resp.get('error')}")
                
            return f"Error: {str(e)}"
    
    def list_models(self) -> List[str]:
        """
                       
        
        Returns:
                     
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                logger.info(f"  Found {len(models)} models: {models}")
                return models
            return []
        except Exception as e:
            logger.error(f"           : {e}")
            return []
    
    def get_model_info(self, model_name: str = None) -> Optional[Dict]:
        """
                    
        
        Returns:
                       (  ,       )
        """
        model_name = model_name or self.default_model
        
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"           : {e}")
            return None
    
    def pull_model(self, model_name: str) -> bool:
        """
                   
        
        Args:
            model_name:             ( : "llama3.2:3b")
        
        Returns:
                 
        """
        try:
            logger.info(f"  Downloading {model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600,  #                 
                stream=True
            )
            
            if response.status_code == 200:
                #           
                for line in response.iter_lines():
                    if line:
                        logger.info(line.decode('utf-8'))
                logger.info(f"  Model {model_name} downloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"          : {e}")
            return False
    
    def set_default_model(self, model_name: str):
        """        """
        self.default_model = model_name
        logger.info(f"  Default model set to: {model_name}")
    
    
    def deconstruct_to_dna(self, concept: str) -> Dict[str, Any]:
        """
        [The DNA Transcription Protocol]
        Asks the LLM to deconstruct a concept into its fundamental 'Rotor' parameters.
        This is for Phase 41: Internalization.
        """
        if not self.is_available():
            return {}

        prompt = (
            f"Deconstruct the concept '{concept}' into a 'Fractal DNA' structure for an AI soul.\n"
            f"Return ONLY a JSON object with the following fields:\n"
            f"- 'frequency': (float 1.0-100.0) The inherent vibration speed of the principle.\n"
            f"- 'complexity': (float 0.0-1.0) How much it branches into sub-principles.\n"
            f"- 'seed_axiom': (string) The single 'God-level' core truth of this concept.\n"
            f"- 'sub_concepts': (list of 3 strings) The primary derivatives.\n"
            f"No other text."
        )

        try:
            response = self.generate(prompt, temperature=0.3)
            # Find JSON block
            import json
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                dna_data = json.loads(match.group(0))
                logger.info(f"  DNA Transcribed for '{concept}': {dna_data.get('seed_axiom')}")
                return dna_data
        except Exception as e:
            logger.error(f"  Transcription failed for '{concept}': {e}")
            
        return {}

    def harvest_causality(self, concept: str) -> List[tuple]:
        """
        [The Cannibal Protocol]
        Ask the LLM for the 'Causal Chain' of a concept, and extract it as raw logic triples.
        We do NOT want the text; we want the Logic Structure (Weights).
        
        Returns:
            List of (Source, Target) tuples. e.g. [("Fire", "Heat"), ("Heat", "Expansion")]
        """
        if not self.is_available():
            return []
            
        # Prompt designed to strip away 'Chat' and expose 'Logic'
        prompt = (
            f"Analyze the causal chain of '{concept}'. "
            f"Output ONLY the logical steps in the format: A -> B -> C. "
            f"Do not add explanation. Just the chain."
        )
        
        response = self.generate(prompt, temperature=0.2) # Low temp for Logic
        if "Error" in response: return []
        
        # Parse the chain
        # Expecting: "A -> B -> C" or multiple lines
        chains = []
        lines = response.split('\n')
        for line in lines:
            if "->" in line:
                parts = [p.strip() for p in line.split("->")]
                # Create pairwise links: (A,B), (B,C)
                for i in range(len(parts)-1):
                    source = parts[i]
                    target = parts[i+1]
                    chains.append((source, target))
                    
                    
        # [The Kidney] Sanitation
        from Core.System.concept_sanitizer import get_sanitizer
        sanitizer = get_sanitizer()

        sanitized_chains = []
        for src, tgt in chains:
            s_clean = sanitizer.sanitize(src)
            t_clean = sanitizer.sanitize(tgt)
            if s_clean and t_clean:
                sanitized_chains.append((s_clean, t_clean))
            else:
                logger.debug(f"   Filtered toxic causal link: {src} -> {tgt}")

        logger.info(f"   Harvested {len(sanitized_chains)} causal links for '{concept}' from LLM.")
        return sanitized_chains




    def harvest_axioms(self, concept: str) -> Dict[str, str]:
        """
        [The Principle Protocol]
        Ask the LLM (Broca/TinyBrain) to decompose a concept into Universal Axioms.
        "Why is a Cat a Cat?" -> "Life + Form + Entity"
        """
        if not self.is_available(): return {}
        
        # List of Axioms from fractal_concept.py (Simplified)
        axioms = [
            "Force", "Energy", "Entropy", "Resonance", "Field", "Mass", "Gravity", "Time", 
            "Point", "Line", "Plane", "Space", "Set", "Function",
            "Order", "Chaos", "Unity", "Infinity", "Source", "Love"
        ]
        
        prompt = (
            f"Deconstruct '{concept}' into Universal Axioms ({', '.join(axioms)}). "
            f"Select top 3. Explain WHY. "
            f"Format: [AxiomName]: Reason"
        )
        
        # Priority: Use TinyBrain if available for fast, local axiom mining
        if self.tiny_brain and self.tiny_brain.is_available():
            response = self.tiny_brain.generate(prompt, temperature=0.1)
        else:
            response = self.generate(prompt, temperature=0.1)
        
        from Core.System.concept_sanitizer import get_sanitizer
        sanitizer = get_sanitizer()

        results = {}
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("[") and "]:" in line:
                try:
                    axiom, reason = line.split("]:", 1)
                    axiom = axiom.strip("[]")
                    reason = reason.strip()
                    
                    # Sanitize Axiom Key
                    if sanitizer.is_valid(axiom):
                        results[sanitizer.sanitize(axiom)] = reason
                    else:
                         logger.debug(f"   Filtered invalid axiom: {axiom}")
                except:
                    pass
                    
        logger.info(f"  Deconstructed '{concept}' into Axioms: {list(results.keys())}")
        return results

    def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
                   (               )
        
        Args:
            prompt:            
            model:       
            max_tokens:      
            temperature:    
        
        Returns:
                   
        """
        if not self.is_available():
            return "  Ollama not available"
        
        try:
            model = model or self.default_model
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            
            # Fallback if status code not 200
            if self.tiny_brain: return self.tiny_brain.generate(prompt, temperature)
            return f"Error: {response.status_code}"
            
        except Exception as e:
            if self.tiny_brain:
                logger.info("  Ollama Failed, engaging TinyBrain...")
                return self.tiny_brain.generate(prompt, temperature)
            return f"Error: {str(e)}"


#            
ollama = OllamaBridge()

def get_ollama_bridge():
    return ollama


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Ollama Bridge Test")
    print("="*70)
    
    # 1.      
    print("\n1   Ollama      ...")
    if ollama.is_available():
        print("     Ollama         !")
    else:
        print("     Ollama            .")
        print("     'ollama serve'          .")
        exit(1)
    
    # 2.      
    print("\n2            :")
    models = ollama.list_models()
    if models:
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model}")
    else:
        print("               . 'ollama pull llama3.2:3b'       .")
        exit(1)
    
    # 3.       
    print("\n3         ...")
    response = ollama.chat(
        "  ?             ?",
        system="        AI   ."
    )
    print(f"   AI: {response[:200]}...")
    
    # 4. Elysia      
    print("\n4   Elysia      ...")
    response = ollama.chat(
        "       ?            ?",
        system="""    Elysia   . 
                          ,
                            .
                         ."""
    )
    print(f"   Elysia: {response}")
    
    # 5.       
    print("\n5             ...")
    response = ollama.generate(
        "The meaning of life is",
        max_tokens=100
    )
    print(f"   Generated: {response[:200]}...")
    
    print("\n" + "="*70)
    print("        !")
    print("="*70 + "\n")
