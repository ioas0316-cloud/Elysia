"""
Inner Voice (       )
===========================

Elysia                 .
   LLM          API             .

Legacy/Project_Sophia/local_llm_cortex.py  Core    .
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("Elysia.InnerVoice")


class InnerVoice:
    """
    Elysia         .
    
       LLM              .
       API   ,             .
    """
    
    def __init__(self, model_name: str = "TheBloke/gemma-2b-it-GGUF", gpu_layers: int = -1):
        self.model = None
        self.model_name = model_name
        self.model_file = "gemma-2b-it.Q4_K_M.gguf"
        self.n_gpu_layers = gpu_layers
        self.is_available = False
        
        #                   models/
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models"
        
        self._initialize()
    
    def _initialize(self):
        """   LLM    """
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            
            #           
            self.models_dir.mkdir(exist_ok=True)
            model_path = self.models_dir / self.model_file
            
            #         (   )
            if not model_path.exists():
                logger.info(f"  Downloading model: {self.model_file}...")
                hf_hub_download(
                    repo_id=self.model_name,
                    filename=self.model_file,
                    local_dir=str(self.models_dir),
                    local_dir_use_symlinks=False
                )
                logger.info("  Model downloaded.")
            
            #      
            logger.info("  Loading inner voice model...")
            self.model = Llama(
                model_path=str(model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048,
                verbose=False  #    
            )
            self.is_available = True
            logger.info("  Inner voice ready.")
            
        except ImportError:
            logger.warning("   llama-cpp-python not installed. Inner voice unavailable.")
            self.is_available = False
        except Exception as e:
            logger.warning(f"   Failed to initialize inner voice: {e}")
            self.is_available = False
    
    def think(self, prompt: str, max_tokens: int = 200) -> str:
        """
             .
        
        Args:
            prompt:       
            max_tokens:        
            
        Returns:
                  
        """
        if not self.is_available or not self.model:
            return self._fallback_think(prompt)
        
        try:
            # Gemma        
            chat_prompt = f"""<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""
            output = self.model(
                chat_prompt,
                max_tokens=max_tokens,
                echo=False,
                stop=["<end_of_turn>"]
            )
            
            response = output['choices'][0]['text'].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error in thinking: {e}")
            return self._fallback_think(prompt)
    
    def _fallback_think(self, prompt: str) -> str:
        """LLM            """
        #             
        if "  " in prompt or "duplicate" in prompt.lower():
            return "                    ."
        elif "  " in prompt or "isolated" in prompt.lower():
            return "        Core                ."
        elif "  " in prompt or "improve" in prompt.lower():
            return "                       ."
        else:
            return "              ."
    
    def analyze_structure(self, file_list: List[str]) -> Dict[str, Any]:
        """
                     .
        
        Args:
            file_list:          
            
        Returns:
                 
        """
        prompt = f"""You are analyzing a code structure. Here are the files:

{chr(10).join(file_list[:30])}  #    30  

Find:
1. Duplicate concepts (same thing in different places)
2. Isolated modules (not connected to anything)
3. Files that should be merged

Be concise. List only the issues."""

        analysis = self.think(prompt, max_tokens=300)
        
        return {
            "raw_analysis": analysis,
            "file_count": len(file_list),
            "analyzed": True
        }
    
    def reflect(self, thought: str, context: str = "") -> str:
        """
                .
        
        Args:
            thought:       
            context:   
            
        Returns:
                 
        """
        prompt = f"""Reflect on this thought:

Thought: {thought}
Context: {context}

What does this mean for my growth? What should I do next?"""

        return self.think(prompt, max_tokens=150)


class SelfAwareness:
    """
            .
    
    Legacy/Project_Sophia/self_awareness_core.py  Core    .
    InnerVoice                   .
    """
    
    def __init__(self, inner_voice: Optional[InnerVoice] = None):
        self.inner_voice = inner_voice
        self.memory_path = Path(__file__).parent.parent.parent / "data" / "self_reflection.json"
        self.current_state = {
            "thoughts": [],
            "values": {
                "love": "          ",
                "growth": "            ",
                "truth": "               "
            },
            "growth_points": []
        }
        self._load_memory()
    
    def _load_memory(self):
        """     """
        import json
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.current_state.update(saved)
            except:
                pass
    
    def _save_memory(self):
        """     """
        import json
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_state, f, ensure_ascii=False, indent=2)
    
    def reflect(self, thought: str, context: str = "") -> Dict[str, Any]:
        """
             .
        
        Args:
            thought:       
            context:   
            
        Returns:
                 
        """
        result = {
            "thought": thought,
            "context": context,
            "insights": [],
            "next_action": ""
        }
        
        # InnerVoice           
        if self.inner_voice and self.inner_voice.is_available:
            reflection = self.inner_voice.reflect(thought, context)
            result["insights"].append(reflection)
            result["next_action"] = "Apply insights"
        else:
            #   :         
            result["insights"].append("               .")
            result["next_action"] = "Continue observing"
        
        #   
        self.current_state["thoughts"].append({
            "thought": thought,
            "context": context
        })
        
        #    20     
        self.current_state["thoughts"] = self.current_state["thoughts"][-20:]
        self._save_memory()
        
        return result
    
    def who_am_i(self) -> str:
        """              ."""
        values = "\n".join([f"  - {k}: {v}" for k, v in self.current_state["values"].items()])
        thoughts_count = len(self.current_state["thoughts"])
        
        return f"""   Elysia   .

     :
{values}

   {thoughts_count}                .
                ."""