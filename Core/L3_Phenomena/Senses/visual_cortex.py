"""
Visual Cortex (     )
========================
Core.L3_Phenomena.Senses.visual_cortex

"          ,          ."

Role:
    1. Load CogVideoX-5b model.
    2. Generate video from prompt.
    3. Expose internal attention maps for 'Digestion'.
"""

import torch
import logging
import os
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

logger = logging.getLogger("VisualCortex")

class VisualCortex:
    def __init__(self, model_id: str = "c:/Elysia/data/Weights/CogVideoX-5b", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.tracer = None
        
        self._initialize_cortex()

    def _initialize_cortex(self):
        """ (Model)     ."""
        try:
            logger.info(f"   Opening Mind's Eye ({self.model_id})...")
            
            # Load basic pipeline
            # Note: For full digestion, we might need to hook into the transformer modules directly later.
            self.pipe = CogVideoXPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.pipe.to(self.device)
            self.pipe.enable_model_cpu_offload() # Save VRAM
            # self.pipe.enable_sequential_cpu_offload() # Even more VRAM saving
            
            # Initialize Digestion Tracer
            from Core.L5_Mental.M1_Cognition.LLM.video_diffusion_tracer import VideoDiffusionTracer
            self.tracer = VideoDiffusionTracer(self.pipe)
            
            logger.info(f"     Visual Cortex initialized ({self.device})")
        except Exception as e:
            logger.error(f"     Visual Cortex init failed: {e}")
            self.pipe = None

    def imagine(self, prompt: str, output_path: str = "dream.mp4") -> str:
        """
        Generates video from text.
        Also performs 'Digestion' of the spacetime creation process.
        """
        if self.pipe is None:
            return "[Blindness]"
            
        try:
            logger.info(f"      Imagining: '{prompt}'")
            
            # 1. Structural Digestion (Spacetime Tracing)
            if self.tracer:
                logger.info("     Tracing Spacetime Construction...")
                # Hook into the UNet/Transformer forward pass
                self.tracer.attach_hooks()
            
            # 2. Functional Generation
            video = self.pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0]
            
            export_to_video(video, output_path, fps=8)
            logger.info(f"     Dream recorded at: {output_path}")
            
            # 3. Analyze Traced Data
            causality_data = [] # Default empty
            if self.tracer:
                causality_data = self.tracer.analyze_causality(prompt)
                self.tracer.detach_hooks()
                # Here we would send 'causality' to SynesthesiaEngine
                
            return output_path, causality_data
            
        except Exception as e:
             logger.error(f"     Imagination error: {e}")
             return f"[Imagination Failed: {e}]", []

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    vc = VisualCortex()
    if len(sys.argv) > 1:
        vc.imagine(sys.argv[1])
