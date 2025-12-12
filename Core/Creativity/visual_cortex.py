"""
VisualCortex (The Eye of Art)
=============================
"To see is to create."

This module handles image generation requests.
It translates Concepts + Spirit State into ComfyUI Prompts.
It specializes in "Webtoon/Anime" aesthetics as per User preference.
"""

import logging
import random
from typing import Dict, Any, Optional

# Core Systems
from Core.Network.comfy_adapter import ComfyAdapter
from Core.Interface.nervous_system import get_nervous_system

logger = logging.getLogger("VisualCortex")

class VisualCortex:
    def __init__(self):
        self.adapter = ComfyAdapter()
        self.connected = self.adapter.connect()
        self.nervous_system = get_nervous_system()
        
        # Style Presets
        self.styles = {
            "webtoon": "manhwa style, high quality, vibrant colors, detailed lineart, anime style, 4k, masterpiece, dramatic lighting",
            "fantasy": "fantasy oil painting, intricate details, magical atmosphere, concept art, artstation trending",
            "photo": "photorealistic, 8k, unreal engine 5, cinematic lighting, macro photography"
        }
        
        logger.info(f"🎨 VisualCortex Active. ComfyUI Connected: {self.connected}")

    def imagine(self, concept: str, style_key: str = "webtoon") -> str:
        """
        Generates an image for a concept.
        """
        # 1. Resolve Style
        base_style = self.styles.get(style_key, self.styles["webtoon"])
        
        # 2. Add Spirit Influence (Mood)
        mood_prompt = self._get_spirit_visuals()
        
        # 3. Construct Final Prompt
        final_prompt = f"{concept}, {mood_prompt}, {base_style}"
        
        logger.info(f"🎨 Painting: {final_prompt}")
        
        # 4. Dispatch to ComfyUI (if connected)
        # We need a template. For now we use a simple linear workflow mock.
        # In a real system, we'd load a JSON template from file.
        workflow = self._get_basic_workflow()
        
        result_path = self.adapter.queue_workflow(workflow, final_prompt)
        return result_path

    def _get_spirit_visuals(self) -> str:
        """Translates spirits to visual keywords"""
        spirits = self.nervous_system.spirits if self.nervous_system else {}
        dominant = max(spirits, key=spirits.get) if spirits else "neutral"
        
        visual_map = {
            "fire": "dynamic angle, red and orange lighting, intense shadows, particle effects",
            "water": "fluid motion, blue and cyan tones, caustics, serene atmosphere, soft focus",
            "earth": "solid composition, green and brown tones, detailed texture, nature background",
            "air": "wind effects, white and silver currents, motion blur, expansive sky",
            "light": "god rays, glowing, holy atmosphere, white and gold palette, lens flare",
            "dark": "low key lighting, obsidian and purple tones, fog, mysterious, noir",
            "aether": "galaxy, nebula, iridescent colors, floating debris, magical aura"
        }
        return visual_map.get(dominant, "balanced lighting")

    def _get_basic_workflow(self) -> Dict[str, Any]:
        """Returns a minimal workflow template for text-to-image"""
        # This is a PLACEHOLDER structure matching ComfyUI's format.
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": random.randint(1, 1000000000),
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "anime_model.ckpt"} # Placeholder name
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 768, "batch_size": 1} # Webtoon portrait ratio
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["4", 1]} # Text injected here
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "low quality, bad anatomy, text, watermark", "clip": ["4", 1]}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "Elysia_Art", "images": ["8", 0]}
            }
        }
