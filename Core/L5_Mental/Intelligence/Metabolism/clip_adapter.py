"""
CLIP Adapter: Image → 7D Wave DNA
=================================
Phase 75: Multi-Modal Prism

"A picture of fire and the word 'fire' must share the same essence."

This adapter uses OpenAI's CLIP model to convert images into
the same 7D Wave DNA space as text.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CLIPAdapter")

# Lazy imports to avoid loading heavy models at module load time
_clip_model = None
_clip_processor = None

@dataclass
class ImageDNA:
    """Wave DNA extracted from an image."""
    source_path: str
    embedding: list  # Raw CLIP embedding
    dynamics: dict   # 7D Wave DNA
    dominant_dimension: str
    description: str = ""


def _load_clip():
    """Lazy load CLIP model."""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            logger.info("🎨 Loading CLIP model (openai/clip-vit-base-patch32)...")
            
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                _clip_model = _clip_model.to("cuda")
                logger.info("   Using CUDA acceleration.")
            
            logger.info("✅ CLIP model loaded.")
            
        except ImportError as e:
            logger.error(f"❌ CLIP dependencies not installed: {e}")
            logger.error("   Run: pip install transformers torch pillow")
            raise
    
    return _clip_model, _clip_processor


def transduce_image(image_path: Union[str, Path]) -> Optional[ImageDNA]:
    """
    Convert an image file to 7D Wave DNA.
    
    The 7 dimensions are mapped from CLIP's 512D embedding:
    - Physical: Low-level features (edges, colors)
    - Functional: Object recognition features
    - Phenomenal: Scene/context features
    - Causal: Action/motion features
    - Mental: Semantic features
    - Structural: Composition features
    - Spiritual: Abstract/emotional features
    
    This is an approximation - CLIP doesn't natively separate these.
    We use different segments of the embedding vector.
    """
    try:
        from PIL import Image
        import torch
        
        model, processor = _load_clip()
        
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"❌ Image not found: {image_path}")
            return None
        
        image = Image.open(image_path).convert("RGB")
        
        # Process through CLIP
        inputs = processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Normalize
        embedding = image_features[0].cpu().numpy()
        embedding = embedding / (embedding ** 2).sum() ** 0.5
        
        # Map 512D to 7D
        # Split embedding into 7 segments and take mean
        segment_size = len(embedding) // 7
        dynamics = {
            "physical": float(abs(embedding[0:segment_size].mean())),
            "functional": float(abs(embedding[segment_size:segment_size*2].mean())),
            "phenomenal": float(abs(embedding[segment_size*2:segment_size*3].mean())),
            "causal": float(abs(embedding[segment_size*3:segment_size*4].mean())),
            "mental": float(abs(embedding[segment_size*4:segment_size*5].mean())),
            "structural": float(abs(embedding[segment_size*5:segment_size*6].mean())),
            "spiritual": float(abs(embedding[segment_size*6:].mean()))
        }
        
        # Find dominant dimension
        dominant = max(dynamics.items(), key=lambda x: x[1])
        
        return ImageDNA(
            source_path=str(image_path),
            embedding=embedding.tolist(),
            dynamics=dynamics,
            dominant_dimension=dominant[0],
            description=f"Image analyzed: {image_path.name}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to transduce image: {e}")
        return None


def compare_image_text(image_path: str, text: str) -> dict:
    """
    Compare an image with text using CLIP.
    Returns similarity score and both DNA representations.
    """
    try:
        from PIL import Image
        import torch
        
        model, processor = _load_clip()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process both
        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # CLIP similarity (logits)
        similarity = outputs.logits_per_image[0][0].item()
        
        # Get DNA for image
        image_dna = transduce_image(image_path)
        
        return {
            "similarity": similarity,
            "image_dna": image_dna.dynamics if image_dna else {},
            "text": text
        }
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return {"similarity": 0.0, "error": str(e)}


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎨 CLIP ADAPTER TEST")
    print("="*50)
    
    # Test with a sample image if available
    test_paths = [
        "data/test_image.jpg",
        "data/test_image.png",
        "C:/Users/USER/Pictures/sample.jpg"
    ]
    
    for path in test_paths:
        if Path(path).exists():
            result = transduce_image(path)
            if result:
                print(f"\n📸 Image: {result.source_path}")
                print(f"   Dominant: {result.dominant_dimension}")
                print(f"   7D DNA:")
                for dim, val in result.dynamics.items():
                    bar = "█" * int(val * 50)
                    print(f"      {dim:12s}: {val:.4f} {bar}")
            break
    else:
        print("\n⚠️ No test image found. Provide an image path to test.")
        print("   Usage: python clip_adapter.py")
