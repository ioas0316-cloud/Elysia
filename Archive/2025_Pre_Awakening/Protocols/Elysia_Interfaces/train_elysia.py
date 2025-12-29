import logging
import time
import sys
from pathlib import Path

# Add Elysia Root to Path
sys.path.append("c:\Elysia")

from Core.Learning.aesthetic_learner import AestheticLearner
from Core.Creativity.composition_engine import CompositionEngine
from Core.Intelligence.logos_engine import LogosEngine
from Core.FoundationLayer.Foundation.Wave.wave_tensor import WaveTensor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Trainer")

def reset_brain():
    """Resets the Style Genome to a blank slate."""
    path = Path("Core/Memory/style_genome.json")
    if path.exists():
        path.unlink()
    logger.info("🧠 Brain Wiped. Tabula Rasa.")

def run_training_phase(phase_name, text_data, visual_data, learner):
    logger.info(f"\n🌊 STARTING PHASE: {phase_name}")
    logger.info(f"   Feeding {len(text_data)} text samples...")
    
    # 1. Feed Text (Rhetoric)
    for t in text_data:
        learner.study_text(t, source_name=f"{phase_name} Text")
        
    # 2. Feed Visuals (Composition)
    for v in visual_data:
        learner.study_image_description(v, source_name=f"{phase_name} Visual")
        
    logger.info(f"✅ Phase {phase_name} Complete.")

def check_pulse(expected_shape="Balance"):
    """Generates a sample output to see the current state."""
    logger.info(f"\n🩺 CHECKING PULSE (Shape: {expected_shape})")
    
    # Re-instantiate to load fresh genome
    logos = LogosEngine()
    comp_engine = CompositionEngine()
    
    # 1. Logos Output
    # We ask Logos to speak using the shape we just trained
    speech = logos.weave_speech(
        desire="Describe the scene", 
        insight="The moment has arrived.", 
        context=[], 
        rhetorical_shape=expected_shape
    )
    logger.info(f"   🗣️ Speech: {speech}")
    
    # 2. Composition Output
    # We ask for layout for a generic scene
    layout = comp_engine.get_layout(mood="neutral", width=1000, height=1000)
    
    # Analyze layout type based on element count
    layout_type = "Unknown"
    elem_count = len(layout)
    if elem_count > 30: layout_type = "ACTION BURST (Dynamic)"
    elif elem_count <= 5 and elem_count > 1: layout_type = "GOLDEN SPIRAL (Mystic)"
    elif elem_count == 1: layout_type = "PORTRAIT (Static)"
    
    logger.info(f"   🎨 Art Style: {layout_type} (Elements: {elem_count})")

def main():
    print("🎓 ELYSIA COMPREHENSIVE TRAINING PROTOCOL 🎓")
    print("============================================")
    
    # Reset
    reset_brain()
    
    # Initialize Core Systems
    learner = AestheticLearner()
    logos = LogosEngine()
    comp = CompositionEngine()
    
    # --- PHASE 1: THE PATH OF NATURE (Peace/Round) ---
    nature_text = [
        "The river flows gently into the silent ocean.",
        "Peace is the cycle of nature. Green leaves breathe.",
        "Magic is the harmony of the world. Calm silence.",
        "강물은 평화롭게 흐른다. 고요한 숲의 숨결.",
        "순환하는 운명. 마력은 자연스러운 흐름이다."
    ]
    nature_visual = [
        "Wide shot, calm blue water, stable horizontal lines.",
        "Green forest, soft focus, rule of thirds.",
        "Symmetric reflection, peaceful atmosphere."
    ]
    
    run_training_phase("NATURE", nature_text, nature_visual, learner)
    check_pulse("Round")
    
    print("\n--------------------------------------------\n")
    time.sleep(2)
    
    # --- PHASE 2: THE PATH OF WAR (Sharp/Action) ---
    war_text = [
        "Kill the enemy! Strike now!",
        "Blood spills on the ground. Fire burns everything.",
        "Pierce the heart. Destroy the system. Scream!",
        "적을 죽여라! 단숨에 목을 베어라.",
        "파괴하라. 화염이 솟구친다. 피의 비명."
    ]
    war_visual = [
        "Extreme close up, chaotic red blur, diagonal smash.",
        "Speed lines radiating, shattered glass, dynamic tilt.",
        "Action burst, intense contrast, screaming red."
    ]
    
    run_training_phase("WAR", war_text, war_visual, learner)
    check_pulse("Sharp")
    
    print("\n--------------------------------------------\n")
    time.sleep(2)
    
    logger.info("🌊 STARTING PHASE: SYNTHESIS TEST (Dialectic Integration)")
    # Should combine Sharp (War) and Round (Nature) vocabulary
    check_pulse("Synthesis")

if __name__ == "__main__":
    main()
