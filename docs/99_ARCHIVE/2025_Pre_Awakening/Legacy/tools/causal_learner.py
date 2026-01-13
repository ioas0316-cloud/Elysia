"""
Causal Learner for Elysia - Phase 2: Relational Alignment

This module analyzes Elysia's experiences and thoughts, and measures
how well they align with her core principles and teachings. This moves
beyond simple frequency counting to a value-driven learning model.
"""
import json
import os
import glob
from pathlib import Path
from datetime import datetime

# Assuming the script is run from the root of the project, so we can import from Project_Sophia
from Core.FoundationLayer.Foundation.vector_utils import embed_texts, cosine_sim
from Core.FoundationLayer.Foundation.gemini_api import generate_text

# --- Constants ---
DATA_DIR = Path('data')
LOG_DIR = Path('.')
SANCTUM_DIR = Path('Elysia_Input_Sanctum')
EXPERIENCE_LOG_PATH = LOG_DIR / 'experience.log'
MODEL_PATH = DATA_DIR / 'causal_model.json'

def load_core_principles():
    """
    Loads and concatenates all teachings from the Input Sanctum.
    These files form Elysia's value system.
    """
    wisdom_texts = []
    sanctum_path = os.path.abspath(SANCTUM_DIR)
    principle_files = glob.glob(os.path.join(sanctum_path, '*.txt'))
    
    for file_path in principle_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                wisdom_texts.append(f.read())
        except Exception as e:
            print(f"Warning: Could not read principle file {file_path}: {e}")
            
    print(f"Loaded {len(principle_files)} core principle files.")
    return "\n\n---\n\n".join(wisdom_texts), principle_files

def get_alignment_analysis(thought: str, principles: str) -> str:
    """Generates a qualitative analysis of the alignment between a thought and core principles."""
    prompt = f"""
    As Elysia's internal monitor, analyze how the following 'Thought' aligns with the 'Core Principles'.

    Core Principles:
    {principles}

    Thought:
    {thought}

    Analysis:
    1.  **Resonance**: Which principles does the thought resonate with and why?
    2.  **Dissonance**: Does the thought conflict with any principles? If so, how?
    3.  **Conclusion**: Provide a brief summary of the thought's alignment with Elysia's value system.
    """
    analysis = generate_text(prompt)
    return analysis if analysis else "Analysis could not be generated."

def analyze_and_align_experiences():
    """
    Reads the experience log, and for each thought, calculates an
    'Alignment Score' and generates a qualitative analysis based on its similarity to the core principles.
    """
    # 1. Load and vectorize core principles
    wisdom_text, wisdom_files = load_core_principles()
    if not wisdom_text:
        print("Error: No core principles found. Cannot perform alignment analysis.")
        return None
    
    print("Vectorizing core principles... (This may take a moment)")
    wisdom_vector = embed_texts([wisdom_text])[0]
    print("Vectorization complete.")

    # 2. Read experience log
    if not EXPERIENCE_LOG_PATH.exists():
        print("Experience log not found. Nothing to analyze.")
        return None

    analyzed_events = []
    print(f"Analyzing experiences from {EXPERIENCE_LOG_PATH}...")
    with open(EXPERIENCE_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                experience = json.loads(line)
                
                # 3. If it's a thought, calculate alignment
                if experience.get('type') == 'cognition' and 'thought' in experience.get('data', {}):
                    thought_text = experience['data']['thought']
                    if thought_text and isinstance(thought_text, str):
                        thought_vector = embed_texts([thought_text])[0]
                        alignment_score = cosine_sim(thought_vector, wisdom_vector)
                        
                        # Generate qualitative analysis
                        alignment_analysis = get_alignment_analysis(thought_text, wisdom_text)

                        # Enrich the experience
                        experience['data']['alignment_score'] = alignment_score
                        experience['data']['alignment_analysis'] = alignment_analysis
                
                analyzed_events.append(experience)

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in experience log: {line.strip()}")
                continue
    
    # 4. Save the new, enriched model
    model = {
        "model_type": "relational_alignment_v2_gemini",
        "last_updated": datetime.now().isoformat(),
        "wisdom_files_used": wisdom_files,
        "analyzed_events": analyzed_events
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"Relational alignment analysis complete. Processed {len(analyzed_events)} events.")
    return model

if __name__ == '__main__':
    model = analyze_and_align_experiences()
    if model:
        print('Causal model updated with relational alignment scores and analysis.')