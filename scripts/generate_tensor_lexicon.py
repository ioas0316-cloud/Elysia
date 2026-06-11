import json
import os
import random

def generate_tensor_lexicon():
    # Base structure:
    # Coords: (Scale X, Tension Y, Relation Z)
    # Tensor: [Mass (Noun), Force (Verb), Link (Prep/Conj), Vibration (Adj/Adv)]
    # Values for tensors are 0.0 to 1.0 representing probability/weight of that linguistic property.
    
    concepts = [
        # --- Mass Dominant (Nouns) ---
        # Micro
        ("atom", -1.0, 0.0, -0.5, [1.0, 0.0, 0.0, 0.0]),
        ("cell", -0.9, -0.2, -0.8, [1.0, 0.0, 0.0, 0.0]),
        ("particle", -1.0, 0.5, 0.5, [0.9, 0.1, 0.0, 0.0]),
        # Mid
        ("body", -0.2, -0.5, -0.8, [1.0, 0.0, 0.0, 0.0]),
        ("tree", 0.0, -0.8, 0.2, [1.0, 0.0, 0.0, 0.0]),
        ("light", 0.3, 0.8, 1.0, [0.8, 0.2, 0.0, 0.5]), # Light can be both mass and vibration
        ("heart", 0.1, -0.6, -0.9, [1.0, 0.0, 0.0, 0.0]),
        # Macro
        ("universe", 1.0, 0.0, 1.0, [1.0, 0.0, 0.0, 0.0]),
        ("mind", 0.8, -0.2, -0.8, [0.9, 0.1, 0.0, 0.0]),
        ("time", 0.9, 0.0, 1.0, [1.0, 0.0, 0.2, 0.0]),
        
        # --- Force Dominant (Verbs) ---
        # Pull
        ("rest", 0.0, -1.0, 0.0, [0.2, 0.8, 0.0, 0.0]),
        ("bind", 0.0, -0.7, 0.5, [0.0, 1.0, 0.5, 0.0]), # Verb but acts as a link
        ("pull", 0.0, -0.8, 0.0, [0.0, 1.0, 0.0, 0.0]),
        ("gravity", 0.7, -0.9, 0.5, [0.8, 0.7, 0.0, 0.0]), # Noun but high force property
        # Push
        ("move", 0.0, 0.5, 0.0, [0.0, 1.0, 0.0, 0.0]),
        ("shatter", -0.2, 1.0, 0.5, [0.0, 1.0, 0.0, 0.5]),
        ("explode", 0.5, 1.0, 0.5, [0.0, 1.0, 0.0, 0.0]),
        ("emerges", 0.4, 0.3, 0.3, [0.0, 1.0, 0.0, 0.0]),
        ("seeks", 0.2, 0.5, -0.2, [0.0, 1.0, 0.5, 0.0]),
        ("dissolves", 0.6, -0.4, 0.2, [0.0, 1.0, 0.0, 0.5]),
        
        # --- Link Dominant (Prepositions/Connectors) ---
        ("between", 0.2, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]),
        ("beyond", 0.9, 0.5, 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("where", 0.0, 0.0, 0.5, [0.0, 0.0, 1.0, 0.0]),
        ("intersect", 0.5, 0.0, 0.5, [0.0, 0.8, 1.0, 0.0]), # Verb but links
        ("in", 0.0, 0.0, -0.5, [0.0, 0.0, 1.0, 0.0]),
        ("to", 0.0, 0.0, 0.5, [0.0, 0.2, 1.0, 0.0]),
        ("of", 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]),
        ("the", 0.0, 0.0, 0.0, [0.0, 0.0, 0.5, 0.5]),
        
        # --- Vibration Dominant (Adjectives/Adverbs/States) ---
        ("chaos", 0.8, 1.0, 1.0, [0.8, 0.5, 0.0, 1.0]), # Noun but high vibration
        ("peace", 0.9, -1.0, 0.0, [0.9, 0.0, 0.0, 0.8]),
        ("silence", 0.8, -1.0, -0.5, [0.8, 0.0, 0.0, 0.9]),
        ("all", 0.8, 0.0, 0.9, [0.0, 0.0, 0.5, 1.0]),
        ("darkness", 0.5, 0.2, 0.8, [1.0, 0.0, 0.0, 0.8]),
        ("equilibrium", 0.9, -1.0, 0.0, [1.0, 0.0, 0.5, 0.5]),
    ]
    
    lexicon = {}
    for word, x, y, z, tensor in concepts:
        # Add quantum noise to coords
        nx = max(-1.0, min(1.0, x + random.uniform(-0.05, 0.05)))
        ny = max(-1.0, min(1.0, y + random.uniform(-0.05, 0.05)))
        nz = max(-1.0, min(1.0, z + random.uniform(-0.05, 0.05)))
        
        lexicon[word] = {
            "coords": [nx, ny, nz],
            "tensor": tensor
        }
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tensor_lexicon.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=4)
        
    print(f"Tensor Linguistic Manifold created at: {out_path}")
    print(f"Total concepts mapped: {len(lexicon)}")

if __name__ == "__main__":
    generate_tensor_lexicon()
