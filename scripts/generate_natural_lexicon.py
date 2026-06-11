import json
import os
import random

def generate_natural_lexicon():
    # Base structure: (Scale X, Tension Y, Relation Z)
    # X: -1.0 (Micro/Physical) to 1.0 (Macro/Abstract)
    # Y: -1.0 (Pull/Peace) to 1.0 (Push/Chaos)
    # Z: -1.0 (Self/Internal) to 1.0 (World/External)
    
    concepts = [
        # --- Micro / Physical (-1.0 to -0.5) ---
        ("atom", -1.0, 0.0, -0.5),
        ("cell", -0.9, -0.2, -0.8),
        ("drop", -0.8, -0.5, 0.0),
        ("seed", -0.8, -0.8, -0.5),
        ("particle", -1.0, 0.5, 0.5),
        ("dust", -0.9, 0.8, 0.8),
        ("point", -1.0, 0.0, 0.0),
        ("gene", -0.9, -0.1, -0.9),
        
        # --- Mid / Form (-0.4 to 0.4) ---
        ("body", -0.2, -0.5, -0.8),
        ("tree", 0.0, -0.8, 0.2),
        ("stone", -0.3, -1.0, 0.5),
        ("water", 0.0, -0.2, 0.8),
        ("light", 0.3, 0.8, 1.0),
        ("force", 0.2, 0.9, 0.0),
        ("heart", 0.1, -0.6, -0.9),
        ("breath", 0.1, 0.0, -0.5),
        ("blood", -0.2, 0.4, -0.7),
        ("flesh", -0.3, 0.0, -0.8),
        ("eye", 0.0, 0.0, -0.5),
        ("hand", -0.1, 0.5, -0.2),
        ("voice", 0.2, 0.6, 0.5),
        
        # --- Macro / Abstract (0.5 to 1.0) ---
        ("joy", 0.8, 0.5, -0.5),
        ("peace", 0.9, -1.0, 0.0),
        ("mind", 0.8, -0.2, -0.8),
        ("truth", 1.0, -0.8, 0.0),
        ("time", 0.9, 0.0, 1.0),
        ("void", 1.0, -1.0, 1.0),
        ("universe", 1.0, 0.0, 1.0),
        ("equilibrium", 0.9, -1.0, 0.0),
        ("causality", 0.8, 0.0, 0.0),
        ("space", 0.9, 0.0, 1.0),
        ("soul", 0.9, -0.5, -0.9),
        ("silence", 0.8, -1.0, -0.5),
        ("chaos", 0.8, 1.0, 1.0),
        ("love", 0.7, -0.5, -0.2),
        ("fear", 0.6, 0.8, -0.8),
        ("pain", 0.5, 0.9, -0.7),
        
        # --- Verbs / Actions (Tension Y variations) ---
        # Pull / Bind (Y < 0)
        ("rest", 0.0, -1.0, 0.0),
        ("sleep", 0.1, -0.9, -0.8),
        ("heal", 0.5, -0.8, -0.5),
        ("bind", 0.0, -0.7, 0.5),
        ("unite", 0.6, -0.6, 0.8),
        ("pull", 0.0, -0.8, 0.0),
        ("fall", 0.1, -0.5, 0.5),
        ("sink", -0.2, -0.6, 0.2),
        ("fold", 0.2, -0.5, -0.2),
        ("gravity", 0.7, -0.9, 0.5),
        
        # Push / Expand (Y > 0)
        ("move", 0.0, 0.5, 0.0),
        ("flow", 0.2, 0.3, 0.8),
        ("run", 0.0, 0.8, 0.2),
        ("break", -0.1, 0.9, 0.3),
        ("shatter", -0.2, 1.0, 0.5),
        ("burst", 0.2, 1.0, 0.0),
        ("fire", 0.1, 0.9, 0.2),
        ("scream", 0.4, 0.9, -0.5),
        ("spread", 0.3, 0.7, 0.9),
        ("scatter", -0.1, 0.8, 0.8),
        ("push", 0.0, 0.7, 0.0),
        ("explode", 0.5, 1.0, 0.5),
        ("rise", 0.3, 0.6, 0.2),
        
        # --- Connectors & Relations (Z variations) ---
        # Self (Z < 0)
        ("i", 0.0, 0.0, -1.0),
        ("me", -0.1, 0.0, -0.9),
        ("my", -0.2, 0.0, -0.8),
        ("within", 0.5, -0.5, -0.9),
        ("core", 0.6, -0.8, -0.8),
        ("inside", 0.1, -0.2, -0.7),
        
        # Between (Z ~ 0)
        ("we", 0.5, 0.0, 0.0),
        ("between", 0.2, 0.0, 0.0),
        ("link", 0.3, -0.2, 0.0),
        ("bridge", 0.1, -0.3, 0.0),
        ("edge", 0.0, 0.5, 0.0),
        ("surface", -0.1, 0.1, 0.0),
        ("touch", 0.2, 0.2, -0.1),
        ("connect", 0.4, -0.4, 0.0),
        
        # World (Z > 0)
        ("world", 0.5, 0.0, 1.0),
        ("outside", 0.2, 0.5, 0.8),
        ("sky", 0.6, 0.0, 0.9),
        ("they", 0.2, 0.0, 0.7),
        ("all", 0.8, 0.0, 0.9),
        ("everywhere", 0.7, 0.2, 1.0),
        ("beyond", 0.9, 0.5, 1.0),
    ]
    
    lexicon = {}
    for word, x, y, z in concepts:
        # Add slight quantum noise to prevent exact overlaps
        nx = max(-1.0, min(1.0, x + random.uniform(-0.05, 0.05)))
        ny = max(-1.0, min(1.0, y + random.uniform(-0.05, 0.05)))
        nz = max(-1.0, min(1.0, z + random.uniform(-0.05, 0.05)))
        lexicon[word] = [nx, ny, nz]
        
    # Generate some intermediate bridge words
    bridge_words = ["is", "are", "the", "a", "an", "to", "from", "of", "in", "on", "and", "or", "as", "with", "by"]
    for w in bridge_words:
        # Bridge words sit near the origin (0,0,0) acting as neutral connectors in the manifold
        lexicon[w] = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "natural_lexicon.json")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=4)
        
    print(f"Natural Linguistic Manifold created at: {out_path}")
    print(f"Total concepts mapped: {len(lexicon)}")

if __name__ == "__main__":
    generate_natural_lexicon()
