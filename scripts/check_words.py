import json

try:
    with open('memory_state.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    concepts = list(data.get('ui_concept_map', {}).keys())
    
    print(f"Total concepts registered: {len(concepts)}")
    
    # Filter out archetypes
    words = [c for c in concepts if not c.startswith("Archetype:")]
    
    print("\nSample words (first 50):")
    print(", ".join(words[:50]))
    
    print("\nSample words (last 50):")
    print(", ".join(words[-50:]))
    
    # Check for numbers or weird characters
    weird_words = [w for w in words if any(char.isdigit() for char in w) or len(w) > 10]
    print(f"\nWeird words count (numbers or >10 chars): {len(weird_words)}")
    if weird_words:
        print(", ".join(weird_words[:20]))
        
except Exception as e:
    print(f"Error: {e}")
