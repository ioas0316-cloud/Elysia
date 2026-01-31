
import json
import os

def sanitize_json_brain():
    path = "c:/Elysia/Core/Memory/semantic_field.json"
    if not os.path.exists(path):
        print("Brain not found.")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    noise_prefix = "I feel deeply that"
    
    # 1. Sanitize Glossary
    new_glossary = {}
    for key, val in data.get("glossary", {}).items():
        clean_key = key.replace(noise_prefix, "").replace('"', '').replace("'", "").strip()
        # Remove trailing periods or noise
        if clean_key.endswith(":"): clean_key = clean_key[:-1].strip()
        
        # Filter out obvious prompt leaks
        if "analyze this" in clean_key.lower() or "extract the" in clean_key.lower():
            continue
            
        new_glossary[clean_key] = val

    # 2. Sanitize Concepts
    new_concepts = []
    for concept in data.get("concepts", []):
        meaning = concept.get("meaning", "")
        clean_meaning = meaning.replace(noise_prefix, "").replace('"', '').replace("'", "").strip()
        if clean_meaning.endswith(":"): clean_meaning = clean_meaning[:-1].strip()
        
        if "analyze this" in clean_meaning.lower() or "extract the" in clean_meaning.lower():
            continue
            
        concept["meaning"] = clean_meaning
        new_concepts.append(concept)

    data["glossary"] = new_glossary
    data["concepts"] = new_concepts

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Brain Sanitized. Removed conversational noise and prompt leaks.")

if __name__ == "__main__":
    sanitize_json_brain()
