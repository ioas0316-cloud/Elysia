import os
import json
import csv

def build_natural_lexicon():
    tsv_path = r"c:\Elysia\data\kengdic.tsv"
    json_path = r"c:\Elysia\data\natural_lexicon.json"
    
    if not os.path.exists(tsv_path):
        print(f"Error: {tsv_path} not found.")
        return
        
    lexicon = {}
    
    # KENGDIC TSV columns: ID, word, hanja, meaning...
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i == 0 or len(row) < 4:
                continue # Skip header or invalid rows
                
            word = row[1].strip()
            meaning = row[3].strip()
            
            # Skip invalid words or very long sentences
            if not word or not meaning or len(word) > 10:
                continue
                
            # Filter pure Korean words (first char is Hangul)
            if not (0xAC00 <= ord(word[0]) <= 0xD7A3):
                continue
                
            # Keep meaningful English definitions
            clean_meaning = meaning.replace("/", " ").replace(";", " ").strip()
            if not clean_meaning:
                continue
                
            # Overwrite if exists, keep it simple
            lexicon[word] = clean_meaning
            
    print(f"Loaded {len(lexicon)} natural words from KENGDIC.")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {json_path}")

if __name__ == "__main__":
    build_natural_lexicon()
