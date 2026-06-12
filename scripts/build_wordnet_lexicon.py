import os
import json
import nltk
from nltk.corpus import wordnet as wn

def build_lexicon():
    print("Downloading WordNet data...")
    nltk.download('wordnet', quiet=True)
    
    lexicon = {}
    print("Building true semantic lexicon...")
    for synset in list(wn.all_synsets()):
        # Get the primary word for this synset
        word = synset.lemmas()[0].name().lower().replace('_', ' ')
        
        # Get the definition
        definition = synset.definition()
        
        # Only keep it if it hasn't been added (or if we want to overwrite with a new sense)
        if word not in lexicon:
            lexicon[word] = definition

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "natural_lexicon.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
        
    print(f"Built true dictionary with {len(lexicon)} words at {out_path}.")

if __name__ == "__main__":
    build_lexicon()
