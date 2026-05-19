"""
Fractal Knowledge System (Holographic Storage)
==============================================

"               ,              ."

                (Storage),
      **         (Triple)**            (Digestion).
                    **   (Reconstruction)**   .

[NEW 2025-12-16] Extreme Compression & Logic Safety
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

logger = logging.getLogger("FractalKnowledge")

@dataclass
class KnowledgeTriple:
    """
             :           
    Subject --(Predicate)--> Object
    """
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    
    def __repr__(self):
        return f"[{self.subject}] --({self.predicate})--> [{self.object}]"

class KnowledgeGraph:
    """       (In-memory Graph)"""
    def __init__(self):
        # Index for fast lookup: subject -> list of triples
        self.triples: List[KnowledgeTriple] = []
        self.index: Dict[str, List[KnowledgeTriple]] = {}
        
    def add(self, head: str, relation: str, tail: str, weight: float = 1.0):
        triple = KnowledgeTriple(head, relation, tail, weight)
        self.triples.append(triple)
        
        if head not in self.index:
            self.index[head] = []
        self.index[head].append(triple)
        
    def query(self, subject: str) -> List[KnowledgeTriple]:
        """                """
        return self.index.get(subject, [])

class FractalKnowledgeSeed:
    """
                  
    """
    def __init__(self):
        self.graph = KnowledgeGraph()
        logger.info("  FractalKnowledgeSeed initialized")
        
    def digest(self, text: str):
        """
                             (Prototype NLP)
        """
        sentences = re.split(r'[.?!]\s*', text)
        for sent in sentences:
            if not sent.strip(): continue
            self._extract_triples(sent)
            
    def _extract_triples(self, sentence: str):
        """
                           (Korean/English)
        """
        sent = sentence.strip()
        logger.info(f"Digesting sentence: '{sent}'")
        
        # 1. "A  B  " (Definition)
        # Regex improvement: Allow optional whitespace and non-greedy matching
        match_def = re.match(r'(.+?)[    ]\s*(.+?)[ ]?[  ]$', sent)
        if not match_def:
             # Try simpler pattern "A  B "
             match_def = re.match(r'(.+?)[    ]\s*(.+?)[ ]$', sent)
             
        if match_def:
            subj, obj = match_def.groups()
            logger.info(f" -> Found Def: {subj} = {obj}")
            self.graph.add(subj.strip(), "IsA", obj.strip())
            return

        # 2. "A  B     /    " (Property)
        match_prop = re.match(r'(.+?)[    ]\s+(.+?)[  ]\s+(?:   |    )', sent)
        if match_prop:
            subj, obj = match_prop.groups()
            logger.info(f" -> Found Prop: {subj} has {obj}")
            self.graph.add(subj.strip(), "Has", obj.strip())
            return
            
        # 3. "A  C  B " (Attribute)
        match_attr = re.match(r'(.+?) \s+(.+?)[    ]\s+(.+?)[ ]? ', sent)
        if match_attr:
            subj, attr, obj = match_attr.groups()
            logger.info(f" -> Found Attr: {subj}.{attr} = {obj}")
            self.graph.add(subj.strip(), f"Has{attr}", obj.strip())
            return
        
        logger.warning(f" -> No pattern matched for: '{sent}'")


    def reconstruct(self, subject: str, depth: int = 1) -> str:
        """
                             (     )
        Directional Safety: Only follow outgoing arrows initially.
        """
        triples = self.graph.query(subject)
        if not triples:
            return f"{subject}             ."
        
        description = []
        
        # 1.    (IsA)   
        definitions = [t for t in triples if t.predicate == "IsA"]
        others = [t for t in triples if t.predicate != "IsA"]
        
        for t in definitions:
            description.append(f"{t.subject} ( ) {t.object}   .")
            
            # Recursive depth (Connectivity helping regeneration)
            if depth > 0:
                sub_desc = self.reconstruct(t.object, depth=depth-1)
                if "        " not in sub_desc:
                    description.append(f"   > (  : {sub_desc})")
        
        # 2.      
        for t in others:
            if t.predicate.startswith("Has"):
                attr = t.predicate.replace("Has", "")
                if attr: # HasColor -> Color
                    description.append(f"{t.subject}  {attr} ( ) {t.object}   .")
                else: # Has
                    description.append(f"{t.subject} ( ) {t.object} ( )      .")
            else:
                description.append(f"{t.subject} -> {t.predicate} -> {t.object}.")
                
        return " ".join(description)

    # --- [NEW] Persistence & Visualization ---

    def save_knowledge(self, filepath: str = "data/fractal_knowledge.json"):
        """                (Persistence)"""
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "triples": [
                {
                    "s": t.subject,
                    "p": t.predicate,
                    "o": t.object,
                    "w": t.weight
                }
                for t in self.graph.triples
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"  Knowledge saved to {filepath} ({len(self.graph.triples)} triples)")
        except Exception as e:
            logger.error(f"  Failed to save knowledge: {e}")

    def load_knowledge(self, filepath: str = "data/fractal_knowledge.json"):
        """                """
        import json
        import os
        
        if not os.path.exists(filepath):
            logger.warning(f"   No knowledge file found at {filepath}")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            count = 0
            for item in data.get("triples", []):
                self.graph.add(item['s'], item['p'], item['o'], item.get('w', 1.0))
                count += 1
            
            logger.info(f"  Knowledge loaded from {filepath} ({count} triples)")
        except Exception as e:
            logger.error(f"  Failed to load knowledge: {e}")
            
    def visualize_neighborhood(self, subject: str, depth: int = 2) -> str:
        """
                             (Mermaid-like Tree)
        User Visibility: Shows exactly HOW knowledge is structured.
        """
        lines = [f"  Concept: [{subject}]"]
        
        visited = set()
        
        def _visit(node, current_depth, prefix=""):
            if current_depth > depth: return
            if node in visited: return
            visited.add(node)
            
            triples = self.graph.query(node)
            for i, t in enumerate(triples):
                is_last = (i == len(triples) - 1)
                connector = "   " if is_last else "   "
                
                # Visual: [S] --(P)--> [O]
                lines.append(f"{prefix}{connector} ({t.predicate}) -> [{t.object}]")
                
                # Recurse
                new_prefix = prefix + ("    " if is_last else "    ")
                _visit(t.object, current_depth + 1, new_prefix)
                
        _visit(subject, 1)
        return "\n".join(lines)


# Singleton
_seed = None
def get_fractal_seed() -> FractalKnowledgeSeed:
    global _seed
    if _seed is None:
        _seed = FractalKnowledgeSeed()
    return _seed
