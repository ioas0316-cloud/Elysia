# This file contains the expanded WisdomCortex.
import re
from kg_manager import KGManager


class WisdomCortex:
    def __init__(self):
        self.kg = KGManager()

    def read_and_digest(self, filepath):
        """
        Reads a text file and extracts knowledge using lightweight, rule-based
        heuristics. Also detects simple causal patterns (e.g., 'X causes Y') and
        merges the extracted knowledge into the project knowledge graph.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"[{self.__class__.__name__}] The source of wisdom could not be found.")
            return None

        knowledge = {"nodes": set(), "edges": []}
        # Split into naive sentences
        sentences = re.split(r'[\.\n]+', text)

        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue

            # Heuristic: "X causes Y" or "X cause Y" or "X led to Y"
            m = re.search(r"([A-Za-z0-9_\s]+?)\s+(causes|cause|led to|leads to|results in|results from)\s+([A-Za-z0-9_\s]+)", s, re.IGNORECASE)
            if m:
                subj = m.group(1).strip().lower()
                rel = 'causes'
                obj = m.group(3).strip().lower()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': rel})
                continue

            #            
            m = re.search(r"([\w\s - ]+?)\s*(   |     |    |  |   |  |   |  |  | )\s*([\w\s - ]+)", s)
            if m:
                obj = m.group(1).strip()
                rel = 'causes'
                subj = m.group(3).strip()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': rel})
                continue
                
            #          
            m = re.search(r"([\w\s - ]+?)\s*(  | |  |   |  |  | |  )\s*([\w\s - ]+)", s)
            if m:
                subj = m.group(1).strip()
                rel = 'causes'
                obj = m.group(3).strip()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': rel})
                continue

            # Heuristic: "X is Y"
            m = re.search(r"(\w[\w\s\-]+?)\s+is\s+([^,;]+)", s, re.IGNORECASE)
            if m:
                subj = m.group(1).strip().lower()
                obj = m.group(2).strip().lower()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'is'})
                continue
                
            #     is   
            m = re.search(r"([\w\s - ]+?)\s*( | | | )\s*([\w\s - ]+?)(?:  | |   |   | |  | |  | |  |  |  |  ~| | ~)", s)
            if m:
                subj = m.group(1).strip()
                obj = m.group(3).strip()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'is'})
                continue
                
            #          
            m = re.search(r"([\w\s - ]+?)\s*( )\s*([\w\s - ]+?)(?: | | | )\s*([\w\s - ]+)", s)
            if m:
                subj = m.group(1).strip()
                prop = m.group(3).strip()
                val = m.group(4).strip()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(val)
                knowledge['edges'].append({'from': subj, 'to': val, 'relation': prop})
                continue

            # Heuristic: "X involves Y"
            m = re.search(r"([\w\s]+?)\s+involves\s+(.+)", s, re.IGNORECASE)
            if m:
                subj = m.group(1).strip().lower()
                obj = m.group(2).strip().lower()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'involves'})

            # Korean heuristics
            # Pattern: "X  Y  / /   " -> X is Y
            m = re.search(r"([ - A-Za-z0-9_\s\-]+?)(?: | | | )\s+(.+?)(?:  |   | |   \.| \.|$)", s)
            if m:
                subj = m.group(1).strip().lower()
                obj = m.group(2).strip().lower()
                # avoid overly short captures
                if len(subj) > 0 and len(obj) > 0 and subj != obj:
                    knowledge['nodes'].add(subj)
                    knowledge['nodes'].add(obj)
                    knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'is'})
                    continue

            # Korean:   /    /      -> involves
            m = re.search(r"([ - A-Za-z0-9_\s\-]+?)\s+(  |    |     |    )\s+(.+)", s)
            if m:
                subj = m.group(1).strip().lower()
                obj = m.group(3).strip().lower()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'involves'})
                continue

            # Korean causal cues:     ,     ,       ,         
            m = re.search(r"([ - A-Za-z0-9_\s\-]+?)\s+(  |    |  |    |     |       )\s+(.+)", s)
            if m:
                subj = m.group(1).strip().lower()
                obj = m.group(3).strip().lower()
                knowledge['nodes'].add(subj)
                knowledge['nodes'].add(obj)
                knowledge['edges'].append({'from': subj, 'to': obj, 'relation': 'causes'})
                continue

        # Convert node set to list
        knowledge['nodes'] = list(knowledge['nodes'])

        # Merge into KG with provenance
        provenance = {'source': filepath}
        if knowledge['nodes'] or knowledge['edges']:
            self.kg.merge_knowledge(knowledge, provenance=provenance)
            return knowledge
        return None

    def parse_sentence(self, sentence):
        match = re.search(r"(\w+)\s+is\s+(.*)", sentence)
        if match:
            return {"subject": match.group(1), "object": match.group(2).strip()}
        return None