"""
Entity normalizer for Elysia knowledge graph.

Cleans up node text, normalizes words, and merges duplicates to improve KG quality.
Uses lightweight rule-based approach (no external NLP libraries).
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple

KG_PATH = Path('data') / 'kg.json'

def clean_text(text: str) -> str:
    """Basic text cleanup."""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # italic
    
    # Remove table markers and extra spaces
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up punctuation
    text = text.replace('(', ' ').replace(')', ' ')
    text = re.sub(r'[.,;:!?\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip().lower()

def is_subset_text(a: str, b: str) -> bool:
    """Check if text a is effectively contained in b after normalization."""
    a_words = set(clean_text(a).split())
    b_words = set(clean_text(b).split())
    return a_words.issubset(b_words)

def find_duplicate_groups(nodes: List[str]) -> List[Set[str]]:
    """Find groups of duplicate/subset nodes to merge."""
    groups: List[Set[str]] = []
    used = set()
    
    for i, node in enumerate(nodes):
        if node in used:
            continue
        
        group = {node}
        used.add(node)
        
        # Look for duplicates/subsets
        for j, other in enumerate(nodes):
            if i != j and other not in used:
                if is_subset_text(node, other) or is_subset_text(other, node):
                    group.add(other)
                    used.add(other)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups

def choose_canonical(group: Set[str]) -> str:
    """Choose the best representative from a group of duplicates."""
    # Prefer longer, more informative strings
    return max(group, key=lambda x: len(clean_text(x).split()))

def update_edge(edge: Dict, old_to_new: Dict[str, str]) -> Dict:
    """Update edge endpoints using old->new mapping."""
    if edge['from'] in old_to_new:
        edge['from'] = old_to_new[edge['from']]
    if edge['to'] in old_to_new:
        edge['to'] = old_to_new[edge['to']]
    return edge

def normalize_kg() -> Tuple[int, int]:
    """
    Load KG, normalize entities, and save back. Returns (merged_count, updated_edges).
    """
    if not KG_PATH.exists():
        return (0, 0)
    
    with open(KG_PATH, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    
    old_node_count = len(kg['nodes'])
    
    # Find duplicate groups
    groups = find_duplicate_groups(kg['nodes'])
    
    if not groups:
        return (0, 0)
    
    # Build old->new mapping
    old_to_new = {}
    for group in groups:
        canonical = choose_canonical(group)
        for node in group:
            if node != canonical:
                old_to_new[node] = canonical
    
    # Update nodes list (remove old, keep canonical)
    kg['nodes'] = [n for n in kg['nodes'] if n not in old_to_new]
    
    # Update edges
    updated_edges = 0
    new_edges = []
    seen_edges = set()  # track unique (from,to,rel) to remove duplicates
    
    for edge in kg['edges']:
        updated = update_edge(edge.copy(), old_to_new)
        key = (updated['from'], updated['to'], updated.get('relation',''))
        if key not in seen_edges:
            new_edges.append(updated)
            seen_edges.add(key)
            if updated != edge:
                updated_edges += 1
    
    kg['edges'] = new_edges
    
    with open(KG_PATH, 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)
    
    return (old_node_count - len(kg['nodes']), updated_edges)

if __name__ == '__main__':
    merged, updated = normalize_kg()
    print(f'Merged {merged} nodes and updated {updated} edges')