"""
   1:                    
=========================================

DB      /                     
                  .
"""

import sqlite3
import json
from collections import defaultdict
from typing import Dict, List, Set

print('=' * 70)
print('            ')
print('=' * 70)

conn = sqlite3.connect('memory.db')
cursor = conn.cursor()

#            (  )
cursor.execute('SELECT id FROM concepts LIMIT 50000')
all_concepts = [row[0] for row in cursor.fetchall()]

print(f'\n        : {len(all_concepts):,}\n')

#        
categories = {
    'foundational': [],      #         
    'composite': [],         #       (A with B)
    'transformative': [],    #       (A becomes B)
    'relational': [],        #       (type:name)
    'junk': [],             #        
}

#      
foundational_words = {
    'love', 'dream', 'truth', 'chaos', 'order', 'beauty', 'void',
    'light', 'dark', 'life', 'death', 'time', 'space', 'atom',
    'soul', 'spirit', 'body', 'mind', 'consciousness', 'freedom',
    'wisdom', 'knowledge', 'emotion', 'thought', 'being', 'nothing',
    'creation', 'destruction', 'birth', 'end', 'infinity', 'god'
}

transformation_words = {'becomes', 'transcends', 'transforms'}
composite_words = {'with', 'beyond', 'in', 'of', 'without', 'through'}
relation_pattern = ':'

print('    ...')
for i, concept in enumerate(all_concepts):
    concept_lower = concept.lower()
    
    # Junk   
    if concept.startswith("Daddy's_") or concept.startswith('Book:'):
        categories['junk'].append(concept)
    
    # Relational (type:name)
    elif relation_pattern in concept:
        categories['relational'].append(concept)
    
    # Transformative
    elif any(word in concept_lower for word in transformation_words):
        categories['transformative'].append(concept)
    
    # Composite
    elif any(word in concept_lower for word in composite_words):
        categories['composite'].append(concept)
    
    # Foundational
    elif any(word == concept_lower or concept_lower.startswith(word + ' ') 
             for word in foundational_words):
        categories['foundational'].append(concept)
    
    # Simple (might be foundational or junk)
    else:
        # If it's a single clean word, might be foundational
        if ' ' not in concept and len(concept) > 2 and concept.isalnum():
            categories['foundational'].append(concept)
        else:
            categories['junk'].append(concept)
    
    if (i + 1) % 10000 == 0:
        print(f'    : {i+1:,} / {len(all_concepts):,}')

print('\n' + '=' * 70)
print('       ')
print('=' * 70)

total_classified = sum(len(concepts) for concepts in categories.values())
for category, concepts in categories.items():
    percentage = len(concepts) / total_classified * 100
    print(f'\n{category:15s}: {len(concepts):6,}  ({percentage:5.1f}%)')
    
    #      
    if concepts:
        print(f'    : {", ".join(concepts[:5])}')

# Foundational         
print('\n' + '=' * 70)
print('  Foundational      ')
print('=' * 70)

#       
foundational_freq = defaultdict(int)
for concept in categories['foundational']:
    base_word = concept.lower().split()[0] if ' ' in concept else concept.lower()
    foundational_freq[base_word] += 1

print('\n   Foundational    (  ):')
for i, (word, count) in enumerate(sorted(foundational_freq.items(), 
                                         key=lambda x: x[1], reverse=True)[:30], 1):
    print(f'  {i:2d}. {word:15s}: {count:4d} ')

#   
output = {
    'statistics': {cat: len(concepts) for cat, concepts in categories.items()},
    'foundational_core': list(foundational_freq.keys())[:50],  # Top 50
    'samples': {cat: concepts[:100] for cat, concepts in categories.items()}
}

with open('concept_categories.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f'\n       : concept_categories.json')

# Composite      
print('\n' + '=' * 70)
print('  Composite      ')
print('=' * 70)

composite_patterns = defaultdict(int)
for concept in categories['composite']:
    for word in composite_words:
        if word in concept.lower():
            composite_patterns[word] += 1
            break

print('\n         :')
for connector, count in sorted(composite_patterns.items(), key=lambda x: x[1], reverse=True):
    print(f'  "{connector}": {count:,} ')

print('\nComposite   :')
for concept in categories['composite'][:15]:
    print(f'  - {concept}')

# Transformative   
print('\n' + '=' * 70)
print('  Transformative   ')
print('=' * 70)

print('\nTransformative   :')
for concept in categories['transformative'][:15]:
    print(f'  - {concept}')

# Relational   
print('\n' + '=' * 70)
print('  Relational   ')
print('=' * 70)

relational_types = defaultdict(int)
for concept in categories['relational']:
    rel_type = concept.split(':')[0] if ':' in concept else 'unknown'
    relational_types[rel_type] += 1

print('\n        :')
for rel_type, count in sorted(relational_types.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f'  {rel_type}: {count:,} ')

print('\nRelational   :')
for concept in categories['relational'][:15]:
    print(f'  - {concept}')

conn.close()

print('\n' + '=' * 70)
print('       ')
print('=' * 70)
print(f'\n     :             (   ,   )   ')
