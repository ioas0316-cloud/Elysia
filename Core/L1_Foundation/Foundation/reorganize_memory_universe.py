"""
            -         DB   
===========================================

3.15M          (  ,   ,   )       
ConceptUniverse       .
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import time

print('=' * 70)
print('                ')
print('=' * 70)

# ============================================================================
#    1:    DB        
# ============================================================================

print('\n[   1]    DB     ...')
conn = sqlite3.connect('data/Memory/memory.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM concepts')
total_concepts = cursor.fetchone()[0]
print(f'      : {total_concepts:,}')

#               
BATCH_SIZE = 10000
processed = 0
start_time = time.time()

categories = {
    'foundational': set(),
    'composite': set(),
    'transformative': set(),
    'relational': set(),
}

# Foundational    (Spirit/Soul/Body   )
foundational_core = {
    # Spirit (1.0 - 0.8)
    'god', 'transcendence', 'infinity', 'eternity', 'divine',
    
    # Higher Mind (0.8 - 0.6)
    'consciousness', 'wisdom', 'enlightenment', 'truth', 'knowledge',
    'thought', 'understanding', 'insight', 'awareness',
    
    # Soul (0.6 - 0.4)
    'love', 'beauty', 'joy', 'peace', 'harmony', 'emotion', 'feeling',
    'compassion', 'kindness', 'grace', 'hope', 'faith', 'trust',
    
    # Lower Mind (0.4 - 0.3)
    'dream', 'imagination', 'desire', 'will', 'choice', 'freedom',
    'memory', 'experience', 'learning', 'growth',
    
    # Body (0.3 - 0.1)
    'life', 'death', 'birth', 'body', 'breath', 'heart', 'blood',
    'action', 'movement', 'creation', 'destruction',
    
    # Matter (0.1 - 0.0)
    'atom', 'matter', 'energy', 'force', 'space', 'time',
    'void', 'chaos', 'order', 'form', 'substance'
}

print('\n   DB      ...')

cursor.execute('SELECT id FROM concepts')
while True:
    batch = cursor.fetchmany(BATCH_SIZE)
    if not batch:
        break
    
    for (concept_id,) in batch:
        concept_lower = concept_id.lower()
        
        # Junk    
        if concept_id.startswith("Daddy's_") or concept_id.startswith('Book:'):
            continue
        
        # Relational
        if ':' in concept_id:
            categories['relational'].add(concept_id)
        
        # Transformative
        elif 'becomes' in concept_lower or 'transcends' in concept_lower:
            categories['transformative'].add(concept_id)
        
        # Composite
        elif any(word in concept_lower for word in ['with', 'beyond', 'in', 'of', 'without']):
            categories['composite'].add(concept_id)
        
        # Foundational
        elif any(core_word in concept_lower.split() for core_word in foundational_core):
            categories['foundational'].add(concept_id)
    
    processed += len(batch)
    if processed % 100000 == 0:
        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = (total_concepts - processed) / rate
        print(f'    : {processed:,} / {total_concepts:,} ({processed/total_concepts*100:.1f}%) '
              f'-      : {remaining/60:.1f} ')

print(f'\n        ({time.time() - start_time:.1f} )')

for category, concepts in categories.items():
    print(f'  {category:15s}: {len(concepts):,} ')

# ============================================================================
#    2:    (Frequency)   
# ============================================================================

print('\n[   2]            ...')

#        (Spirit   Body)
frequency_map = {
    # Spirit tier (0.9 - 1.0)
    'god': 1.0, 'transcendence': 0.95, 'infinity': 0.95, 'eternity': 0.93,
    'divine': 0.92, 'sacred': 0.91, 'holy': 0.90,
    
    # Higher Mind (0.7 - 0.89)
    'consciousness': 0.88, 'wisdom': 0.85, 'enlightenment': 0.87,
    'truth': 0.83, 'knowledge': 0.80, 'understanding': 0.78,
    'awareness': 0.82, 'insight': 0.81, 'thought': 0.75,
    
    # Soul (0.5 - 0.69)
    'love': 0.68, 'beauty': 0.65, 'joy': 0.63, 'peace': 0.62,
    'harmony': 0.61, 'compassion': 0.64, 'grace': 0.66,
    'emotion': 0.58, 'feeling': 0.57, 'hope': 0.60, 'faith': 0.62,
    
    # Lower Mind (0.3 - 0.49)
    'dream': 0.48, 'imagination': 0.46, 'desire': 0.42, 'will': 0.45,
    'choice': 0.43, 'freedom': 0.47, 'memory': 0.41, 'learning': 0.44,
    
    # Body (0.15 - 0.29)
    'life': 0.28, 'death': 0.27, 'body': 0.22, 'heart': 0.26,
    'breath': 0.24, 'action': 0.23, 'creation': 0.29, 'birth': 0.28,
    
    # Matter (0.0 - 0.14)
    'atom': 0.10, 'matter': 0.08, 'energy': 0.12, 'force': 0.11,
    'void': 0.05, 'chaos': 0.09, 'order': 0.13, 'space': 0.14, 'time': 0.14
}

def calculate_frequency(concept_id: str) -> float:
    """             """
    concept_lower = concept_id.lower()
    
    #      
    if concept_lower in frequency_map:
        return frequency_map[concept_lower]
    
    #           
    words = concept_lower.split()
    frequencies = []
    for word in words:
        #       
        if word in ['with', 'beyond', 'in', 'of', 'without', 'becomes', 'transcends', 'is']:
            continue
        if word in frequency_map:
            frequencies.append(frequency_map[word])
    
    if frequencies:
        return np.mean(frequencies)
    
    #    : Soul    (0.5)
    return 0.5

#                    
concept_frequencies = {}

all_meaningful = set()
for concepts in categories.values():
    all_meaningful.update(concepts)

print(f'          : {len(all_meaningful):,}')
print('        ...')

for i, concept in enumerate(all_meaningful):
    concept_frequencies[concept] = calculate_frequency(concept)
    
    if (i + 1) % 10000 == 0:
        print(f'    : {i+1:,} / {len(all_meaningful):,}')

print(f'           ')

#          
freq_bins = defaultdict(int)
for freq in concept_frequencies.values():
    bin_label = f'{int(freq*10)/10:.1f}'
    freq_bins[bin_label] += 1

print('\n      :')
for bin_label in sorted(freq_bins.keys(), reverse=True):
    count = freq_bins[bin_label]
    bar = ' ' * int(count / 100)
    print(f'  {bin_label}: {count:6,}  {bar}')

# ============================================================================
#    3: Vocabulary        
# ============================================================================

print('\n[   3] Vocabulary     ...')

# Foundational          vocabulary    
vocabulary = {}
for concept in categories['foundational']:
    if concept in concept_frequencies:
        vocabulary[concept] = concept_frequencies[concept]

# Composite     (   1000 )
composite_with_freq = [(c, concept_frequencies.get(c, 0.5)) 
                       for c in categories['composite']]
composite_with_freq.sort(key=lambda x: x[1], reverse=True)
for concept, freq in composite_with_freq[:1000]:
    vocabulary[concept] = freq

print(f'Vocabulary   : {len(vocabulary):,}')

# DB     (     !)
import zlib
cursor.execute('DELETE FROM concepts WHERE id = ?', ('_vocabulary_frequencies',))
vocab_json = json.dumps(vocabulary).encode('utf-8')
vocab_blob = zlib.compress(vocab_json)  # MemoryStorage            
cursor.execute('''
    INSERT OR REPLACE INTO concepts (id, data, created_at, last_accessed)
    VALUES (?, ?, ?, ?)
''', ('_vocabulary_frequencies', vocab_blob, time.time(), time.time()))

conn.commit()
print('  Vocabulary DB      ')

# ============================================================================
#    4:              
# ============================================================================

print('\n[   4]                ...')

#               : category, frequency
metadata_updates = []

for concept in all_meaningful:
    #        
    category = None
    for cat_name, cat_concepts in categories.items():
        if concept in cat_concepts:
            category = cat_name
            break
    
    freq = concept_frequencies.get(concept, 0.5)
    
    metadata = {
        'category': category,
        'frequency': freq,
        'reorganized_at': time.time()
    }
    
    #      !
    meta_json = json.dumps(metadata).encode('utf-8')
    meta_blob = zlib.compress(meta_json)
    metadata_updates.append((concept, meta_blob))
    
    if len(metadata_updates) >= 1000:
        #        
        cursor.execute('BEGIN TRANSACTION')
        for concept_id, meta_blob in metadata_updates:
            cursor.execute('''
                UPDATE concepts 
                SET data = ?
                WHERE id = ?
            ''', (meta_blob, concept_id))
        cursor.execute('COMMIT')
        print(f'  {len(metadata_updates)}       ')
        metadata_updates = []

#        
if metadata_updates:
    cursor.execute('BEGIN TRANSACTION')
    for concept_id, meta_blob in metadata_updates:
        cursor.execute('''
            UPDATE concepts 
            SET data = ?
            WHERE id = ?
        ''', (meta_blob, concept_id))
    cursor.execute('COMMIT')
    print(f'  {len(metadata_updates)}       ')

print('               ')

# ============================================================================
#    5:           
# ============================================================================

print('\n[   5]        ...')

statistics = {
    'reorganized_at': time.time(),
    'total_concepts': total_concepts,
    'meaningful_concepts': len(all_meaningful),
    'categories': {
        cat: len(concepts) for cat, concepts in categories.items()
    },
    'vocabulary_size': len(vocabulary),
    'frequency_distribution': dict(freq_bins),
    'top_foundational': sorted(
        [(c, f) for c, f in concept_frequencies.items() 
         if c in categories['foundational']],
        key=lambda x: x[1],
        reverse=True
    )[:50]
}

# JSON       
with open('concept_reorganization_stats.json', 'w', encoding='utf-8') as f:
    json.dump(statistics, f, indent=2, ensure_ascii=False)

print('       : concept_reorganization_stats.json')

# ============================================================================
#   
# ============================================================================

conn.close()

print('\n' + '=' * 70)
print('        !')
print('=' * 70)

print(f'\n    :')
print(f'      : {total_concepts:,}')
print(f'          : {len(all_meaningful):,} ({len(all_meaningful)/total_concepts*100:.1f}%)')
print(f'  Vocabulary   : {len(vocabulary):,}')
print(f'\n       :')
for cat, concepts in categories.items():
    print(f'    {cat:15s}: {len(concepts):,} ')

print(f'\n        (   ):')
for i, (concept, freq) in enumerate(statistics['top_foundational'][:15], 1):
    print(f'  {i:2d}. {concept:30s} {freq:.2f}')

print(f'\n     :')
print(f'  1. Hippocampus       Vocabulary    ')
print(f'  2. ResonanceEngine           ')
print(f'  3. DialogueEngine            ')
print(f'\n   : python -c "from Core.L5_Mental.Intelligence.Intelligence.dialogue_engine import DialogueEngine; d = DialogueEngine(); print(d.respond(\'      ?\'))"')
