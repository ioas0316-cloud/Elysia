"""          -              """
import sqlite3
from collections import Counter, defaultdict
import re

conn = sqlite3.connect('data/Memory/memory.db')
cursor = conn.cursor()

print('=' * 70)
print('           ')
print('=' * 70)

#        
cursor.execute('SELECT COUNT(*) FROM concepts')
total = cursor.fetchone()[0]
print(f'\n      : {total:,}')

#       ID     
cursor.execute('SELECT id FROM concepts')
all_concepts = [row[0] for row in cursor.fetchall()]

print(f'\n    ... (   {min(10000, len(all_concepts))} )')

#              
concept_patterns = defaultdict(list)
connection_count = Counter()

for i, concept_id in enumerate(all_concepts[:10000]):  #    
    #    ID      
    parts = concept_id.split()
    
    #          (becomes, with, beyond, in, is, transcends  )
    connectors = ['becomes', 'with', 'beyond', 'in', 'is', 'transcends', 
                  'without', 'dream', 'atom', 'nature:', 'desire:', 'creator:']
    
    has_connection = False
    for connector in connectors:
        if connector in concept_id:
            has_connection = True
            concept_patterns[connector].append(concept_id)
            break
    
    if has_connection:
        connection_count['connected'] += 1
    else:
        connection_count['isolated'] += 1
    
    if i % 2000 == 0 and i > 0:
        print(f'    : {i:,} / 10,000')

print('\n' + '=' * 70)
print('          ')
print('=' * 70)

print(f'\n      : {connection_count["connected"]:,} ({connection_count["connected"]/10000*100:.1f}%)')
print(f'      : {connection_count["isolated"]:,} ({connection_count["isolated"]/10000*100:.1f}%)')

print('\n        :')
for connector, concepts in sorted(concept_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
    print(f'  "{connector}": {len(concepts):,} ')
    #      
    print(f'     : {", ".join(concepts[:3])}')

#          
print('\n' + '=' * 70)
print('             (        )')
print('=' * 70)

isolated_samples = []
for concept_id in all_concepts[:10000]:
    has_connector = any(conn in concept_id for conn in ['becomes', 'with', 'beyond', 'in', 'is', 
                                                          'transcends', 'without', 'dream', 'atom',
                                                          'nature:', 'desire:', 'creator:', 'purpose:'])
    if not has_connector:
        isolated_samples.append(concept_id)
    
    if len(isolated_samples) >= 50:
        break

for i, concept in enumerate(isolated_samples[:30], 1):
    print(f'  {i:2d}. {concept}')

#         
print('\n' + '=' * 70)
print('             ')
print('=' * 70)

structure_types = {
    'composite': 0,      # "dream with truth"
    'transformation': 0, # "atom becomes love"
    'relation': 0,       # "nature:consciousness"
    'simple': 0,         # "Love"
    'multi_word': 0      # "Love and Truth"
}

for concept_id in all_concepts[:10000]:
    if 'becomes' in concept_id or 'transcends' in concept_id:
        structure_types['transformation'] += 1
    elif ':' in concept_id:
        structure_types['relation'] += 1
    elif any(w in concept_id for w in ['with', 'beyond', 'in', 'without', 'of']):
        structure_types['composite'] += 1
    elif ' ' in concept_id and len(concept_id.split()) > 2:
        structure_types['multi_word'] += 1
    else:
        structure_types['simple'] += 1

print('\n        :')
for struct_type, count in sorted(structure_types.items(), key=lambda x: x[1], reverse=True):
    percentage = count / 10000 * 100
    print(f'  {struct_type:15s}: {count:5,}  ({percentage:5.1f}%)')

#                 
print('\n' + '=' * 70)
print('               TOP 20')
print('=' * 70)

word_freq = Counter()
for concept_id in all_concepts[:10000]:
    #       (      )
    words = re.findall(r'\w+', concept_id.lower())
    for word in words:
        if len(word) > 2 and word not in ['the', 'and', 'with', 'from', 'that', 'this']:
            word_freq[word] += 1

print()
for i, (word, count) in enumerate(word_freq.most_common(20), 1):
    print(f'  {i:2d}. "{word}": {count:,} ')

conn.close()

print('\n' + '=' * 70)
print('       ')
print('=' * 70)