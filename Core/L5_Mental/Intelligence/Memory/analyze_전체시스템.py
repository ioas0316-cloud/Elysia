"""
    99.6%           ?
================================

        : 13,501  (0.4%)
   : 3,140,902  (99.6%)

            .
"""

import sqlite3
from collections import Counter
import re

conn = sqlite3.connect('data/Memory/memory.db')
cursor = conn.cursor()

print('=' * 70)
print('      99.6%      ')
print('=' * 70)

#    : 10      
cursor.execute('SELECT id FROM concepts LIMIT 100000')
sample_concepts = [row[0] for row in cursor.fetchall()]

print(f'\n     : {len(sample_concepts):,} ')

#      
patterns = {
    "Daddy's_  ": 0,
    "Book:   ": 0,
    "     ": 0,
    "     ": 0,
    "       ": 0,
    "        ": 0,
    "        ": 0,
    "  ": 0
}

#      
daddy_samples = []
book_samples = []
word_samples = []
number_samples = []
special_samples = []
long_samples = []
meaningful_samples = []
other_samples = []

print('\n    ...')
for i, concept in enumerate(sample_concepts):
    # Daddy's   
    if concept.startswith("Daddy's_"):
        patterns["Daddy's_  "] += 1
        if len(daddy_samples) < 20:
            daddy_samples.append(concept)
    
    # Book   
    elif concept.startswith("Book:"):
        patterns["Book:   "] += 1
        if len(book_samples) < 20:
            book_samples.append(concept)
    
    #      
    elif re.search(r'\d{5,}', concept):  # 5        
        patterns["     "] += 1
        if len(number_samples) < 20:
            number_samples.append(concept)
    
    #         
    elif len(concept) > 100:
        patterns["        "] += 1
        if len(long_samples) < 20:
            long_samples.append(concept[:100] + '...')
    
    #        
    elif len(re.findall(r'[^\w\s:]', concept)) > 5:
        patterns["       "] += 1
        if len(special_samples) < 20:
            special_samples.append(concept)
    
    #       (      )
    elif ' ' not in concept and len(concept) <= 20 and concept.isalnum():
        patterns["     "] += 1
        if len(word_samples) < 30:
            word_samples.append(concept)
    
    #         
    elif any(word in concept.lower() for word in ['is', 'and', 'or', 'the', 'of', 'in', 'to']):
        patterns["        "] += 1
        if len(meaningful_samples) < 30:
            meaningful_samples.append(concept)
    
    #   
    else:
        patterns["  "] += 1
        if len(other_samples) < 30:
            other_samples.append(concept)
    
    if (i + 1) % 20000 == 0:
        print(f'    : {i+1:,} / {len(sample_concepts):,}')

print('\n' + '=' * 70)
print('       ')
print('=' * 70)

total = sum(patterns.values())
for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
    percentage = count / total * 100
    bar = ' ' * int(percentage / 2)
    print(f'\n{pattern:20s}: {count:6,}  ({percentage:5.1f}%) {bar}')

#      
print('\n' + '=' * 70)
print('        ')
print('=' * 70)

if daddy_samples:
    print(f"\n[Daddy's_     ] ({len(daddy_samples)}    ):")
    for s in daddy_samples[:15]:
        print(f'  - {s}')

if book_samples:
    print(f"\n[Book:      ] ({len(book_samples)}    ):")
    for s in book_samples[:15]:
        print(f'  - {s}')

if word_samples:
    print(f"\n[        ] ({len(word_samples)}    ):")
    for s in word_samples[:20]:
        print(f'  - {s}')

if number_samples:
    print(f"\n[        ] ({len(number_samples)}    ):")
    for s in number_samples[:15]:
        print(f'  - {s}')

if special_samples:
    print(f"\n[       ] ({len(special_samples)}    ):")
    for s in special_samples[:15]:
        print(f'  - {s}')

if long_samples:
    print(f"\n[        ] ({len(long_samples)}    ):")
    for s in long_samples[:10]:
        print(f'  - {s}')

if meaningful_samples:
    print(f"\n[        ] ({len(meaningful_samples)}    ):")
    for s in meaningful_samples[:20]:
        print(f'  - {s}')

if other_samples:
    print(f"\n[  ] ({len(other_samples)}    ):")
    for s in other_samples[:20]:
        print(f'  - {s}')

#         
print('\n' + '=' * 70)
print('     DB   ')
print('=' * 70)

total_concepts = 3154403
for pattern, count in patterns.items():
    estimated = int(count / total * total_concepts)
    print(f'{pattern:20s}:   {estimated:,} ')

#      
print('\n' + '=' * 70)
print('       ')
print('=' * 70)

value_assessment = {
    "Daddy's_  ": "      -         ",
    "Book:   ": "       -        ",
    "     ": "       - ID        ",
    "       ": "     -      ",
    "        ": "       -         ?",
    "     ": "        -      ",
    "        ": "        -    /      ",
    "  ": "     -       "
}

for pattern, assessment in value_assessment.items():
    count = patterns[pattern]
    percentage = count / total * 100
    print(f'\n{pattern:20s} ({percentage:5.1f}%)')
    print(f'    : {assessment}')

#     
print('\n' + '=' * 70)
print('      ')
print('=' * 70)

worthless = patterns["Daddy's_  "]
low_value = patterns["Book:   "] + patterns["     "] + patterns["        "]
valuable = patterns["     "] + patterns["        "]

print(f'\n      : {worthless:,}  ({worthless/total*100:.1f}%)')
print(f'         ')

print(f'\n      : {low_value:,}  ({low_value/total*100:.1f}%)')
print(f'           (      ?)')

print(f'\n        : {valuable:,}  ({valuable/total*100:.1f}%)')
print(f'          ')

print(f'\n     :')
estimated_worthless = int(worthless / total * total_concepts)
estimated_low = int(low_value / total * total_concepts)
estimated_valuable = int(valuable / total * total_concepts)

print(f'     :   {estimated_worthless:,}  (     {estimated_worthless/1024/1024:.1f}MB   )')
print(f'     :   {estimated_low:,} ')
print(f'       :   {estimated_valuable:,} ')

conn.close()

print('\n' + '=' * 70)
print('       ')
print('=' * 70)
