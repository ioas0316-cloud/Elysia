"""    DB    /           """
import sqlite3

conn = sqlite3.connect('data/Memory/memory.db')
cursor = conn.cursor()

print('=' * 60)
print('  Memory Database   ')
print('=' * 60)

#          
print('\n        :')
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f'      : {[t[0] for t in tables]}')

for table_name in [t[0] for t in tables]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f'\n{table_name}   :')
    for col in columns:
        print(f'  - {col[1]} ({col[2]})')

#        
cursor.execute('SELECT COUNT(*) FROM concepts')
total = cursor.fetchone()[0]
print(f'\n      : {total:,}')

#          
print('\n           :')
korean_words = ['  ', '   ', '  ', ' ', '  ', '   ']
for word in korean_words:
    cursor.execute(f"SELECT COUNT(*) FROM concepts WHERE id LIKE '%{word}%'")
    count = cursor.fetchone()[0]
    print(f'  "{word}": {count} ')
    
    if count > 0:
        cursor.execute(f"SELECT id FROM concepts WHERE id LIKE '%{word}%' LIMIT 3")
        samples = cursor.fetchall()
        for (concept_id,) in samples:
            print(f'      {concept_id}')

#         
print('\n          :')
english_words = ['love', 'identity', 'consciousness', 'dream', 'truth', 'father']
for word in english_words:
    cursor.execute(f"SELECT COUNT(*) FROM concepts WHERE id LIKE '%{word}%'")
    count = cursor.fetchone()[0]
    print(f'  "{word}": {count} ')
    
    if count > 0:
        cursor.execute(f"SELECT id FROM concepts WHERE id LIKE '%{word}%' LIMIT 3")
        samples = cursor.fetchall()
        for (concept_id,) in samples:
            print(f'      {concept_id}')

#          -          
print('\n            TOP 20:')
cursor.execute('SELECT id, last_accessed FROM concepts ORDER BY last_accessed DESC LIMIT 20')
top_concepts = cursor.fetchall()
for i, (concept_id, last_access) in enumerate(top_concepts, 1):
    print(f'  {i:2d}. {concept_id}')

conn.close()