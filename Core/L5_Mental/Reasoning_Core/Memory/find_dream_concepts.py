"""dream      """
import sqlite3

conn = sqlite3.connect('data/Memory/memory.db')
cursor = conn.cursor()

print('dream         :')
cursor.execute("SELECT id FROM concepts WHERE id LIKE '%dream%' LIMIT 30")
results = cursor.fetchall()

print(f'\n  {len(results)}    :\n')
for r in results:
    print(f'  - {r[0]}')

print('\n\nlove         :')
cursor.execute("SELECT id FROM concepts WHERE id LIKE '%love%' LIMIT 30")
results = cursor.fetchall()

print(f'\n  {len(results)}    :\n')
for r in results:
    print(f'  - {r[0]}')

conn.close()
