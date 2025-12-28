
import sys
import os
sys.path.append(os.getcwd())
from Core._01_Foundation._02_Logic.Mind.memory_storage import MemoryStorage

storage = MemoryStorage()
count = storage.count_concepts()
print(f"Total concepts in DB: {count}")
storage.close()
