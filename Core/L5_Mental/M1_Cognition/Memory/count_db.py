
import sys
import os
sys.path.append(os.getcwd())
from Core.L1_Foundation.M1_Keystone.Mind.memory_storage import MemoryStorage

storage = MemoryStorage()
count = storage.count_concepts()
print(f"Total concepts in DB: {count}")
storage.close()
