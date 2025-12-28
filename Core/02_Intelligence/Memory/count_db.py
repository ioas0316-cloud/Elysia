
import sys
import os
sys.path.append(os.getcwd())
from Core.01_Foundation.05_Foundation_Base.Foundation.Mind.memory_storage import MemoryStorage

storage = MemoryStorage()
count = storage.count_concepts()
print(f"Total concepts in DB: {count}")
storage.close()
