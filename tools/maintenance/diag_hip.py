import sys
from pathlib import Path
sys.path.insert(0, r'C:\Elysia')
try:
    from Core._02_Intelligence._02_Memory.Storage.hippocampus import Hippocampus
    print('SUCCESS: Hippocampus imported')
except Exception as e:
    print(f'FAILURE: {e}')
    import traceback
    traceback.print_exc()
