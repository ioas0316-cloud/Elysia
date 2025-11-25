import os
from typing import Dict, Any


def read_file_safe(filepath: str, max_bytes: int = 1024 * 1024) -> Dict[str, Any]:
    """
    Safely reads a text file with size limit.
    Returns {'content': str, 'truncated': bool} or {'error': str}
    """
    try:
        if not os.path.exists(filepath):
            return {'error': f'File not found: {filepath}'}
        size = os.path.getsize(filepath)
        if size > max_bytes:
            return {'error': f'File too large ({size} bytes) > limit {max_bytes}'}
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            data = f.read()
        return {'content': data, 'truncated': False}
    except Exception as e:
        return {'error': str(e)}

