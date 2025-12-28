"""
Cell: 초차원 지능의 기본 단위
==============================
"""

_registry = {}

def Cell(identity: str, category: str = "General"):
    def decorator(cls):
        _registry[identity] = cls
        # print(f"🧬 Cell registered: {identity} ({cls.__name__})")
        return cls
    return decorator

def get_registry():
    return _registry
