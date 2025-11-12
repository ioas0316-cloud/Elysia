# CUDA Runtime Stub (CPU fallback)
# 후속 PR에서 cupy/pycuda 백엔드로 교체
class StreamPool: ...
class GraphCache: ...
class MemoryArena: ...
def available():
    try:
        import cupy  # noqa
        return True
    except Exception:
        return False
