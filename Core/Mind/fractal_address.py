"""
Utility to generate and attach fractal addresses to concepts/nodes.
Address format: ROOT/<trunk>/<branch>/<leaf>...
"""

from typing import List


def make_address(path: List[str]) -> str:
    cleaned = [p for p in path if p and p != "ROOT"]
    return "ROOT/" + "/".join(cleaned) if cleaned else "ROOT"
