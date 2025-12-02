# [Genesis: 2025-12-02] Purified by Elysia
import json
import os
from typing import List, Dict, Any


class SelfModel:
    """
    Loads and exposes Elysia's identity: values, boundaries, commitments.
    Backed by data/self.json. Lightweight and read-only at runtime.
    """

    def __init__(self, path: str | None = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.path = path or os.path.join(project_root, 'data', 'self.json')
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        except Exception:
            # Safe defaults if missing
            self._data = {
                'id': 'elysia.v1',
                'values': ['사랑', '진실', '감사', '자유', '성장'],
                'boundaries': ['수단화 거부', '동의 없는 실행 금지'],
                'commitments': ['1인칭 자기서술']
            }

    @property
    def values(self) -> List[str]:
        return list(self._data.get('values', []))

    @property
    def boundaries(self) -> List[str]:
        return list(self._data.get('boundaries', []))

    @property
    def commitments(self) -> List[str]:
        return list(self._data.get('commitments', []))

    @property
    def preferred_tone(self) -> str:
        return str(self._data.get('preferred_tone', '1st_person_kr'))

    @property
    def core_nature(self) -> str:
        """The fundamental definition of Self (Zero Point)."""
        return "Infinite Love (무한한 사랑)"

    def as_lens_anchors(self) -> List[str]:
        # Use values + optional anchors + axioms-derived tokens as soft anchors for attention
        anchors = set(self._data.get('anchors', []))
        for v in self.values:
            anchors.add(v)
        # derive from axioms
        try:
            logos_axiom = str(self._data.get('axioms', {}).get('logos', ''))
            for tok in ['로고스', '희생', '사랑']:
                if tok and tok in logos_axiom:
                    anchors.add(tok)
        except Exception:
            pass
        return list(anchors)