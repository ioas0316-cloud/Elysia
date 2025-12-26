import os
import json
from typing import List, Dict, Any
from datetime import datetime


class AssociativeMemory:
    """
    Lightweight keyword-based associative memory.
    Stores compressed "gists" that can be recalled by keyword overlap.
    File: data/associative_index.json
    """

    def __init__(self, path: str | None = None):
        base = os.path.join('data')
        os.makedirs(base, exist_ok=True)
        self.path = path or os.path.join(base, 'associative_index.json')
        self.index: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_gist(self, keywords: List[str], summary: str, context: Dict[str, Any] | None = None) -> str:
        keywords = [str(k).lower() for k in keywords if k]
        # simple bigram enrichment from keywords sequence
        bigrams = []
        for i in range(len(keywords) - 1):
            bg = f"{keywords[i]}_{keywords[i+1]}"
            bigrams.append(bg)
        enriched_keywords = keywords[:12] + bigrams[:6]
        gist = {
            'id': datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ'),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'keywords': enriched_keywords,
            'summary': summary.strip()[:500],
            'context': context or {},
        }
        self.index.append(gist)
        self._save()
        return gist['id']

    def search(self, query_tokens: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        q = {str(t).lower() for t in query_tokens if t}
        if not q:
            return []
        scored = []
        for item in self.index:
            kws = set(item.get('keywords', []))
            score = len(q.intersection(kws))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]
