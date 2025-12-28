import os
import json
import re
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup

try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None


DEFAULT_POLICY = {
    "allow_domains": [
        "github.com",
        "docs.python.org",
        "pypi.org",
        "readthedocs.io",
        "wikipedia.org"
    ],
    "deny_domains": [],
    "max_bytes": 262144,  # 256KB
    "timeout_sec": 5,
    "max_redirects": 2,
    "allowed_content_types": [
        "text/html", "text/plain", "application/json"
    ],
    "risk_threshold_block": 0.7,
    "risk_threshold_confirm": 0.4
}


class WebSanctum:
    """
    Objective-text web fetcher with risk/trust scoring.
    Does HEAD → limited GET (text only) → sanitize & extract links → risk/trust → decision.
    """

    def __init__(self, telemetry: Telemetry | None = None, policy_path: str | None = None):
        self.telemetry = telemetry
        self.policy = self._load_policy(policy_path)

    def _load_policy(self, policy_path: str | None) -> Dict[str, Any]:
        path = policy_path or os.path.join('Elysia_Input_Sanctum', 'web_sanctum.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    p = json.load(f)
                # fill defaults
                for k, v in DEFAULT_POLICY.items():
                    p.setdefault(k, v)
                return p
            except Exception:
                pass
        return dict(DEFAULT_POLICY)

    def _domain_ok(self, url: str) -> bool:
        host = urlparse(url).hostname or ''
        host = host.lower()
        if any(host.endswith(d) for d in self.policy.get('deny_domains', [])):
            return False
        allow = self.policy.get('allow_domains', [])
        return any(host.endswith(d) for d in allow) if allow else False

    def _emit(self, event: str, payload: Dict[str, Any]):
        if self.telemetry:
            try:
                self.telemetry.emit(event, payload)
            except Exception:
                pass

    def _sanitize_and_links(self, content_type: str, body: str) -> Tuple[str, List[str], Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        if 'text/html' in content_type:
            soup = BeautifulSoup(body, 'html.parser')
            # remove script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = soup.get_text(' ')
            text = re.sub(r"\s+", " ", text).strip()
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                links.append(href)
            title = soup.title.string.strip() if (soup.title and soup.title.string) else ''
            meta['title'] = title
            return text, links, meta
        if 'application/json' in content_type:
            # return compacted json string
            try:
                o = json.loads(body)
                text = json.dumps(o, ensure_ascii=False)[:self.policy['max_bytes']]
            except Exception:
                text = body[: self.policy['max_bytes']]
            return text, [], meta
        # text/plain or others treated as plain
        return body[: self.policy['max_bytes']], [], meta

    def _risk_score(self, url: str, content_type: str, headers: Dict[str, Any], body: str, links: List[str]) -> float:
        score = 0.0
        host = (urlparse(url).hostname or '').lower()
        # unknown domain increases risk
        if not self._domain_ok(url):
            score += 0.25
        # content-type not allowed
        if not any(ct in content_type for ct in self.policy['allowed_content_types']):
            score += 0.4
        # large body near limit
        if len(body) > self.policy['max_bytes'] * 0.8:
            score += 0.1
        # suspicious links
        sus = sum(1 for h in links if h.startswith('data:') or h.startswith('javascript:'))
        if sus:
            score += min(0.2, 0.05 * sus)
        # if HTML, rough tag counts (scripts removed already, but inline events may remain)
        if 'text/html' in content_type:
            if re.search(r'onclick\s*=|onload\s*=', body, re.I):
                score += 0.1
            if re.search(r'<iframe', body, re.I):
                score += 0.15
        return min(1.0, score)

    def _trust_score(self, url: str, text: str, meta: Dict[str, Any]) -> Tuple[float, List[str]]:
        host = (urlparse(url).hostname or '').lower()
        evidence: List[str] = []
        trust = 0.3
        if any(host.endswith(d) for d in self.policy.get('allow_domains', [])):
            trust += 0.2
            evidence.append('allowlist_domain')
        if meta.get('title'):
            trust += 0.05
            evidence.append('has_title')
        # presence of references/links increases trust slightly
        if text and len(re.findall(r'https?://', text)) >= 2:
            trust += 0.1
            evidence.append('has_links_in_text')
        # presence of dates/numbers
        if re.search(r'\b20\d{2}\b', text):
            trust += 0.05
            evidence.append('has_year')
        return min(1.0, trust), evidence

    def safe_fetch(self, url: str) -> Dict[str, Any]:
        self._emit('web_request_started', {'url': url})
        if not url.lower().startswith('http'):
            self._emit('web_request_blocked', {'url': url, 'reason': 'invalid_scheme'})
            return {'error': 'invalid scheme', 'blocked': True, 'reason': 'invalid_scheme'}
        # HEAD
        try:
            h = requests.head(url, allow_redirects=True, timeout=self.policy['timeout_sec'])
            ct = h.headers.get('content-type', '').split(';')[0].strip().lower()
            redirects = len(getattr(h, 'history', []) or [])
            if redirects > self.policy['max_redirects']:
                self._emit('web_request_blocked', {'url': url, 'reason': 'too_many_redirects'})
                return {'error': 'too many redirects', 'blocked': True, 'reason': 'too_many_redirects'}
        except Exception as e:
            self._emit('web_request_blocked', {'url': url, 'reason': 'head_failed'})
            return {'error': f'head_failed: {e}', 'blocked': True, 'reason': 'head_failed'}

        # Disallow content types not in allowed list
        if not any(allowed in ct for allowed in self.policy['allowed_content_types']):
            self._emit('web_request_blocked', {'url': url, 'reason': 'content_type_disallowed', 'content_type': ct})
            return {'error': 'content type not allowed', 'blocked': True, 'reason': 'content_type_disallowed', 'content_type': ct}

        # GET limited
        try:
            r = requests.get(url, stream=True, timeout=self.policy['timeout_sec'])
            buf = bytearray()
            maxb = int(self.policy['max_bytes'])
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > maxb:
                    break
            body = buf.decode(r.encoding or 'utf-8', errors='replace')
        except Exception as e:
            self._emit('web_request_blocked', {'url': url, 'reason': 'get_failed'})
            return {'error': f'get_failed: {e}', 'blocked': True, 'reason': 'get_failed'}

        text, links, meta = self._sanitize_and_links(ct, body)
        self._emit('web_content_sanitized', {'url': url, 'bytes': len(body), 'content_type': ct})
        if links:
            self._emit('web_links_extracted', {'url': url, 'num_links': len(links)})

        risk = self._risk_score(url, ct, dict(r.headers), body, links)
        trust, evidence = self._trust_score(url, text, meta)
        self._emit('content_trust_evaluated', {'url': url, 'risk': risk, 'trust': trust})

        decision = 'allow'
        if risk >= self.policy['risk_threshold_block']:
            decision = 'block'
            self._emit('web_request_blocked', {'url': url, 'reason': 'risk_high', 'risk': risk})
        elif risk >= self.policy['risk_threshold_confirm']:
            decision = 'confirm'

        # Derive a simple trust label for UI/consumers (content reliability, not safety)
        if trust >= 0.7:
            trust_label = 'substantiated'
        elif trust >= 0.5:
            trust_label = 'partial_evidence'
        else:
            trust_label = 'uncorroborated'

        return {
            'url': url,
            'content_type': ct,
            'sanitized_text': text,
            'links': links,
            'meta': meta,
            'risk_score': risk,
            'trust_score': trust,
            'trust_label': trust_label,
            'evidence': evidence,
            'decision': decision,
        }
