import json
import os
import re
from typing import List, Dict, Tuple


# Basic Korean-aware preprocessing without external libs.
# Goal: separate content words from particles(Josa) and endings(Eomi)
# to mimic how a child learns words vs. grammatical glue.

KOREAN_WORD_RE = re.compile(r"[가-힣A-Za-z0-9_]+")

# Common particles (조사) and endings (어미). This is not exhaustive but practical.
JOsa = [
    "은", "는", "이", "가", "을", "를", "에", "에서", "에게", "께", "께서",
    "으로", "로", "과", "와", "도", "만", "까지", "부터", "밖에", "마다", "처럼", "같이",
    "보다", "께", "한테", "에게서", "부터", "라도", "라도",
]
EOMI = [
    "다", "요", "이다", "입니다", "였어요", "이에요", "예요", "어요", "아요",
    "했다", "했다가", "한다", "하고", "하며", "하는", "하려고", "하려면", "할게요", "해요",
]

# Longer strings should match before shorter ones
JOsa.sort(key=len, reverse=True)
EOMI.sort(key=len, reverse=True)


def load_synonyms(path: str = os.path.join("data", "lexicon", "synonyms_ko.json")) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            # map both directions conservatively if specified as list
            mapping: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    mapping[k] = v
                elif isinstance(v, list) and v:
                    base = v[0]
                    for term in [k, *v]:
                        mapping[term] = base
            return mapping
    except Exception:
        return {}


def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u200b", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[str]:
    t = normalize(text)
    parts = re.split(r"([.!?\n]+)", t)
    out: List[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if re.match(r"[.!?\n]+", p):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def segment_token(tok: str) -> List[Tuple[str, str]]:
    """Segment a single token into [(form, tag)] where tag in {word,josa,eomi,number}.
    Very lightweight heuristic: strip trailing josa/eomi greedily.
    """
    original = tok
    if tok.isdigit():
        return [(tok, "number")]
    # strip endings and particles
    segs: List[Tuple[str, str]] = []
    # keep peeling recognized tails
    tail = True
    while tail:
        tail = False
        for e in EOMI:
            if tok.endswith(e) and len(tok) > len(e):
                tok, tailpart = tok[: -len(e)], tok[-len(e) :]
                segs.insert(0, (tailpart, "eomi"))
                tail = True
                break
        if tail:
            continue
        for j in JOsa:
            if tok.endswith(j) and len(tok) > len(j):
                tok, tailpart = tok[: -len(j)], tok[-len(j) :]
                segs.insert(0, (tailpart, "josa"))
                tail = True
                break
    # remaining base
    if tok:
        segs.insert(0, (tok, "word"))
    return segs


def segment_text(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for w in KOREAN_WORD_RE.findall(text or ""):
        for seg in segment_token(w):
            out.append(seg)
    return out


def extract_content_tokens(text: str, synonyms: Dict[str, str] | None = None) -> List[str]:
    """Return content tokens (base words) with synonyms folded, excluding josa/eomi.
    English is lowercased, Korean left as-is for readability.
    """
    syn = synonyms or load_synonyms()
    tokens: List[str] = []
    for form, tag in segment_text(text):
        if tag != "word":
            continue
        base = form.lower()
        base = syn.get(base, base)
        tokens.append(base)
    return tokens

