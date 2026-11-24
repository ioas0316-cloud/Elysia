r"""
ELYSIA Real-World Cultural & Multimodal Expression Ingestor
===========================================================

Discovers and absorbs how genuine human utterances, voice prosody, and visual motion
arise out of real causal context:
    Situation -> Cause -> Process Verbs -> Emotion / Mind -> Intent -> Utterance

Ingests existing books, drama/film scripts, `.srt`/`.vtt` subtitles (paired with
`.wav`/`.mp4` audio-video slices when present), Korean Wikisource classics, and
Project Gutenberg public literature — replacing canned developer templates with
resonance across real human expression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
import wave
import zlib

import numpy as np

from core.embodiment.language_scale_rotor import compute_continuous_spectrum
from core.orchestration.autonomous_open_explorer import _extract_lexical_tokens
from core.orchestration.external_web_bridge import HttpFetcher, default_http_get
from core.topology.structural_scale_rotor import extract_structure_from_prose


def _encode_lexical_64d(text: str) -> np.ndarray:
    """Deterministic 64-D L2-normalized embedding over tokens and character 2-grams."""
    vec = np.zeros(64, dtype=np.float64)
    clean = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not clean:
        return vec

    tokens = _extract_lexical_tokens(clean)
    for tok in tokens:
        idx = zlib.crc32(tok.encode("utf-8")) % 64
        vec[idx] += 2.0

    for i in range(max(0, len(clean) - 1)):
        bg = clean[i : i + 2]
        if bg.isspace():
            continue
        idx = zlib.crc32(bg.encode("utf-8")) % 64
        vec[idx] += 1.0

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def infer_emotion_and_intent(
    situation: str,
    cause: str,
    expression_text: str,
    voice_prosody: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[str], str]:
    """
    Infers continuous chromatic triad `[flux_energy, order_coherence, tension]`,
    physical-structural signals, and continuous spectrum signature from context
    and prosody (zero hardcoded emotion word lists or intent categories).
    """
    rms = float(voice_prosody.get("rms_energy", 0.0)) if voice_prosody else 0.0
    t, r, f, o = compute_continuous_spectrum(situation, cause, expression_text, prosody_rms=rms)
    chroma = [round(f, 3), round(o, 3), round(t, 3)]
    signals: List[str] = []
    if t >= 0.55:
        signals.append("high_tension_open")
    if r >= 0.55:
        signals.append("resolving_shift")
    if f >= 0.55:
        signals.append("high_flux_energy")
    if o >= 0.65:
        signals.append("high_order_coherence")
    if not signals:
        signals.append("balanced_field")
    intent_spec = f"spec_t{t:.1f}_r{r:.1f}_f{f:.1f}"
    return chroma, signals, intent_spec


@dataclass
class SituatedExpressionRotor:
    """Single real-world situated utterance carrying context, causality, emotion, and intent."""

    rotor_id: str
    source_title: str
    source_kind: str  # "book" | "drama_subtitle" | "audio_video_scene" | "user_dialogue"
    situation: str
    cause: str
    process_verbs: List[str]
    emotion_chroma: List[float]
    emotion_signals: List[str]
    intent_purpose: str
    expression_text: str
    voice_prosody: Optional[Dict[str, float]] = None
    visual_motion: Optional[Dict[str, float]] = None
    lexical_vec: List[float] = field(default_factory=list)

    def ensure_vector(self) -> None:
        if not self.lexical_vec or len(self.lexical_vec) != 64:
            vec = _encode_lexical_64d(f"{self.situation} {self.cause} {self.expression_text}")
            self.lexical_vec = [round(float(x), 5) for x in vec]

    def to_dict(self) -> Dict[str, Any]:
        self.ensure_vector()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SituatedExpressionRotor":
        obj = cls(
            rotor_id=str(raw.get("rotor_id", "")),
            source_title=str(raw.get("source_title", "unknown")),
            source_kind=str(raw.get("source_kind", "book")),
            situation=str(raw.get("situation", "")),
            cause=str(raw.get("cause", "")),
            process_verbs=[str(v) for v in raw.get("process_verbs", [])],
            emotion_chroma=[float(c) for c in raw.get("emotion_chroma", [0.33, 0.34, 0.33])],
            emotion_signals=[str(s) for s in raw.get("emotion_signals", [])],
            intent_purpose=str(raw.get("intent_purpose", "express_feeling")),
            expression_text=str(raw.get("expression_text", "")),
            voice_prosody=dict(raw["voice_prosody"]) if raw.get("voice_prosody") else None,
            visual_motion=dict(raw["visual_motion"]) if raw.get("visual_motion") else None,
            lexical_vec=[float(x) for x in raw.get("lexical_vec", [])],
        )
        obj.ensure_vector()
        return obj


class SituatedExpressionField:
    """Persistent associative field indexing real-world situated expressions."""

    MAX_ROTORS = 2000

    def __init__(self, rotors: Sequence[SituatedExpressionRotor] = ()) -> None:
        self.rotors: List[SituatedExpressionRotor] = []
        self._seen_exprs: set[str] = set()
        if rotors:
            self.add_rotors(rotors)

    def add_rotors(self, rotors: Sequence[SituatedExpressionRotor]) -> int:
        added = 0
        for r in rotors:
            clean = r.expression_text.strip()
            if len(clean) < 3:
                continue
            key = clean.lower()
            if key in self._seen_exprs:
                continue
            r.ensure_vector()
            self._seen_exprs.add(key)
            self.rotors.append(r)
            added += 1
        if len(self.rotors) > self.MAX_ROTORS:
            self.rotors = self.rotors[-self.MAX_ROTORS :]
            self._seen_exprs = {it.expression_text.strip().lower() for it in self.rotors}
        return added

    def resonate(
        self,
        query_text: str,
        situation_text: str,
        chroma: Tuple[float, float, float],
        intent_hint: str = "",
        top_k: int = 3,
    ) -> List[Tuple[SituatedExpressionRotor, float, str]]:
        """Finds highest-resonance real-world utterances matching context, chroma, and intent."""
        n = len(self.rotors)
        if n == 0:
            return []

        q_vec = _encode_lexical_64d(f"{situation_text} {query_text}")
        q_norm = float(np.linalg.norm(q_vec))
        L = np.stack([np.asarray(r.lexical_vec, dtype=np.float64) for r in self.rotors], axis=0)
        if q_norm > 1e-8:
            lex_sim = np.clip(L @ q_vec, 0.0, 1.0)
        else:
            lex_sim = np.full(n, 0.25, dtype=np.float64)

        q_chr = np.array(chroma[:3], dtype=np.float64)
        C = np.stack([np.asarray(r.emotion_chroma[:3], dtype=np.float64) for r in self.rotors], axis=0)
        chr_dist = np.linalg.norm(C - q_chr[None, :], axis=1)
        chr_sim = np.clip(1.0 - chr_dist / math.sqrt(3.0), 0.0, 1.0)

        q_tokens = set(_extract_lexical_tokens(f"{situation_text} {query_text}"))
        tok_sim = np.zeros(n, dtype=np.float64)
        intent_bonus = np.zeros(n, dtype=np.float64)
        source_bias = np.zeros(n, dtype=np.float64)

        for i, rot in enumerate(self.rotors):
            r_toks = set(_extract_lexical_tokens(f"{rot.situation} {rot.cause} {rot.expression_text}"))
            if q_tokens and r_toks:
                tok_sim[i] = len(q_tokens & r_toks) / max(1, len(q_tokens | r_toks))
            elif not q_tokens:
                tok_sim[i] = 0.2
            if intent_hint and rot.intent_purpose == intent_hint:
                intent_bonus[i] = 1.0
            if rot.source_kind != "user_dialogue":
                source_bias[i] = 0.04

        scores = 0.56 * lex_sim + 0.22 * tok_sim + 0.12 * chr_sim + 0.10 * intent_bonus + source_bias
        order = np.argsort(-scores)[: max(1, int(top_k))]
        results: List[Tuple[SituatedExpressionRotor, float, str]] = []
        for idx in order:
            rot = self.rotors[int(idx)]
            sc = float(scores[int(idx)])
            why = (
                f"[{rot.source_title} · {rot.intent_purpose}] 상황 '{rot.situation[:36]}'의 "
                f"맥락/어휘/색채 공명(점수={sc:.3f})"
            )
            results.append((rot, round(sc, 4), why))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {"rotors": [r.to_dict() for r in self.rotors]}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SituatedExpressionField":
        rotors = [SituatedExpressionRotor.from_dict(it) for it in raw.get("rotors", [])]
        return cls(rotors=rotors)


def _extract_process_verbs(prose: str) -> List[str]:
    if not prose.strip():
        return []
    st = extract_structure_from_prose("__expr_probe__", domain="expression", prose=prose)
    return list(dict.fromkeys(rel.verb.value for rel in st.process_chain))


def _parse_timestamp_to_s(ts_str: str) -> float:
    clean = ts_str.strip().replace(",", ".")
    parts = clean.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0


def _measure_wav_prosody(wav_path: Path, start_s: float, end_s: float) -> Optional[Dict[str, float]]:
    if not wav_path.is_file():
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if framerate <= 0 or n_frames <= 0:
                return None

            s_frame = max(0, min(n_frames - 1, int(start_s * framerate)))
            e_frame = max(s_frame + 1, min(n_frames, int(max(end_s, start_s + 0.25) * framerate)))
            wf.setpos(s_frame)
            raw_bytes = wf.readframes(e_frame - s_frame)
            if not raw_bytes:
                return None

            if sampwidth == 1:
                samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) - 128.0
                scale = 128.0
            elif sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
                scale = 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float64)
                scale = 2147483648.0
            else:
                return None

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            norm = samples / scale
            if norm.size < 2:
                return None

            rms = float(np.sqrt(np.mean(norm ** 2)))
            signs = np.signbit(norm)
            zcr = float(np.mean(signs[1:] != signs[:-1]))
            return {
                "rms_energy": round(rms, 4),
                "zero_cross_rate": round(zcr, 4),
                "duration_s": round((e_frame - s_frame) / float(framerate), 3),
            }
    except Exception:
        return None


def _measure_video_motion(mp4_path: Path, start_s: float, end_s: float) -> Optional[Dict[str, float]]:
    if not mp4_path.is_file():
        return None
    cap = None
    try:
        import cv2

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s * 1000.0))
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, end_s * 1000.0))
        ok2, f2 = cap.read()
        if not ok1 or f1 is None:
            return None
        a1 = cv2.resize(f1, (32, 24)).astype(np.float64) / 255.0
        motion_flux = 0.0
        if ok2 and f2 is not None:
            a2 = cv2.resize(f2, (32, 24)).astype(np.float64) / 255.0
            motion_flux = float(np.mean(np.abs(a2 - a1)))
        contrast_std = float(a1.std())
        warm_cool = float((a1[..., 2].mean() + 0.1) / (a1[..., 0].mean() + 0.1))  # BGR in cv2
        return {
            "motion_flux": round(motion_flux, 4),
            "contrast_std": round(contrast_std, 4),
            "warm_cool_ratio": round(warm_cool, 4),
        }
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def parse_srt_or_vtt_subtitles(
    path: Path | str,
    media_path: Optional[Path | str] = None,
    max_rotors: int = 80,
) -> List[SituatedExpressionRotor]:
    """Parses `.srt`/`.vtt` subtitles and pairs matching `.wav`/`.mp4` prosody/motion."""
    sub_path = Path(path)
    if not sub_path.is_file():
        return []

    raw_text = sub_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")
    blocks = re.split(r"\r?\n\s*\r?\n", raw_text)
    cues: List[Tuple[float, float, str]] = []
    ts_pat = re.compile(r"([\d:,.]+)\s*-->\s*([\d:,.]+)")

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE")):
            continue
        ts_idx = -1
        m = None
        for idx, ln in enumerate(lines):
            m = ts_pat.search(ln)
            if m:
                ts_idx = idx
                break
        if m is None or ts_idx < 0:
            continue
        start_s = _parse_timestamp_to_s(m.group(1))
        end_s = _parse_timestamp_to_s(m.group(2))
        text_lines = [re.sub(r"<[^>]+>", "", ln).strip() for ln in lines[ts_idx + 1 :]]
        cue_text = " ".join(t for t in text_lines if t).strip()
        if cue_text:
            cues.append((start_s, end_s, cue_text))

    wav_cand: Optional[Path] = None
    mp4_cand: Optional[Path] = None
    if media_path is not None:
        mp = Path(media_path)
        if mp.suffix.lower() == ".wav":
            wav_cand = mp
        elif mp.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov"):
            mp4_cand = mp
    else:
        same_wav = sub_path.with_suffix(".wav")
        same_mp4 = sub_path.with_suffix(".mp4")
        if same_wav.is_file():
            wav_cand = same_wav
        if same_mp4.is_file():
            mp4_cand = same_mp4

    source_kind = "audio_video_scene" if (wav_cand or mp4_cand) else "drama_subtitle"
    rotors: List[SituatedExpressionRotor] = []
    for i, (s_sec, e_sec, expr) in enumerate(cues[:max_rotors]):
        if len(expr) < 3:
            continue
        prev_two = [cues[j][2] for j in range(max(0, i - 2), i)]
        situation = " / ".join(prev_two) if prev_two else f"{sub_path.stem} scene start"
        cause = cues[i - 1][2] if i > 0 else "scene opening"

        prosody = _measure_wav_prosody(wav_cand, s_sec, e_sec) if wav_cand else None
        vis_mot = _measure_video_motion(mp4_cand, s_sec, e_sec) if mp4_cand else None
        chroma, signals, intent = infer_emotion_and_intent(situation, cause, expr, voice_prosody=prosody)
        verbs = _extract_process_verbs(f"{situation} {cause} {expr}")

        rot = SituatedExpressionRotor(
            rotor_id=f"sub::{sub_path.stem}::{i}",
            source_title=sub_path.stem,
            source_kind=source_kind,
            situation=situation[:280],
            cause=cause[:220],
            process_verbs=verbs,
            emotion_chroma=chroma,
            emotion_signals=signals,
            intent_purpose=intent,
            expression_text=expr[:260],
            voice_prosody=prosody,
            visual_motion=vis_mot,
        )
        rot.ensure_vector()
        rotors.append(rot)
    return rotors


_QUOTE_REGEX = re.compile(
    r"(?:“([^”]{4,240})”|「([^」]{4,240})」|『([^』]{4,240})』|\"([^\"]{4,240})\")"
)
_SCRIPT_TURN_REGEX = re.compile(r"^([A-Za-z0-9가-힣 _.-]{1,24}):\s+(.{4,240})$")
_STATUTE_TITLE_RE = re.compile(r"(?:조례|시행령|시행규칙|법률|훈령|예규|규정)$")
_STATUTE_DOC_MARKERS = ("조례", "시행령", "시행규칙", "법률", "훈령", "예규", "분류:", "글로벌 세계 대백과사전", "위키문헌:")


def extract_dialogues_from_literature(
    text: str,
    source_title: str,
    source_kind: str = "book",
    max_rotors: int = 60,
) -> List[SituatedExpressionRotor]:
    """
    Extracts genuine quoted speech and dramatic speaker turns along with their
    preceding narrative Situation and triggering Cause from real literature.
    """
    if not text or not text.strip():
        return []

    rotors: List[SituatedExpressionRotor] = []
    seen_exprs: set[str] = set()
    normalized = re.sub(r"\r\n?", "\n", text)

    # 1. Extract quoted literary dialogue ("...", “...”, 「...」) with preceding context
    last_quote_end = 0
    for match in _QUOTE_REGEX.finditer(normalized):
        expr = next((g for g in match.groups() if g), "").strip()
        if not expr or len(expr) < 4:
            continue
        low = expr.lower()
        if low in seen_exprs or "http://" in low or "https://" in low:
            continue
        if _STATUTE_TITLE_RE.search(expr.rstrip(".)")):
            continue
        seen_exprs.add(low)

        start = match.start()
        nl_bound = normalized.rfind("\n", 0, start) + 1
        left_bound = max(0, start - 280, last_quote_end, nl_bound)
        ctx_raw = normalized[left_bound:start].strip()
        if not ctx_raw:
            # Immediate back-to-back quote: keep previous turn or narrow line slice cleanly
            ctx_raw = normalized[max(0, start - 160, last_quote_end) : start].strip()
        last_quote_end = match.end()
        # Strip incomplete fragment at the left boundary
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。\n])\s+", ctx_raw) if s.strip()]
        situation = " ".join(sentences[-3:])[-240:] if sentences else f"{source_title} narrative"
        cause = sentences[-1][-180:] if sentences else situation

        chroma, signals, intent = infer_emotion_and_intent(situation, cause, expr)
        verbs = _extract_process_verbs(f"{situation} {expr}")
        rot = SituatedExpressionRotor(
            rotor_id=f"lit::{len(rotors)}_{zlib.crc32(expr.encode('utf-8')) % 100000}",
            source_title=source_title,
            source_kind=source_kind,
            situation=situation,
            cause=cause,
            process_verbs=verbs,
            emotion_chroma=chroma,
            emotion_signals=signals,
            intent_purpose=intent,
            expression_text=expr,
        )
        rot.ensure_vector()
        rotors.append(rot)
        if len(rotors) >= max_rotors:
            return rotors

    # 2. Extract play/script turn lines (Speaker: dialogue) when sparse quoted speech
    if len(rotors) < max_rotors:
        lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
        prev_turns: List[str] = []
        for ln in lines:
            m = _SCRIPT_TURN_REGEX.match(ln)
            if not m:
                prev_turns.append(ln)
                prev_turns = prev_turns[-3:]
                continue
            speaker, expr = m.group(1).strip(), m.group(2).strip()
            low = expr.lower()
            if low in seen_exprs or speaker.lower() in ("http", "https", "title", "author"):
                continue
            seen_exprs.add(low)

            situation = f"[{speaker}] " + (" / ".join(prev_turns[-2:]) if prev_turns else source_title)
            cause = prev_turns[-1] if prev_turns else source_title
            chroma, signals, intent = infer_emotion_and_intent(situation, cause, expr)
            verbs = _extract_process_verbs(f"{situation} {expr}")

            rot = SituatedExpressionRotor(
                rotor_id=f"script::{len(rotors)}_{zlib.crc32(expr.encode('utf-8')) % 100000}",
                source_title=source_title,
                source_kind=source_kind,
                situation=situation[:240],
                cause=cause[:180],
                process_verbs=verbs,
                emotion_chroma=chroma,
                emotion_signals=signals,
                intent_purpose=intent,
                expression_text=expr,
            )
            rot.ensure_vector()
            rotors.append(rot)
            prev_turns.append(ln)
            prev_turns = prev_turns[-3:]
            if len(rotors) >= max_rotors:
                break

    return rotors


def fetch_public_literature_from_web(
    lang: str = "ko",
    query: str = "",
    max_works: int = 2,
    fetcher: Optional[HttpFetcher] = None,
) -> List[SituatedExpressionRotor]:
    """
    Fetches real public Korean literature from Korean Wikisource MediaWiki API,
    or English drama/novels from Project Gutenberg (Gutendex), and extracts
    situated `(Situation, Cause, Emotion, Intent, Utterance)` rotors.
    """
    get_fn = fetcher or default_http_get
    collected: List[SituatedExpressionRotor] = []
    clean_lang = (lang or "ko").strip().lower()

    if clean_lang == "ko":
        search_terms = [query.strip()] if query.strip() else ["희곡", "소설"]
        visited_titles: set[str] = set()
        works_loaded = 0

        for term in search_terms:
            if works_loaded >= max_works:
                break
            sr_q = f'"{term}"' if " " in term else term
            search_url = (
                "https://ko.wikisource.org/w/api.php?action=query&list=search"
                f"&srsearch={quote(sr_q)}&srlimit=5&format=json"
            )
            st, raw_b, _ = get_fn(search_url, 8.0, {"Accept": "application/json"})
            if st != 200 or not raw_b:
                continue
            try:
                sjson = json.loads(raw_b.decode("utf-8", errors="replace"))
                hits = (sjson.get("query") or {}).get("search") or []
            except Exception:
                continue

            for hit in hits:
                title = str(hit.get("title") or "").strip()
                if not title or title in visited_titles:
                    continue
                if any(m in title for m in _STATUTE_DOC_MARKERS):
                    continue
                visited_titles.add(title)
                ext_url = (
                    "https://ko.wikisource.org/w/api.php?action=query&prop=extracts"
                    f"&explaintext=1&titles={quote(title)}&format=json"
                )
                est, eraw, _ = get_fn(ext_url, 10.0, {"Accept": "application/json"})
                if est != 200 or not eraw:
                    continue
                try:
                    ejson = json.loads(eraw.decode("utf-8", errors="replace"))
                    pages = (ejson.get("query") or {}).get("pages") or {}
                except Exception:
                    continue
                for page in pages.values():
                    extract_txt = str(page.get("extract") or "")
                    if len(extract_txt) < 120 or re.search(r"제1조\s*\(목적\)", extract_txt[:600]):
                        continue
                    extracted = extract_dialogues_from_literature(
                        extract_txt[:60000],
                        source_title=str(page.get("title") or title),
                        source_kind="book",
                        max_rotors=40,
                    )
                    if extracted:
                        collected.extend(extracted)
                        works_loaded += 1
                        break
                if works_loaded >= max_works:
                    break

    if not collected:
        en_query = query.strip() if query.strip() and clean_lang != "ko" else "drama"
        g_url = f"https://gutendex.com/books/?search={quote(en_query)}"
        gst, graw, _ = get_fn(g_url, 10.0, {"Accept": "application/json"})
        if gst == 200 and graw:
            try:
                gjson = json.loads(graw.decode("utf-8", errors="replace"))
                results = gjson.get("results") or []
            except Exception:
                results = []
            works_loaded = 0
            for book in results:
                if works_loaded >= max_works:
                    break
                formats = book.get("formats") or {}
                txt_url = ""
                for fmt_key, candidate in formats.items():
                    if str(fmt_key).startswith("text/plain") and str(candidate).startswith("http"):
                        txt_url = str(candidate)
                        break
                if not txt_url:
                    continue
                tst, traw, _ = get_fn(txt_url, 10.0, None)
                if tst != 200 or not traw:
                    continue
                book_txt = traw[:120000].decode("utf-8", errors="replace")
                extracted = extract_dialogues_from_literature(
                    book_txt,
                    source_title=str(book.get("title") or "Project Gutenberg"),
                    source_kind="book",
                    max_rotors=40,
                )
                if extracted:
                    collected.extend(extracted)
                    works_loaded += 1

    return collected


def ingest_local_world_media(
    directory: Path | str,
    max_files: int = 10,
) -> List[SituatedExpressionRotor]:
    """Scans a local directory for subtitles (`.srt`/`.vtt`) paired with audio/video and literary texts."""
    root = Path(directory)
    if not root.exists():
        return []

    collected: List[SituatedExpressionRotor] = []
    scanned = 0
    for fpath in sorted(root.rglob("*")):
        if scanned >= max_files:
            break
        if not fpath.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in fpath.parts):
            continue
        ext = fpath.suffix.lower()
        if ext in (".srt", ".vtt"):
            collected.extend(parse_srt_or_vtt_subtitles(fpath))
            scanned += 1
        elif ext in (".txt", ".md"):
            txt = fpath.read_text(encoding="utf-8", errors="replace")[:80000]
            collected.extend(extract_dialogues_from_literature(txt, source_title=fpath.stem, source_kind="book"))
            scanned += 1
    return collected
