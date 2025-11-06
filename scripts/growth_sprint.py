import argparse
import os
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tools.kg_manager import KGManager
from tools import activity_registry as act
from tools.text_preprocessor import extract_content_tokens
from Project_Sophia.wisdom_virus import WisdomVirus, VirusEngine
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.scheduler import Scheduler
from nano_core.message import Message
from nano_core.bots.linker import LinkerBot
from nano_core.bots.validator import ValidatorBot
from nano_core.bots.summarizer import SummarizerBot
from nano_core.bots.composer import ComposerBot
from nano_core.bots.explainer import ExplainerBot


def ingest_literature_root(root: Path, kg: KGManager):
    from scripts.ingest_literature import ingest_folder
    for sub in root.iterdir():
        if sub.is_dir():
            ingest_folder(sub, kg, label=sub.name)
    kg.save()


def ensure_journal_docs(kg: KGManager, journal_dir: Path = Path("data/journal")):
    if not journal_dir.exists():
        return []
    added = []
    for p in journal_dir.glob("*.txt"):
        doc_id = f"journal_doc:{p.stem}"
        if not kg.get_node(doc_id):
            kg.add_node(doc_id, properties={
                "type": "journal_doc",
                "path": str(p),
                "experience_text": str(p),
            })
            added.append(doc_id)
    if added:
        kg.save()
    return added


def collect_docs(kg: KGManager) -> List[Dict]:
    docs = []
    for n in kg.kg.get("nodes", []):
        t = str(n.get("type", ""))
        if t in ("literature_doc", "journal_doc") and n.get("path"):
            docs.append(n)
    return docs


def build_tfidf(docs: List[Dict]) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    tf: Dict[str, Counter] = {}
    df: Counter = Counter()
    for n in docs:
        path = Path(n.get("path"))
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        toks = extract_content_tokens(text)
        c = Counter(toks)
        tf[n["id"]] = c
        for w in c:
            df[w] += 1
    return tf, df


def link_keywords_to_concepts(kg: KGManager, docs: List[Dict], tf: Dict[str, Counter], df: Dict[str, int], topk: int = 5):
    N = max(1, len(docs))
    for n in docs:
        did = n["id"]
        counts = tf.get(did, Counter())
        # compute tf-idf
        scores: List[Tuple[str, float]] = []
        for w, cnt in counts.items():
            idf = math.log((N + 1) / (1 + df.get(w, 0))) + 1.0
            scores.append((w, float(cnt) * idf))
        scores.sort(key=lambda x: x[1], reverse=True)
        for w, s in scores[:topk]:
            concept_id = f"concept:{w}"
            kg.add_node(concept_id, properties={"type": "concept", "label": w})
            # confidence scaled 0..1 within this doc
            conf = 0.0
            if scores:
                conf = min(1.0, s / (scores[0][1] + 1e-9) * 0.9 + 0.1)
            kg.add_edge(did, concept_id, "supports", properties={
                "confidence": conf,
                "evidence_paths": [n.get("path", "")],
            })
    kg.save()


def nano_augment_after_keywords(kg: KGManager, df: Dict[str, int]) -> None:
    """Lightweight nano step: summarize top concepts and compose simple pairs."""
    try:
        top = gather_top_concepts(df)[:4]
        if not top:
            return
        bus = MessageBus()
        reg = ConceptRegistry()
        bots = [LinkerBot(), ValidatorBot(), SummarizerBot(), ComposerBot(), ExplainerBot()]
        sched = Scheduler(bus, reg, bots)
        # Summarize top concepts
        for t in top[:3]:
            bus.post(Message(verb='summarize', slots={'target': f'concept:{t}'}, strength=0.6, ttl=1))
        # Compose first two concepts into a combo
        if len(top) >= 2:
            a, b = f'concept:{top[0]}', f'concept:{top[1]}'
            bus.post(Message(verb='compose', slots={'a': a, 'b': b}, strength=0.8, ttl=2))
        sched.step(max_steps=50)
    except Exception:
        pass


def run_simple_viruses(kg: KGManager, top_concepts: List[str]):
    engine = VirusEngine(kg_manager=kg)
    for token in top_concepts[:3]:
        v = WisdomVirus(
            id=f"wisdom:focus_{token}",
            statement=f"오늘의 핵심 개념은 '{token}' 입니다.",
            seed_hosts=[f"concept:{token}"],
            reinforce=0.35,
            decay=0.01,
            max_hops=3,
        )
        engine.propagate(v, context_tag="growth_sprint")


def gather_top_concepts(df: Dict[str, int]) -> List[str]:
    # Return globally frequent tokens (length>=2)
    items = [(w, c) for w, c in df.items() if len(w) >= 2]
    items.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in items[:10]]


def maybe_run_daily_report():
    try:
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, "-m", "scripts.run_daily_report"], check=False)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Growth Sprint: ingest -> keywords->concepts -> viruses -> report")
    ap.add_argument("--ingest", action="store_true", help="Ingest data/corpus/literature into KG")
    ap.add_argument("--keywords", action="store_true", help="Link doc keywords to concept:* with supports")
    ap.add_argument("--virus", action="store_true", help="Propagate simple focus viruses from top concepts")
    ap.add_argument("--report", action="store_true", help="Generate daily report at the end")
    ap.add_argument("--root", default="data/corpus/literature", help="Literature root folder")
    args = ap.parse_args()

    kg = KGManager()
    act.start('growth_sprint', {'ingest': args.ingest, 'keywords': args.keywords, 'virus': args.virus, 'report': args.report})
    
    if args.ingest:
        ingest_literature_root(Path(args.root), kg)
        ensure_journal_docs(kg)

    docs = collect_docs(kg)
    if args.keywords and docs:
        tf, df = build_tfidf(docs)
        link_keywords_to_concepts(kg, docs, tf, df, topk=5)
        nano_augment_after_keywords(kg, df)
        if args.virus:
            top_concepts = gather_top_concepts(df)
            run_simple_viruses(kg, top_concepts)

    if args.report:
        maybe_run_daily_report()

    print("[growth_sprint] Completed.")
    act.stop('growth_sprint')


if __name__ == "__main__":
    main()
