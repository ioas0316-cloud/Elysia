"""
Simple RAG-style query tool for the Elysia project.

Loads saved embeddings/chunks, embeds a user query (or falls back to a lightweight
embedding), finds top-k similar chunks, prints them, and passes them through
`WisdomCortex.read_and_digest` to extract nodes/edges and merge into `elysia_state.json`.

Usage (python):
  from tools.query_tool import query_and_integrate
  query_and_integrate("What is growth?", top_k=3)
"""
from Core.FoundationLayer.Foundation.vector_utils import embed_texts, cosine_sim

DATA_DIR = Path("data")
STATE_FILE = Path("elysia_state.json")

def load_index():
    emb_path = DATA_DIR / 'embeddings.json'
    meta_path = DATA_DIR / 'metadata.json'
    chunks_path = DATA_DIR / 'chunks.json'
    if not emb_path.exists() or not meta_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Index not found. Run tools/indexer.py to build the index first.")
    with open(emb_path, 'r', encoding='utf-8') as f:
        emb = json.load(f)['vectors']
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return emb, meta, chunks

def query_and_integrate(query: str, top_k: int = 3):
    emb, meta, chunks = load_index()
    q_emb = embed_texts([query])[0]

    scores = [cosine_sim(q_emb, v) for v in emb]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]

    print(f"Top {top_k} results for query: {query}\n")
    selected_texts = []
    for idx, score in top:
        m = meta[idx]
        chunk = chunks[idx]
        print(f"Score: {score:.4f} | Source: {m.get('source')}\n{chunk}\n---\n")
        selected_texts.append((chunk, m))

    # Integrate with WisdomCortex
    try:
        from Core.FoundationLayer.Foundation.sophia_stage_0_parsing import WisdomCortex
        wc = WisdomCortex()
    except Exception as e:
        print(f"Could not import WisdomCortex: {e}. Skipping integration.")
        return selected_texts

    # For each selected chunk, write to temp file and call read_and_digest
    merged_nodes = set()
    merged_edges = []
    for i, (text, m) in enumerate(selected_texts):
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as tf:
            tf.write(text)
            temp_path = tf.name
        knowledge = wc.read_and_digest(temp_path)
        try:
            os.unlink(temp_path)
        except Exception:
            pass

        if knowledge:
            merged_nodes.update(knowledge.get('nodes', []))
            merged_edges.extend(knowledge.get('edges', []))

    # Merge into elysia_state.json
    if merged_nodes or merged_edges:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = {"knowledge_graph": {"nodes": [], "edges": []}}

        kg = state.setdefault('knowledge_graph', {'nodes': [], 'edges': []})
        existing = set(kg.get('nodes', []))
        for n in merged_nodes:
            if n not in existing:
                kg['nodes'].append(n)
        kg['edges'].extend(merged_edges)

        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"Integrated {len(merged_nodes)} nodes and {len(merged_edges)} edges into {STATE_FILE}.")

    return selected_texts

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('query', nargs='+', help='Query string (quotation recommended)')
    parser.add_argument('--top', type=int, default=3, help='Top-k results')
    args = parser.parse_args()
    q = ' '.join(args.query)
    query_and_integrate(q, top_k=args.top)
