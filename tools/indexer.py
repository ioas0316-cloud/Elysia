"""
Local document indexer for Elysia project.

Scans `Elysia_Input_Sanctum` for text files, chunks them, computes embeddings
using `sentence-transformers` when available, and falls back to a lightweight
hash-based embedding if not. Saves embeddings and metadata under `data/`.

Usage (python):
  from tools.indexer import build_index
  build_index()
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path("data")
SANCTUM_DIR = Path("Elysia_Input_Sanctum")

def list_text_files(sanctum: Path) -> List[Path]:
    files = []
    for p in sanctum.iterdir():
        if p.is_file() and p.suffix.lower() in ['.txt', '.md']:
            files.append(p)
    return files

def chunk_text(text: str, max_len: int = 400) -> List[str]:
    """Simple chunker: split by paragraphs, then by sentence length approximation."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            # naive sentence split
            sentences = [s.strip() for s in p.split('.') if s.strip()]
            cur = []
            cur_len = 0
            for s in sentences:
                sl = s + '.'
                if cur_len + len(sl) > max_len and cur:
                    chunks.append(' '.join(cur))
                    cur = [sl]
                    cur_len = len(sl)
                else:
                    cur.append(sl)
                    cur_len += len(sl)
            if cur:
                chunks.append(' '.join(cur))
    return chunks

def try_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Try to use sentence-transformers, otherwise fallback to hash-based vectors."""
    ModelClass = try_import_sentence_transformer()
    if ModelClass:
        model = ModelClass('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    # Fallback embedding: hashed bag-of-words into fixed dimension
    dim = 256
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for w in t.split():
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            v[idx] += 1.0
        # normalize
        norm = sum(x*x for x in v) ** 0.5
        if norm > 0:
            v = [x / norm for x in v]
        vecs.append(v)
    return vecs

def save_index(embeddings, metadata, chunk_texts):
    DATA_DIR.mkdir(exist_ok=True)
    # Save embeddings as JSON-friendly structure (list of lists)
    with open(DATA_DIR / 'embeddings.json', 'w', encoding='utf-8') as f:
        json.dump({'vectors': embeddings}, f)
    with open(DATA_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

def build_index(sanctum_dir: Path = SANCTUM_DIR):
    files = list_text_files(sanctum_dir)
    all_chunks = []
    metadata = []
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadata.append({'source': str(fpath), 'chunk_index': i})

    if not all_chunks:
        print("No chunks found to index.")
        return False

    embeddings = embed_texts(all_chunks)
    save_index(embeddings, metadata, all_chunks)
    print(f"Indexed {len(all_chunks)} chunks from {len(files)} files.")
    return True

if __name__ == '__main__':
    build_index()
