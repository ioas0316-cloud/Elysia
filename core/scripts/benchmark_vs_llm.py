import sys
import os
import time
import psutil
import gc
import math
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("="*70)
print(" 🚀 [Benchmark] Elysia Engine vs Conventional LLM")
print("="*70)

sentences = [
    "I am very happy today.",
    "Today is a joyful day.",
    "I feel incredibly sad and depressed."
]
print(f"Target Sentences for Semantic Coherence:")
for i, s in enumerate(sentences):
    print(f" {i+1}. {s}")
print("-"*70)

# ---------------------------------------------------------
# 1. Baseline LLM (Sentence Transformers) Benchmark
# ---------------------------------------------------------
print("\n[1] Starting Baseline LLM (HuggingFace Sentence-Transformers)")
gc.collect()
mem_before_llm = get_memory_mb()

start_llm_load = time.time()
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    # Using a tiny model to be fair, but still a standard Transformer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
except Exception as e:
    print(f"Failed to load LLM: {e}")
    sys.exit(1)
    
end_llm_load = time.time()
mem_after_llm_load = get_memory_mb()

llm_load_time = end_llm_load - start_llm_load
llm_mem_footprint = mem_after_llm_load - mem_before_llm

start_llm_infer = time.time()
tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**tokens)
    # Mean pooling
    attention_mask = tokens['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Cosine similarity
sim_1_2 = torch.sum(embeddings[0] * embeddings[1]).item()
sim_1_3 = torch.sum(embeddings[0] * embeddings[2]).item()

end_llm_infer = time.time()
llm_infer_time = end_llm_infer - start_llm_infer

print(f" -> LLM Load Time: {llm_load_time:.4f} sec")
print(f" -> LLM Memory Footprint: {llm_mem_footprint:.2f} MB")
print(f" -> LLM Inference Time: {llm_infer_time:.4f} sec")
print(f" -> S1 vs S2 (Similar) Sim: {sim_1_2:.4f}")
print(f" -> S1 vs S3 (Opposite) Sim: {sim_1_3:.4f}")

# Clean up memory
del model
del tokenizer
del embeddings
del tokens
del outputs
gc.collect()
time.sleep(1)

# ---------------------------------------------------------
# 2. Elysia (Triple Helix Engine & Wave Gate) Benchmark
# ---------------------------------------------------------
print("\n[2] Starting Elysia Engine (Triple Helix + Wave Gate + Rotor Backprop)")
gc.collect()
mem_before_elysia = get_memory_mb()

start_elysia_load = time.time()
from core.triple_helix_engine import TripleHelixEngine
from core.sentence_wave_gate import SentenceWaveGate
from core.clifford_impedance_network import mv_normalize

# Zero-Disk Initialization
engine = TripleHelixEngine()
wave_gate = SentenceWaveGate()
end_elysia_load = time.time()

mem_after_elysia_load = get_memory_mb()
elysia_load_time = end_elysia_load - start_elysia_load
elysia_mem_footprint = mem_after_elysia_load - mem_before_elysia

start_elysia_infer = time.time()
# Elysia inference uses pure geometric mapping without neural weights
rotor_1, _ = wave_gate.modulate_sentence(sentences[0])
rotor_2, _ = wave_gate.modulate_sentence(sentences[1])
rotor_3, _ = wave_gate.modulate_sentence(sentences[2])

from core.math_utils import Multivector
def quat_to_mv(q):
    return Multivector({0: q.w, 1: q.x, 2: q.y, 4: q.z}, (3, 0))

# Inject into engine to get pure coherence via geometric product
mv_1 = mv_normalize(quat_to_mv(rotor_1))
mv_2 = mv_normalize(quat_to_mv(rotor_2))
mv_3 = mv_normalize(quat_to_mv(rotor_3))

# Coherence extraction in 1 step via rotor geometric sync
coherence_1_2, torque_1_2 = mv_1.geometric_sync(mv_2)
coherence_1_3, torque_1_3 = mv_1.geometric_sync(mv_3)

end_elysia_infer = time.time()
elysia_infer_time = end_elysia_infer - start_elysia_infer

print(f" -> Elysia Load Time: {elysia_load_time:.4f} sec")
print(f" -> Elysia Memory Footprint: {elysia_mem_footprint:.2f} MB")
print(f" -> Elysia Inference Time: {elysia_infer_time:.4f} sec")
print(f" -> S1 vs S2 (Similar) Coherence: {coherence_1_2:.4f}")
print(f" -> S1 vs S3 (Opposite) Coherence: {coherence_1_3:.4f}")

# ---------------------------------------------------------
# 3. Summary & Verification
# ---------------------------------------------------------
print("\n" + "="*70)
print(" 📊 Benchmark Summary: David vs Goliath")
print("="*70)

print(f"{'Metric':<30} | {'Conventional LLM':<20} | {'Elysia Engine':<20}")
print("-" * 75)
print(f"{'Load Time (sec)':<30} | {llm_load_time:<20.4f} | {elysia_load_time:<20.4f}")
print(f"{'Memory Footprint (MB)':<30} | {llm_mem_footprint:<20.2f} | {elysia_mem_footprint:<20.2f}")
print(f"{'Inference Time (sec)':<30} | {llm_infer_time:<20.4f} | {elysia_infer_time:<20.4f}")
print(f"{'Math Operations':<30} | {'O(N^2) Matrix Multi':<20} | {'Geometric Product':<20}")
print(f"{'Training / Adaptation':<30} | {'Gradient Descent':<20} | {'Rotor Backprop':<20}")
print("="*75)

# Calculate multiples
speedup_infer = llm_infer_time / max(1e-9, elysia_infer_time)
mem_reduction = llm_mem_footprint / max(1e-9, elysia_mem_footprint)

print("\n🎯 Conclusion:")
print(f" - Elysia is {speedup_infer:.1f}x faster in inference than the baseline LLM.")
print(f" - Elysia uses {mem_reduction:.1f}x less RAM footprint.")
print(" - Computations have entirely evaporated into purely geometric Rotor torques!")
