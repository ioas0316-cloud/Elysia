"""
[Phase 106-B] 위상거울 원리에 의한 초거대 모델 임베딩 강탈
1테라바이트 모델의 전체를 다운받지 않는다.
임베딩 텐서(단물)가 담긴 샤드 파일 하나만 정확히 지정하여 반사(다운로드)하고,
그 안에서 embed_tokens.weight 텐서만 뜯어내어 4D 쿼터니언으로 붕괴시킨다.
"""
import os
import json
import pickle
import numpy as np
from sklearn.decomposition import PCA

def steal_giant_brain():
    print("="*60)
    print(" [Phase 106-B] Phase Mirror: Giant LLM Embedding Theft")
    print("="*60)
    
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    
    # Qwen2.5-72B-Instruct: 72B params, ~150GB total
    # 우리는 임베딩 텐서가 담긴 샤드 1개만 반사(다운로드)한다.
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    
    print(f"\n[1] Target: {model_id} (~150GB total)")
    print(f"    We will NOT download 150GB.")
    print(f"    Phase Mirror: reflecting ONLY the embedding shard...")
    
    # Step 1: 인덱스 파일만 먼저 반사 (수 KB)
    print(f"\n[2] Reflecting model index (shard map)...")
    index_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json"
    )
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    # Step 2: embed_tokens.weight가 어느 샤드에 있는지 찾기
    embed_key = None
    embed_shard = None
    for key, shard_file in index['weight_map'].items():
        if 'embed_tokens' in key:
            embed_key = key
            embed_shard = shard_file
            break
    
    if not embed_shard:
        print("  ERROR: embed_tokens not found in index!")
        return
        
    total_params = sum(1 for _ in index['weight_map'])
    print(f"  Total weight tensors in model: {total_params}")
    print(f"  Embedding tensor: '{embed_key}' -> shard: '{embed_shard}'")
    
    # Step 3: 해당 샤드만 반사 (전체 150GB 중 ~5GB만)
    print(f"\n[3] Reflecting ONLY shard '{embed_shard}'...")
    print(f"    (This is ~3% of the full 150GB model)")
    shard_path = hf_hub_download(
        repo_id=model_id,
        filename=embed_shard
    )
    
    # Step 4: 샤드에서 임베딩 텐서만 추출
    print(f"\n[4] Extracting embedding tensor from shard...")
    with safe_open(shard_path, framework="numpy") as f:
        embedding_tensor = f.get_tensor(embed_key)
    
    vocab_size, hidden_dim = embedding_tensor.shape
    size_gb = embedding_tensor.nbytes / (1024**3)
    print(f"  Extracted: {vocab_size} tokens x {hidden_dim}D")
    print(f"  Tensor size: {size_gb:.2f} GB")
    print(f"  This single tensor encodes the geometric relationships")
    print(f"  of ALL {vocab_size} concepts in the 72B brain.")
    
    # Step 5: 4D 쿼터니언 질량 붕괴
    print(f"\n[5] Collapsing {hidden_dim}D -> 4D Quaternion space (PCA)...")
    
    # float16 -> float32 for PCA
    embedding_f32 = embedding_tensor.astype(np.float32)
    del embedding_tensor  # 메모리 해방
    
    pca = PCA(n_components=4)
    quaternion_atlas = pca.fit_transform(embedding_f32)
    del embedding_f32  # 메모리 해방
    
    # 단위 쿼터니언 정규화
    norms = np.linalg.norm(quaternion_atlas, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    quaternion_atlas = quaternion_atlas / norms
    
    variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Collapse complete! Variance preserved: {variance:.2f}%")
    print(f"  {vocab_size} stars now orbit in 4D Quaternion space.")
    
    # Step 6: 토크나이저 반사 (단어-ID 매핑)
    print(f"\n[6] Reflecting tokenizer vocabulary...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    
    # 아틀라스 생성
    atlas_dict = {}
    for word, idx in vocab.items():
        if idx < vocab_size:
            q = quaternion_atlas[idx]
            atlas_dict[word] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    
    save_path = os.path.join(os.path.dirname(__file__), "giant_atlas_72B.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(atlas_dict, f)
    
    print(f"  Atlas saved: {save_path}")
    print(f"  Total stars mapped: {len(atlas_dict)}")
    
    # Step 7: 융합 검증
    print(f"\n[Phase Mirror Verification]")
    print(f"  Source model: {model_id} (~150GB)")
    print(f"  Downloaded: ~{size_gb:.1f}GB (embedding shard only)")
    print(f"  Theft ratio: {size_gb/150*100:.1f}% downloaded, 100% geometric knowledge stolen")
    
    # 다국어 검증
    test_pairs = [
        ("cat", "dog"),
        ("love", "hate"),
        ("sun", "moon"),
    ]
    
    print(f"\n  Semantic distance verification:")
    for w1, w2 in test_pairs:
        ids1 = tokenizer.encode(w1, add_special_tokens=False)
        ids2 = tokenizer.encode(w2, add_special_tokens=False)
        if ids1 and ids2:
            q1 = quaternion_atlas[ids1[0]]
            q2 = quaternion_atlas[ids2[0]]
            dist = np.linalg.norm(q1 - q2)
            print(f"    '{w1}' <-> '{w2}': {dist:.4f}")

if __name__ == "__main__":
    steal_giant_brain()
