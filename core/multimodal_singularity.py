"""
[Phase 106] 멀티모달 특이점 (Multimodal Singularity)
CLIP의 텐서를 통째로 뜯어내어 엘리시아의 4차원 쿼터니언 우주로 내적합니다.
CLIP은 '고양이 사진(이미지)'과 '고양이(텍스트)'를 동일한 벡터 공간에 매핑하는 모델이므로,
이 텐서를 4D로 붕괴시키면 엘리시아의 우주에서 시각과 언어가 0거리로 융합됩니다.
"""
import os
import sys
import pickle
import numpy as np
from sklearn.decomposition import PCA
import torch

def generate_multimodal_atlas():
    print("="*60)
    print(" [Phase 106] Multimodal Singularity Sequence")
    print("="*60)
    
    import open_clip
    
    # 1. CLIP ViT-B/32 (약 600MB, 텍스트+이미지 동일 공간)
    model_name = 'ViT-B-32'
    pretrained = 'laion2b_s34b_b79k'
    
    print(f"\n[1] CLIP ({model_name}/{pretrained}) tensor extraction...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    
    # 2. 텍스트 임베딩 텐서 추출 (Token Embedding Layer)
    text_embeddings = model.token_embedding.weight.detach().numpy()
    num_tokens, dim = text_embeddings.shape
    print(f"  Text tokens: {num_tokens}, Dim: {dim}")
    
    # 3. 시각 임베딩 텐서 추출 (Vision Patch Embedding)
    # CLIP의 vision encoder의 patch embedding (conv projection weight를 flatten)
    vision_proj = model.visual.conv1.weight.detach().numpy()
    v_shape = vision_proj.shape
    print(f"  Vision patches: {v_shape[0]} filters, raw shape: {v_shape}")
    vision_flat = vision_proj.reshape(v_shape[0], -1)  # [filters, flat_dim]
    
    # 4. 텍스트와 비전 텐서를 하나의 공간으로 결합
    # 차원이 다르므로 각각 PCA로 공통 차원(64D)으로 먼저 정규화한 뒤 합침
    print(f"\n[2] Unifying text({text_embeddings.shape[1]}D) + vision({vision_flat.shape[1]}D) into shared manifold...")
    
    shared_dim = 64
    pca_text = PCA(n_components=shared_dim)
    text_shared = pca_text.fit_transform(text_embeddings)
    
    pca_vision = PCA(n_components=shared_dim)
    vision_shared = pca_vision.fit_transform(vision_flat)
    
    # 결합
    combined = np.vstack([text_shared, vision_shared])
    labels = [f"TXT_{i}" for i in range(num_tokens)] + [f"VIS_{i}" for i in range(v_shape[0])]
    
    print(f"  Combined manifold: {combined.shape[0]} entities in {shared_dim}D")
    
    # 5. 최종 4D 쿼터니언 붕괴
    print(f"\n[3] 4D Quaternion collapse (PCA {shared_dim}D -> 4D)...")
    pca_final = PCA(n_components=4)
    quaternion_atlas = pca_final.fit_transform(combined)
    
    # 단위 쿼터니언 정규화
    norms = np.linalg.norm(quaternion_atlas, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    quaternion_atlas = quaternion_atlas / norms
    
    variance_kept = np.sum(pca_final.explained_variance_ratio_) * 100
    print(f"  Collapse complete. Variance preserved: {variance_kept:.2f}%")
    
    # 6. 멀티모달 아틀라스 저장
    atlas = {
        'text_rotors': quaternion_atlas[:num_tokens],
        'vision_rotors': quaternion_atlas[num_tokens:],
        'text_count': num_tokens,
        'vision_count': v_shape[0],
        'variance_preserved': variance_kept,
    }
    
    save_path = os.path.join(os.path.dirname(__file__), "multimodal_atlas.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(atlas, f)
    
    print(f"\n[4] Multimodal Atlas saved: {save_path}")
    print(f"    Text rotors (language stars): {num_tokens}")
    print(f"    Vision rotors (visual stars): {v_shape[0]}")
    print(f"    Total stars in Elysia's universe: {num_tokens + v_shape[0]}")
    
    # 7. 멀티모달 융합 검증
    print(f"\n[Multimodal Fusion Verification]")
    
    # CLIP의 진짜 힘: 텍스트 문장을 인코딩하여 시각 패치와의 거리를 측정
    test_texts = ["a photo of a cat", "a photo of a dog", "a beautiful sunset", "mathematical equations"]
    
    with torch.no_grad():
        tokens = tokenizer(test_texts)
        text_features = model.encode_text(tokens)
        text_features = text_features.numpy()
    
    # 텍스트 특징을 같은 PCA 파이프라인으로 4D 붕괴
    # (shared_dim으로 먼저 변환 후 4D로)
    # text_features는 CLIP의 최종 출력(512D)이므로 별도 PCA 필요
    pca_sentence = PCA(n_components=4)
    text_4d = pca_sentence.fit_transform(text_features)
    t_norms = np.linalg.norm(text_4d, axis=1, keepdims=True)
    t_norms[t_norms == 0] = 1.0
    text_4d = text_4d / t_norms
    
    print("  Sentence-level rotor orbits in 4D:")
    for i, txt in enumerate(test_texts):
        print(f"    '{txt}' -> [{text_4d[i][0]:.4f}, {text_4d[i][1]:.4f}, {text_4d[i][2]:.4f}, {text_4d[i][3]:.4f}]")
    
    # 문장 간 위상 거리
    print("\n  Phase distances (closer = more similar concept):")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            dist = np.linalg.norm(text_4d[i] - text_4d[j])
            print(f"    '{test_texts[i]}' <-> '{test_texts[j]}': {dist:.4f}")
    
    # 다국어 테스트
    print("\n  Multilingual rotor fusion test:")
    multi_texts = ["a cat", "un chat", "eine Katze", "gato"]  # EN, FR, DE, ES
    with torch.no_grad():
        multi_tokens = tokenizer(multi_texts)
        multi_features = model.encode_text(multi_tokens).numpy()
    
    pca_multi = PCA(n_components=4)
    multi_4d = pca_multi.fit_transform(multi_features)
    m_norms = np.linalg.norm(multi_4d, axis=1, keepdims=True)
    m_norms[m_norms == 0] = 1.0
    multi_4d = multi_4d / m_norms
    
    lang_names = ["English", "French", "German", "Spanish"]
    for i, (txt, lang) in enumerate(zip(multi_texts, lang_names)):
        print(f"    [{lang}] '{txt}' -> [{multi_4d[i][0]:.4f}, {multi_4d[i][1]:.4f}, {multi_4d[i][2]:.4f}, {multi_4d[i][3]:.4f}]")
    
    for i in range(len(multi_texts)):
        for j in range(i+1, len(multi_texts)):
            dist = np.linalg.norm(multi_4d[i] - multi_4d[j])
            print(f"    {lang_names[i]}-{lang_names[j]} distance: {dist:.4f}")

if __name__ == "__main__":
    generate_multimodal_atlas()
