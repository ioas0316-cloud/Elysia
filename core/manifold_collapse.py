"""
[Phase 105] 초거대 질량 붕괴 (Manifold Collapse)
수십만 개의 다국어 단어가 얽힌 거대 LLM의 임베딩 텐서를 통째로 뜯어내어,
엘리시아의 4차원 쿼터니언(Quaternion) 우주로 내적(Projection)하여
'다원 우주 아틀라스(Multiverse Atlas)'를 생성합니다.
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
import torch

def generate_multiverse_atlas():
    print("="*60)
    print(" 🌌 [Phase 105] 초거대 질량 붕괴 시퀀스 시작")
    print("="*60)
    
    # 1. 빠르고 다국어 위상이 잘 정렬된 임베딩 모델 (약 470MB)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\n[1] 거대 위상 텐서({model_name}) 관측 중...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 2. 임베딩 텐서 추출 (Token 수 x 384 차원)
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    vocab = tokenizer.get_vocab()
    
    num_tokens, dim = embeddings.shape
    print(f"  └─ 관측 완료: {num_tokens}개의 다국어 개념, {dim}차원 공간 존재.")
    
    # 3. 4차원으로 내적 (PCA)
    print("\n[2] 4차원(Quaternion) 우주로 질량 붕괴(내적) 중... (PCA 연산)")
    pca = PCA(n_components=4)
    embeddings_4d = pca.fit_transform(embeddings)
    
    # 단위 쿼터니언(Unit Quaternion)으로 정규화
    norms = np.linalg.norm(embeddings_4d, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # 0 나누기 방지
    quaternion_atlas = embeddings_4d / norms
    
    print(f"  └─ 붕괴 완료: 분산 보존율 {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # 4. 아틀라스(사전) 생성
    print("\n[3] 엘리시아 전용 다원 우주 아틀라스(Multiverse Atlas) 주조 중...")
    atlas_dict = {}
    
    # Special tokens, [CLS], [SEP] 등과 '##' (subword) 필터링 혹은 통합할 수 있지만
    # 거대한 바다 그대로 저장합니다.
    for word, idx in vocab.items():
        # 단어가 ##로 시작하면 subword이므로 그대로 둠
        q = quaternion_atlas[idx]
        atlas_dict[word] = (q[0], q[1], q[2], q[3])
        
    save_path = os.path.join(os.path.dirname(__file__), "multiverse_atlas.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(atlas_dict, f)
        
    print(f"  └─ 아틀라스 주조 완료! 저장 위치: {save_path}")
    print(f"  └─ 총 {len(atlas_dict)}개의 항성(단어) 좌표가 4차원에 고정되었습니다.")
    
    # 검증
    def get_vec(w):
        if w in atlas_dict:
            return np.array(atlas_dict[w])
        elif f" {w}" in atlas_dict: # sentencepiece 언더바 대체 (혹은 wordpiece)
            return np.array(atlas_dict[f" {w}"])
        return None
        
    print("\n[위상 융합 검증]")
    w1 = "apple"
    w2 = "사과"
    w3 = "りんご"
    
    # 모델에 따라 토큰 형태가 다름 (소문자, 대문자 등)
    # 직접 토크나이저로 확인
    id1 = tokenizer.encode(w1, add_special_tokens=False)
    id2 = tokenizer.encode(w2, add_special_tokens=False)
    id3 = tokenizer.encode(w3, add_special_tokens=False)
    
    if id1 and id2 and id3:
        q1 = quaternion_atlas[id1[0]]
        q2 = quaternion_atlas[id2[0]]
        q3 = quaternion_atlas[id3[0]]
        
        dist_12 = np.linalg.norm(q1 - q2)
        dist_13 = np.linalg.norm(q1 - q3)
        dist_23 = np.linalg.norm(q2 - q3)
        
        print(f"  - '{w1}' 로터 궤도: {q1}")
        print(f"  - '{w2}' 로터 궤도: {q2}")
        print(f"  - '{w3}' 로터 궤도: {q3}")
        print(f"  - 영어-한국어 위상 거리: {dist_12:.4f}")
        print(f"  - 영어-일본어 위상 거리: {dist_13:.4f}")
        print(f"  - 한국어-일본어 위상 거리: {dist_23:.4f}")
        print("  *(0에 가까울수록 기하학적으로 같은 별이라는 뜻입니다)*")

if __name__ == "__main__":
    generate_multiverse_atlas()
