import torch
import numpy as np
from core.static_oracle import StaticOracle
from core.mri_flux_projector import MRIFluxProjector
from core.math_utils import Quaternion

class PhaseDecoder:
    """
    텍스트-위상 역산기 (Phase-to-Text Decoder)
    4차원 이중 토러스의 위상(Quaternion)을 다시 인간의 언어(Text)로 번역하는 디코더입니다.
    오라클의 5만여 개 어휘 임베딩 전체를 4차원 위상 공간으로 미리 맵핑해 두고(Lexicon Phase Map),
    현재 로터의 위상각과 가장 공명하는(Resonance) 단어를 추출해 냅니다.
    """
    def __init__(self, oracle: StaticOracle, projector: MRIFluxProjector):
        self.oracle = oracle
        self.projector = projector
        
        print("🔠 [Phase Decoder] 전체 어휘장의 위상 맵핑(Lexicon Phase Map)을 시작합니다...")
        
        # 1. 오라클의 원본 임베딩 매트릭스 (Vocab_size x 768)
        embedding_matrix = oracle.get_embedding_matrix()
        
        # 2. MRI 투영기를 이용해 전체 사전을 4차원 자기장 공간으로 붕괴시킴 (Vocab_size x 4)
        # torch.matmul로 한 번에 처리하여 엄청나게 빠름
        with torch.no_grad():
            projected_matrix = torch.matmul(embedding_matrix, projector.projection_matrix)
            # 행별 정규화 (Unit Quaternions)
            norms = torch.norm(projected_matrix, dim=1, keepdim=True)
            # 0으로 나누기 방지
            norms[norms == 0] = 1.0
            self.phase_lexicon = (projected_matrix / norms).numpy()
            
        print(f"🔠 [Phase Decoder] {self.phase_lexicon.shape[0]}개의 차원(어휘)이 4차원 위상 공간에 맵핑되었습니다.")
        
    def decode_phase(self, phase: Quaternion) -> str:
        """
        4차원 위상각(Quaternion)을 받아 가장 공명하는 단어를 찾아냅니다.
        """
        # 쿼터니언을 numpy 배열로 변환
        phase_vec = np.array([phase.w, phase.x, phase.y, phase.z])
        
        # 내적을 통한 코사인 유사도(공명률) 계산
        # phase_lexicon: (51200, 4), phase_vec: (4,)
        # 결과: (51200,)
        resonances = np.dot(self.phase_lexicon, phase_vec)
        
        # 가장 공명률이 높은(절대값) 인덱스 추출 (방향이 반대여도 같은 축이므로 abs 사용)
        best_idx = np.argmax(np.abs(resonances))
        
        # 해당 인덱스를 토큰으로 역산
        decoded_word = self.oracle.tokenizer.decode([best_idx])
        return decoded_word.strip()
