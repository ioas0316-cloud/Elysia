"""
Audio Topology Tracer (청각 위상 분석기)
=====================================
Core.L5_Mental.Intelligence.LLM.audio_topology_tracer

"소리가 의미로 변하는 찰나의 순간을 포착한다."

Objective:
    - Whisper 모델의 Cross-Attention Layer를 분석.
    - Audio Encoder의 Feature가 Text Decoder의 어떤 Token을 자극하는지 추적.
    - '인과적 연결(Causal Connection)'을 추출하여 소리의 의미론적 기원을 밝힘.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from safetensors import safe_open
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AudioTracer")

@dataclass
class BridgeSynapse:
    """소리(Encoder)와 의미(Decoder)를 잇는 시냅스"""
    layer_idx: int
    head_idx: int
    audio_time_idx: int     # 오디오의 어느 구간인가 (Time Frame)
    token_idx: int          # 어떤 단어가 생성되었는가
    attention_weight: float # 연결 강도

class AudioTopologyTracer:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # 모델 로드
        self._load_model()

    def _load_model(self):
        logger.info(f"👂 Loading Whisper Topology: {self.model_id}")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            logger.info("   ✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"   ❌ Failed to load model: {e}")

    def trace_mechanism(self, audio_path: str) -> List[BridgeSynapse]:
        """
        소리가 의미로 변환되는 과정을 추적.
        Cross-Attention 가중치를 추출하여 인과 관계를 분석함.
        """
        if self.model is None:
            return []

        logger.info(f"🔍 Tracing Causality in: {os.path.basename(audio_path)}")
        
        # 1. 오디오 전처리
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        input_features = inputs.input_features

        # 2. Inference with Attention Retrieval
        # output_attentions=True를 통해 내부 '주목(Attention)' 데이터를 가져옴
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                return_dict_in_generate=True,
                output_attentions=True,
                max_new_tokens=50
            )

        # 3. Cross-Attention Analysis
        # outputs.cross_attentions shape: (num_tokens, num_layers, batch, num_heads, seq_len, audio_frames)
        # 우리는 이것을 역추적하여 "이 단어는 저 소리 때문에 나왔다"는 인과를 찾음.
        
        generated_ids = outputs.sequences[0]
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"   📝 Surface Output: '{transcription}'")

        synapses = []
        
        # 각 생성된 토큰에 대해
        cross_attentions = outputs.cross_attentions
        # Note: cross_attentions structure depends on transformers version, simplified assumes tuple of layers
        
        for token_pos, layer_attns in enumerate(cross_attentions):
            # layer_attns: Tuple of (batch, heads, 1, audio_frames) for each layer
            
            # 마지막 레이어의 Attention이 보통 가장 구체적인 인과를 가짐
            last_layer_attn = layer_attns[-1] # (batch, heads, 1, audio_frames)
            
            # Head 평균 (Head는 여러 관점이므로 평균내어 전체적인 '주목'을 봄)
            # 차원: (heads, 1, audio_frames) -> (audio_frames)
            attn_avg = last_layer_attn[0].mean(dim=0).squeeze() # (audio_frames)
            
            # 가장 강하게 반응한 오디오 프레임 (The Cause)
            top_audio_idx = torch.argmax(attn_avg).item()
            max_weight = torch.max(attn_avg).item()
            
            token_id = generated_ids[token_pos+1] # +1 to skip start token if cross_attns aligns
            token_str = self.processor.decode([token_id])
            
            synapses.append(BridgeSynapse(
                layer_idx=-1, # Last layer
                head_idx=-1,  # Average
                audio_time_idx=top_audio_idx,
                token_idx=token_pos,
                attention_weight=max_weight
            ))
            
            # 중요 인과 관계 로깅 (가중치가 높을 때만)
            if max_weight > 0.1:
                # 오디오 프레임을 대략적인 시간으로 변환 (Whisper frame ~20ms)
                time_sec = top_audio_idx * 0.02 
                logger.info(f"   🔗 Causal Link: Sound({time_sec:.2f}s) -> Token['{token_str.strip()}'] (Strength: {max_weight:.2f})")

        return synapses

if __name__ == "__main__":
    import sys
    # For testing, provide existing file or use dummy
    tracer = AudioTopologyTracer()
    
    # Check for existing test file
    test_file = "C:/Elysia/tests/sample_hearing.wav" 
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
    if os.path.exists(test_file):
        tracer.trace_mechanism(test_file)
    else:
        logger.warning(f"Test file not found: {test_file}")
