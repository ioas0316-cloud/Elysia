"""
Audio Topology Tracer (         )
=====================================
Core.L5_Mental.M1_Cognition.LLM.audio_topology_tracer

"                        ."

Objective:
    - Whisper     Cross-Attention Layer    .
    - Audio Encoder  Feature  Text Decoder     Token          .
    - '      (Causal Connection)'                      .
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from safetensors import safe_open
from transformers import WhisperForConditionalGeneration, WhisperProcessor

#      
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AudioTracer")

@dataclass
class BridgeSynapse:
    """  (Encoder)    (Decoder)        """
    layer_idx: int
    head_idx: int
    audio_time_idx: int     #              (Time Frame)
    token_idx: int          #              
    attention_weight: float #      

class AudioTopologyTracer:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        #      
        self._load_model()

    def _load_model(self):
        logger.info(f"  Loading Whisper Topology: {self.model_id}")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            logger.info("     Model loaded successfully.")
        except Exception as e:
            logger.error(f"     Failed to load model: {e}")

    def trace_mechanism(self, audio_path: str) -> List[BridgeSynapse]:
        """
                           .
        Cross-Attention                     .
        """
        if self.model is None:
            return []

        logger.info(f"  Tracing Causality in: {os.path.basename(audio_path)}")
        
        # 1.        
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        input_features = inputs.input_features

        # 2. Inference with Attention Retrieval
        # output_attentions=True        '  (Attention)'         
        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                return_dict_in_generate=True,
                output_attentions=True,
                max_new_tokens=50
            )

        # 3. Cross-Attention Analysis
        # outputs.cross_attentions shape: (num_tokens, num_layers, batch, num_heads, seq_len, audio_frames)
        #               "                  "        .
        
        generated_ids = outputs.sequences[0]
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"     Surface Output: '{transcription}'")

        synapses = []
        
        #             
        cross_attentions = outputs.cross_attentions
        # Note: cross_attentions structure depends on transformers version, simplified assumes tuple of layers
        
        for token_pos, layer_attns in enumerate(cross_attentions):
            # layer_attns: Tuple of (batch, heads, 1, audio_frames) for each layer
            
            #          Attention                   
            last_layer_attn = layer_attns[-1] # (batch, heads, 1, audio_frames)
            
            # Head    (Head                     '  '   )
            #   : (heads, 1, audio_frames) -> (audio_frames)
            attn_avg = last_layer_attn[0].mean(dim=0).squeeze() # (audio_frames)
            
            #                    (The Cause)
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
            
            #             (          )
            if max_weight > 0.1:
                #                       (Whisper frame ~20ms)
                time_sec = top_audio_idx * 0.02 
                logger.info(f"     Causal Link: Sound({time_sec:.2f}s) -> Token['{token_str.strip()}'] (Strength: {max_weight:.2f})")

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
