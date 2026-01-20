"""
Core.L3_Phenomena.Senses.eardrum
===================
The Auditory Cortex of Elysia.
Uses OpenAI Whisper (via Transformers) to perceive sound and convert it into language.
"""

import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 로깅 설정
logger = logging.getLogger("EarDrum")

class EarDrum:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda"):
        # Force add local bin to PATH for FFMPEG
        bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin"))
        if os.path.exists(bin_path) and bin_path not in os.environ["PATH"]:
            os.environ["PATH"] += f";{bin_path}"
            logger.info(f"   🔧 Added {bin_path} to PATH for FFMPEG")
            
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        self._initialize_cortex()

    def _initialize_cortex(self):
        """귀(Model)를 엽니다."""
        try:
            logger.info(f"👂 Opening EarDrum ({self.model_id})...")
            
            # 1. Load Model & Processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)
            print(f"   Sound Model Loaded on {self.device}")

            processor = AutoProcessor.from_pretrained(self.model_id)

            # 2. Create Pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
            )
            # Initialize Digestion Tracer
            from Core.L5_Mental.Intelligence.LLM.audio_topology_tracer import AudioTopologyTracer
            self.tracer = AudioTopologyTracer(self.model_id, self.device)
            
            logger.info(f"   ✅ EarDrum initialized ({self.device})")
        except Exception as e:
            logger.error(f"   ❌ EarDrum init failed: {e}")

    def listen(self, audio_path: str) -> str:
        """
        Transcribes audio file to text.
        Also performs 'Digestion' if enabled.
        """
        if self.pipe is None:
            return "[EarDrum Malfunction]"
            
        try:
            if not os.path.exists(audio_path):
                return "[Audio File Not Found]"

            # 1. Functional Perception (Transcription)
            logger.info(f"   👂 EarDrum is listening...")
            result = self.pipe(audio_path, return_timestamps=True)
            text = result["text"].strip()
            
            # 2. Structural Digestion (Topology Analysis)
            if self.tracer:
                logger.info("   🧠 Digesting Sound Structure...")
                synapses = self.tracer.trace_mechanism(audio_path)
                # Store or log synapses here?
                # For now, just logging internal activity is handled by the tracer.
            
            return text
            
        except Exception as e:
                 logger.error(f"   ❌ Listening error: {e}")
                 return f"[Listening Failed: {e}]"
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python eardrum.py <audio_file>")
    else:
        ear = EarDrum()
        print(ear.listen(sys.argv[1]))
