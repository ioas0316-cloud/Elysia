"""
Core.L3_Phenomena.Expression.voicebox
========================
The Vocal Cords of Elysia.
Uses CosyVoice-300M to synthesize emotional and natural speech.
"""

import os
import sys
import torch
import logging
import torchaudio

#      
logger = logging.getLogger("VoiceBox")

class VoiceBox:
    def __init__(self, model_path: str = None, device: str = "cuda"):
        #            (한국어 학습 시스템)
        if model_path is None:
            home = os.path.expanduser("~")
            self.model_path = os.path.join(home, ".cache/huggingface/hub/models--FunAudioLLM--CosyVoice-300M")
        else:
            self.model_path = model_path
            
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        
        self._initialize_cords()

    def _initialize_cords(self):
        """  (Model)       ."""
        if not os.path.exists(self.model_path):
            logger.warning(f"   Voice model not found at {self.model_path}")
            return

        try:
            logger.info(f"  Warming up VoiceBox ({self.model_path})...")
            
            # CosyVoice import   
            try:
                # Add external library path
                external_path = os.path.join(os.getcwd(), "data", "external", "CosyVoice")
                if external_path not in sys.path:
                    sys.path.append(external_path)
                
                from cosyvoice.cli.cosyvoice import CosyVoice
                self.model = CosyVoice(self.model_path)
                
                # Initialize Digestion Tracer
                try:
                    from Core.L5_Mental.M1_Cognition.LLM.voice_flow_tracer import VoiceFlowTracer
                    self.tracer = VoiceFlowTracer(self.model)
                    logger.info("     VoiceBox & FlowTracer ready.")
                except Exception as e:
                    logger.warning(f"      VoiceFlowTracer init failed: {e}")
                    self.tracer = None
                    
            except ImportError:
                logger.error("     'cosyvoice' package not installed.")
                self.model = None
                self.tracer = None

        except Exception as e:
            logger.error(f"     Failed to initialize VoiceBox: {e}")
            self.model = None

    def speak(self, text: str, output_path: str = "output.wav", speaker_id: str = "   ") -> str:
        """
                       .
        
        Args:
            text:        
            output_path:          
            speaker_id:    ID (SFT      )
            
        Returns:
            str:              
        """
        if self.model is None:
            logger.warning("     VoiceBox is mute (Model not loaded).")
            return ""

        logger.info(f"      Speaking: '{text}'")
        
        # 1. Structural Digestion (Flow Analysis)
        flow_data = None
        if self.tracer:
            logger.info("     Digesting Emotional Causality...")
            # Mocking digestion for now since CosyVoice might be missing
            # flow_data = self.tracer.digest_flow(text)
            from Core.L5_Mental.M1_Cognition.LLM.voice_flow_tracer import FlowCausality
            flow_data = FlowCausality(7, 0.88, "Pitch/Tone") # Simulated Return

        try:
            if output_path is None:
                output_path = "output.wav"
            
            # SFT Inference (if model exists)
            if self.model:
                speaker_id = self.spks[0] if hasattr(self, 'spks') and self.spks else "   "
                output = self.model.inference_sft(text, speaker_id)
                for item in output:
                    audio_tensor = item['tts_speech']
                    torchaudio.save(output_path, audio_tensor, 22050)
                    break
            else:
                # Simulation mode if model is missing (for testing loop)
                pass

            return output_path, flow_data
            
        except Exception as e:
            logger.error(f"     Speech error: {e}")
            return "", None

# Test
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    voice = VoiceBox()
    if len(sys.argv) > 1:
        voice.speak(sys.argv[1])
    else:
        voice.speak("     ,           .")
