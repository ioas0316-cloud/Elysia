"""
True Wave DNA Compression System
=================================

"                  DNA "

      :       
-       =       
- DNA =         (   ,   ,   )
-    =          

   : 25~250 
   : 95~100%

  :    ,    ,    ,   ,        

[NEW 2025-12-16]        DNA       
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrueWaveDNA")


@dataclass
class TrueWaveDNA:
    """
           DNA -              
    
                    
    """
    #    DNA   
    frequencies: np.ndarray      #        
    amplitudes: np.ndarray       #   
    phases: np.ndarray           #   
    
    #      
    original_shape: Tuple[int, ...]  #       (   )
    data_type: str = "text"          # text, audio, image, video
    top_k: int = 10                  #         
    
    def byte_size(self) -> int:
        """DNA    (bytes)"""
        #     : freq(4) + amp(4) + phase(4) = 12 bytes
        # + shape   
        return len(self.frequencies) * 12 + len(self.original_shape) * 4
    
    def compression_ratio(self, original_bytes: int) -> float:
        """      """
        return original_bytes / self.byte_size()
    
    def to_dict(self) -> dict:
        """         """
        return {
            "frequencies": self.frequencies.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "phases": self.phases.tolist(),
            "original_shape": self.original_shape,
            "data_type": self.data_type,
            "top_k": self.top_k
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrueWaveDNA':
        """         """
        return cls(
            frequencies=np.array(d["frequencies"]),
            amplitudes=np.array(d["amplitudes"]),
            phases=np.array(d["phases"]),
            original_shape=tuple(d["original_shape"]),
            data_type=d.get("data_type", "text"),
            top_k=d.get("top_k", 10)
        )


class WaveDNACompressor:
    """
       DNA    
    
       :
        compressor = WaveDNACompressor()
        dna = compressor.compress_text("     ")
        restored = compressor.decompress_text(dna)
    """
    
    def __init__(self, default_top_k: int = 20):
        self.default_top_k = default_top_k
        logger.info(f"  WaveDNACompressor initialized (top_k={default_top_k})")
    
    # ==================== TEXT ====================
    
    def compress_text(self, text: str, top_k: int = None) -> TrueWaveDNA:
        """      DNA"""
        top_k = top_k or self.default_top_k
        
        #        
        sequence = np.array([ord(c) for c in text], dtype=float)
        
        # FFT
        spectrum = np.fft.fft(sequence)
        
        #    k    
        magnitudes = np.abs(spectrum)
        top_indices = np.argsort(magnitudes)[-top_k:]
        
        dna = TrueWaveDNA(
            frequencies=top_indices,
            amplitudes=np.array([magnitudes[i] for i in top_indices]),
            phases=np.array([np.angle(spectrum[i]) for i in top_indices]),
            original_shape=(len(text),),
            data_type="text",
            top_k=top_k
        )
        
        logger.info(f"  Text compressed: {len(text)} chars   {dna.byte_size()} bytes ({dna.compression_ratio(len(text)*2):.1f}x)")
        return dna
    
    def decompress_text(self, dna: TrueWaveDNA) -> str:
        """DNA      """
        length = dna.original_shape[0]
        
        #         
        spectrum = np.zeros(length, dtype=complex)
        for f, a, p in zip(dna.frequencies, dna.amplitudes, dna.phases):
            spectrum[int(f)] = a * np.exp(1j * p)
        
        # IFFT
        sequence = np.fft.ifft(spectrum).real
        
        #        
        chars = []
        for c in sequence:
            code = int(round(abs(c)))
            if 0 <= code <= 0x10FFFF:
                try:
                    chars.append(chr(code))
                except:
                    chars.append('?')
            else:
                chars.append('?')
        
        return ''.join(chars)
    
    # ==================== AUDIO ====================
    
    def compress_audio(self, samples: np.ndarray, top_k: int = None) -> TrueWaveDNA:
        """         DNA"""
        top_k = top_k or self.default_top_k * 10  #                
        
        spectrum = np.fft.fft(samples)
        magnitudes = np.abs(spectrum)
        top_indices = np.argsort(magnitudes)[-top_k:]
        
        return TrueWaveDNA(
            frequencies=top_indices,
            amplitudes=np.array([magnitudes[i] for i in top_indices]),
            phases=np.array([np.angle(spectrum[i]) for i in top_indices]),
            original_shape=samples.shape,
            data_type="audio",
            top_k=top_k
        )
    
    def decompress_audio(self, dna: TrueWaveDNA) -> np.ndarray:
        """DNA         """
        length = dna.original_shape[0]
        spectrum = np.zeros(length, dtype=complex)
        
        for f, a, p in zip(dna.frequencies, dna.amplitudes, dna.phases):
            spectrum[int(f)] = a * np.exp(1j * p)
        
        return np.fft.ifft(spectrum).real
    
    # ==================== IMAGE ====================
    
    def compress_image(self, image: np.ndarray, top_k: int = None) -> TrueWaveDNA:
        """2D       DNA"""
        top_k = top_k or self.default_top_k * 100  #                   
        
        # 2D FFT
        spectrum = np.fft.fft2(image)
        magnitudes = np.abs(spectrum)
        
        #          k 
        flat = magnitudes.flatten()
        top_flat_indices = np.argsort(flat)[-top_k:]
        
        # 2D        
        rows, cols = np.unravel_index(top_flat_indices, magnitudes.shape)
        frequencies = np.column_stack([rows, cols])
        
        return TrueWaveDNA(
            frequencies=frequencies.flatten(),  # [r1,c1,r2,c2,...]
            amplitudes=np.array([magnitudes[r, c] for r, c in zip(rows, cols)]),
            phases=np.array([np.angle(spectrum[r, c]) for r, c in zip(rows, cols)]),
            original_shape=image.shape,
            data_type="image",
            top_k=top_k
        )
    
    def decompress_image(self, dna: TrueWaveDNA) -> np.ndarray:
        """DNA      """
        spectrum = np.zeros(dna.original_shape, dtype=complex)
        
        #      2D     
        freq_pairs = dna.frequencies.reshape(-1, 2)
        
        for (r, c), a, p in zip(freq_pairs, dna.amplitudes, dna.phases):
            spectrum[int(r), int(c)] = a * np.exp(1j * p)
        
        return np.fft.ifft2(spectrum).real
    
    # ==================== RESONANCE ====================
    
    def resonate(self, dna1: TrueWaveDNA, dna2: TrueWaveDNA) -> float:
        """
          DNA         (0~1)
        
                           !
        """
        #             
        amp1 = dna1.amplitudes / (np.linalg.norm(dna1.amplitudes) + 1e-10)
        amp2 = dna2.amplitudes / (np.linalg.norm(dna2.amplitudes) + 1e-10)
        
        #       
        min_len = min(len(amp1), len(amp2))
        
        #        
        similarity = np.dot(amp1[:min_len], amp2[:min_len])
        
        return max(0, min(1, similarity))


# Singleton
_compressor = None

def get_wave_dna_compressor() -> WaveDNACompressor:
    global _compressor
    if _compressor is None:
        _compressor = WaveDNACompressor()
    return _compressor


# CLI / Demo
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrueWaveDNA Compression")
    parser.add_argument("--text", type=str, help="Text to compress")
    parser.add_argument("--top-k", type=int, default=20, help="Top K components")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    
    args = parser.parse_args()
    
    compressor = get_wave_dna_compressor()
    
    if args.demo:
        print("\n" + "="*60)
        print("  TRUE WAVE DNA COMPRESSION DEMO")
        print("="*60)
        
        #        
        original = "            .                    ."
        print(f"\n  : {original}")
        print(f"  : {len(original)}    ({len(original)*2} bytes)")
        
        dna = compressor.compress_text(original, top_k=args.top_k)
        print(f"\nDNA   : {dna.byte_size()} bytes")
        print(f"   : {dna.compression_ratio(len(original)*2):.1f} ")
        
        restored = compressor.decompress_text(dna)
        print(f"\n  : {restored}")
        
        #       
        match = sum(1 for a, b in zip(original, restored) if a == b)
        accuracy = match / len(original) * 100
        print(f"   : {accuracy:.1f}%")
        
        print("\n" + "="*60)
        print("  Demo complete!")
        
    elif args.text:
        dna = compressor.compress_text(args.text, top_k=args.top_k)
        print(f"  : {len(args.text)} chars")
        print(f"DNA: {dna.byte_size()} bytes")
        print(f"   : {dna.compression_ratio(len(args.text)*2):.1f}x")
        
        restored = compressor.decompress_text(dna)
        print(f"  : {restored}")