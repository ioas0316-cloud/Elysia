"""
Quaternion Wave DNA Compression (        DNA   )
=========================================================

"DNA                  "

       :
-    2D (   ):      1      
-     4D (    ):   /     2         

     :
- 2D top-5: 0%   
- 4D top-5 2: 100%   

[NEW 2025-12-16]            
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuaternionWaveDNA")


@dataclass
class QuaternionWaveDNA:
    """
            DNA - DNA        
    
    2                     
    """
    #         (      )
    helix1_frequencies: np.ndarray
    helix1_amplitudes: np.ndarray
    helix1_phases: np.ndarray
    
    #         (      )
    helix2_frequencies: np.ndarray
    helix2_amplitudes: np.ndarray
    helix2_phases: np.ndarray
    
    #      
    original_length: int
    top_k: int
    
    def byte_size(self) -> int:
        """DNA    (bytes)"""
        #     : k * (freq + amp + phase) = k * 12 bytes
        return self.top_k * 12 * 2 + 8
    
    def compression_ratio(self, original_bytes: int) -> float:
        return original_bytes / self.byte_size()


class QuaternionCompressor:
    """
               
    
    DNA        :
    -      2         
    -          
    - 2                     
    """
    
    def __init__(self, default_top_k: int = 10):
        self.default_top_k = default_top_k
        logger.info(f"  QuaternionCompressor initialized (top_k={default_top_k})")
    
    def compress(self, text: str, top_k: int = None) -> QuaternionWaveDNA:
        """           DNA"""
        top_k = top_k or self.default_top_k
        
        sequence = np.array([ord(c) for c in text], dtype=float)
        
        # DNA        2       
        helix1 = sequence[::2]   #       
        helix2 = sequence[1::2]  #       
        
        #    FFT
        spec1 = np.fft.fft(helix1)
        spec2 = np.fft.fft(helix2)
        
        #    k    
        mag1 = np.abs(spec1)
        mag2 = np.abs(spec2)
        top1 = np.argsort(mag1)[-top_k:]
        top2 = np.argsort(mag2)[-top_k:]
        
        dna = QuaternionWaveDNA(
            helix1_frequencies=top1,
            helix1_amplitudes=np.array([mag1[i] for i in top1]),
            helix1_phases=np.array([np.angle(spec1[i]) for i in top1]),
            helix2_frequencies=top2,
            helix2_amplitudes=np.array([mag2[i] for i in top2]),
            helix2_phases=np.array([np.angle(spec2[i]) for i in top2]),
            original_length=len(text),
            top_k=top_k
        )
        
        logger.info(f"  Compressed: {len(text)} chars   {dna.byte_size()} bytes ({dna.compression_ratio(len(text)*2):.1f}x)")
        return dna
    
    def decompress(self, dna: QuaternionWaveDNA) -> str:
        """     DNA      """
        #           
        len1 = (dna.original_length + 1) // 2
        len2 = dna.original_length // 2
        
        #         
        spec1 = np.zeros(len1, dtype=complex)
        spec2 = np.zeros(len2, dtype=complex)
        
        for f, a, p in zip(dna.helix1_frequencies, dna.helix1_amplitudes, dna.helix1_phases):
            if f < len1:
                spec1[int(f)] = a * np.exp(1j * p)
        
        for f, a, p in zip(dna.helix2_frequencies, dna.helix2_amplitudes, dna.helix2_phases):
            if f < len2:
                spec2[int(f)] = a * np.exp(1j * p)
        
        #    
        helix1 = np.fft.ifft(spec1).real
        helix2 = np.fft.ifft(spec2).real
        
        #     (            )
        sequence = np.zeros(dna.original_length)
        sequence[::2] = helix1
        sequence[1::2] = helix2
        
        #      
        chars = []
        for c in sequence:
            code = int(round(abs(c)))
            try:
                if 0 <= code <= 0x10FFFF:
                    chars.append(chr(code))
                else:
                    chars.append('?')
            except:
                chars.append('?')
        
        return ''.join(chars)
    
    def calculate_accuracy(self, original: str, restored: str) -> float:
        """         """
        if len(original) != len(restored):
            return 0.0
        match = sum(1 for a, b in zip(original, restored) if a == b)
        return match / len(original) * 100


# Singleton
_compressor = None

def get_quaternion_compressor() -> QuaternionCompressor:
    global _compressor
    if _compressor is None:
        _compressor = QuaternionCompressor()
    return _compressor


# CLI / Demo
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quaternion Wave DNA Compression")
    parser.add_argument("--text", type=str, help="Text to compress")
    parser.add_argument("--top-k", type=int, default=10, help="Top K per helix")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    
    args = parser.parse_args()
    
    compressor = get_quaternion_compressor()
    
    if args.demo:
        print("\n" + "="*60)
        print("  QUATERNION WAVE DNA COMPRESSION DEMO")
        print("="*60)
        
        tests = [
            "     ",
            "               ",
            "DNA                    ",
        ]
        
        for test in tests:
            print(f"\n  : {test}")
            dna = compressor.compress(test, top_k=args.top_k)
            restored = compressor.decompress(dna)
            accuracy = compressor.calculate_accuracy(test, restored)
            
            print(f"  : {restored}")
            print(f"   : {accuracy:.1f}%")
            print(f"   : {dna.compression_ratio(len(test)*2):.1f}x")
        
        print("\n" + "="*60)
        print("  Demo complete!")
        
    elif args.text:
        dna = compressor.compress(args.text, top_k=args.top_k)
        restored = compressor.decompress(dna)
        accuracy = compressor.calculate_accuracy(args.text, restored)
        
        print(f"  : {args.text}")
        print(f"  : {restored}")
        print(f"   : {accuracy:.1f}%")
        print(f"   : {dna.compression_ratio(len(args.text)*2):.1f}x")