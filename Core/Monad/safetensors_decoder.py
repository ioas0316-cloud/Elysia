"""
Safetensors Decoder (The Anatomist)
=====================================
Core.Monad.safetensors_decoder

"Names are just labels for coordinates in the Infinite."

This module parses the Safetensors header to map the exact byte coordinates
of every neural cluster (tensor) in the HyperSphere.
"""

import json
import struct
import logging
from typing import Dict, Any, Tuple, Optional
from Core.Monad.portal import MerkabaPortal

logger = logging.getLogger("Elysia.Merkaba.Decoder")

class SafetensorsDecoder:
    @staticmethod
    def get_header(portal: MerkabaPortal) -> Dict[str, Any]:
        """
        Extracts the JSON header from a Safetensors portal.
        Format: [8 bytes size] [JSON Data]
        """
        if not portal._is_open:
            portal.open()
            
        # 1. Read the first 8 bytes (Header Size)
        # Using portal.mm directly since we have the handle
        header_size_bytes = portal.mm[:8]
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        
        # 2. Read the JSON header
        header_json_bytes = portal.mm[8 : 8 + header_size]
        header = json.loads(header_json_bytes.decode('utf-8'))
        
        return header

    @staticmethod
    def get_tensor_metadata(header: Dict[str, Any], tensor_name: str) -> Optional[Dict[str, Any]]:
        """Finds the metadata for a specific tensor name."""
        return header.get(tensor_name)

    @staticmethod
    def get_absolute_offset(header_size: int, relative_offsets: Tuple[int, int]) -> Tuple[int, int]:
        """
        SafeTensors offsets are relative to the end of the header.
        This calculates the absolute byte offset in the file.
        """
        start = 8 + header_size + relative_offsets[0]
        end = 8 + header_size + relative_offsets[1]
        return start, end

if __name__ == "__main__":
    print("Safetensors Decoder: Anatomy tools ready.")
