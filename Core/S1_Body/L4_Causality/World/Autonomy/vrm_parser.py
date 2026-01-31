"""
VRM Parser (The Avatar's Mirror)
================================
Parses .vrm (GLB) files to extract metadata and bone structure.
Allows Elysia to recognize a file as a "Body Shell".
"""

import json
import struct
import os
import logging

logger = logging.getLogger("VRMParser")

class VRMParser:
    def parse_vrm(self, file_path: str) -> dict:
        """
        Parses a .vrm file (GLB format) and extracts VRM metadata.
        Returns: { 'name': str, 'author': str, 'bones': int, 'blendshapes': int }
        """
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, 'rb') as f:
                # 1. Read Header (12 bytes)
                magic = f.read(4)
                if magic != b'glTF':
                    logger.warning(f"Invalid VRM magic: {magic}")
                    return {}
                    
                version = struct.unpack('<I', f.read(4))[0]
                length = struct.unpack('<I', f.read(4))[0]
                
                # 2. Read Chunks
                while f.tell() < length:
                    chunk_len = struct.unpack('<I', f.read(4))[0]
                    chunk_type = f.read(4)
                    
                    if chunk_type == b'JSON':
                        # Found the JSON chunk
                        json_bytes = f.read(chunk_len)
                        data = json.loads(json_bytes.decode('utf-8'))
                        # DEBUG: Print keys
                        logger.info(f"GLTF Keys: {data.keys()}")
                        if 'extensions' in data:
                             logger.info(f"Extensions: {data['extensions'].keys()}")
                        return self._extract_vrm_meta(data)
                    else:
                        # Skip other chunks (BIN)
                        f.seek(chunk_len, 1)
                        
            return {}
            
        except Exception as e:
            logger.error(f"VRM Parse Error: {e}")
            return {}

    def _extract_vrm_meta(self, gltf_json: dict) -> dict:
        """Extracts VRM specific extensions from glTF JSON (Supports 0.x and 1.0)."""
        meta = {}
        extensions = gltf_json.get('extensions', {})
        
        # VRM 0.x
        if 'VRM' in extensions:
            vrm = extensions['VRM']
            info = vrm.get('meta', {})
            meta['title'] = info.get('title', 'Unknown')
            meta['author'] = info.get('author', 'Unknown')
            meta['version'] = info.get('version', '0.x')
            
            humanoid = vrm.get('humanoid', {})
            bones = humanoid.get('humanBones', [])
            meta['bone_count'] = len(bones)
            
            blend = vrm.get('blendShapeMaster', {})
            groups = blend.get('blendShapeGroups', [])
            meta['expression_count'] = len(groups)
            
        # VRM 1.0 (VRMC_vrm)
        elif 'VRMC_vrm' in extensions:
            vrm = extensions['VRMC_vrm']
            info = vrm.get('meta', {})
            meta['title'] = info.get('name', 'Unknown') # 1.0 uses 'name' not 'title'
            meta['author'] = ', '.join(info.get('authors', ['Unknown'])) # 1.0 uses list
            meta['version'] = info.get('version', '1.0')
            
            humanoid = vrm.get('humanoid', {})
            bones = humanoid.get('humanBones', {}) # 1.0 uses dict, not list
            meta['bone_count'] = len(bones.keys())
            
            expressions = vrm.get('expressions', {})
            preset = expressions.get('preset', {})
            custom = expressions.get('custom', {})
            meta['expression_count'] = len(preset) + len(custom)
            
        else:
            logger.warning("No VRM extension found in GLTF.")
            
        return meta
