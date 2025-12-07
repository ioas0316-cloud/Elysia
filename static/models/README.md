# VRM Models Directory

This directory contains 3D avatar models in VRM format for Elysia's avatar visualization system.

## Quick Start

1. **Place your VRM model here:**
   ```
   static/models/avatar.vrm
   ```

2. **Start the avatar server:**
   ```bash
   python start_avatar_server.py
   ```

3. **Open the client:**
   - Navigate to `Core/Creativity/web/avatar.html` in your browser
   - Or use the file path: `file:///path/to/Elysia/Core/Creativity/web/avatar.html`

## About VRM Format

VRM is an open file format for 3D avatars, based on glTF 2.0.

### Creating VRM Models

You can create VRM models using:
- **VRoid Studio** (Free, Windows/Mac): https://vroid.com/studio
- **Blender** with VRM addon: https://github.com/saturday06/VRM-Addon-for-Blender
- **Unity** with VRM SDK: https://github.com/vrm-c/UniVRM

### Recommended Specifications

For best performance with Elysia avatar system:

- **Polygon Count**: 10,000 - 30,000 triangles
- **Texture Size**: 512x512 to 2048x2048
- **Bones**: Humanoid rig (standard)
- **Blendshapes** (optional but recommended):
  - Mouth: `mouth_smile`, `mouth_sad`, `mouth_open`, `mouth_wide`
  - Eyes: `eye_blink_left`, `eye_blink_right`, `eye_wide`
  - Brows: `brow_up`, `brow_down`, `brow_angry`

### Expression Mapping

The avatar server will map Elysia's emotions to VRM blendshapes:

| Emotion | Expression Parameters | VRM Blendshapes |
|---------|---------------------|-----------------|
| **Hopeful** | mouth_curve: +0.6, eye_open: 0.9 | mouth_smile, eye_wide |
| **Focused** | mouth_curve: +0.1, brow_furrow: 0.3 | brow_down, eye_wide |
| **Calm** | mouth_curve: +0.2, eye_open: 0.8 | neutral, slight_smile |
| **Introspective** | mouth_curve: -0.2, eye_open: 0.6 | mouth_neutral, eye_half |
| **Empty** | mouth_curve: -0.5, eye_open: 0.4 | mouth_sad, eye_closed |

## Current Status

- ✅ Server infrastructure complete
- ✅ Emotion → Expression mapping
- ✅ Spirit energy system (7 elements)
- ✅ WebSocket communication
- ⏳ VRM rendering (currently uses 2D shader fallback)

## Future Enhancement: Full VRM Support

To add full 3D VRM rendering:

1. Install Three.js VRM loader in `avatar.html`:
   ```html
   <script type="importmap">
   {
     "imports": {
       "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
       "@pixiv/three-vrm": "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2/lib/three-vrm.module.js"
     }
   }
   </script>
   ```

2. Load and animate VRM model based on WebSocket data

## Testing

Place a test VRM file (even a simple humanoid) in this directory and start the server to see the system in action.

For more information, see: `docs/AVATAR_SERVER_SYSTEM.md`

## Resources

- VRM Specification: https://vrm.dev/en/
- Three.js VRM: https://github.com/pixiv/three-vrm
- VRoid Hub (Free models): https://hub.vroid.com/
- VRM Consortium: https://vrm-consortium.org/
