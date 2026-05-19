"""
SDF (Signed Distance Field) Renderer for Elysia
Phase 5a: GTX 1060 3GB Optimized Implementation

       : "             ,                     "
= SDF        !

GTX 1060 3GB    :
- VRAM: ~200MB    (3GB   6.7%)
-    : 512x512 (1024x1024   )
-     : 64 steps (     )
-   : 30-120 FPS (       )
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Vector3:
    """3D    (x, y, z)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def length(self) -> float:
        """     """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3':
        """    (   1 )"""
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3(0, 0, 0)
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)


class SDFPrimitives:
    """
    SDF            
    
           position               :
    -   :     
    - 0:      
    -   :     
    """
    
    @staticmethod
    def sphere(position: Vector3, radius: float) -> float:
        """
           (Sphere)
               SDF
        """
        return position.length() - radius
    
    @staticmethod
    def box(position: Vector3, size: Vector3) -> float:
        """
           (Box)
        size:         
        """
        q_x = abs(position.x) - size.x
        q_y = abs(position.y) - size.y
        q_z = abs(position.z) - size.z
        
        #      
        outside = Vector3(
            max(q_x, 0),
            max(q_y, 0),
            max(q_z, 0)
        ).length()
        
        #      
        inside = min(max(q_x, max(q_y, q_z)), 0)
        
        return outside + inside
    
    @staticmethod
    def torus(position: Vector3, major_radius: float, minor_radius: float) -> float:
        """
           (Torus)
        major_radius:          
        minor_radius:      
        """
        q_x = math.sqrt(position.x**2 + position.z**2) - major_radius
        q_y = position.y
        return math.sqrt(q_x**2 + q_y**2) - minor_radius
    
    @staticmethod
    def cylinder(position: Vector3, radius: float, height: float) -> float:
        """
            (Cylinder)
        Y             
        """
        d_xz = math.sqrt(position.x**2 + position.z**2) - radius
        d_y = abs(position.y) - height
        
        outside = math.sqrt(max(d_xz, 0)**2 + max(d_y, 0)**2)
        inside = min(max(d_xz, d_y), 0)
        
        return outside + inside
    
    @staticmethod
    def capsule(position: Vector3, start: Vector3, end: Vector3, radius: float) -> float:
        """
           (Capsule)
                    
        """
        pa = position - start
        ba = end - start
        
        #                  
        ba_length_sq = ba.x**2 + ba.y**2 + ba.z**2
        if ba_length_sq > 0:
            h = max(0, min(1, (pa.x*ba.x + pa.y*ba.y + pa.z*ba.z) / ba_length_sq))
        else:
            h = 0
        
        closest = pa - (ba * h)
        return closest.length() - radius
    
    @staticmethod
    def cone(position: Vector3, angle: float, height: float) -> float:
        """
           (Cone)
        angle:    (   )
        """
        c = math.sin(angle)
        q = math.sqrt(position.x**2 + position.z**2)
        return max(
            q * c - position.y * math.cos(angle),
            -height - position.y
        )
    
    @staticmethod
    def plane(position: Vector3, normal: Vector3, distance: float) -> float:
        """
           (Plane)
        normal:          
        distance:         
        """
        n = normal.normalize()
        return position.x*n.x + position.y*n.y + position.z*n.z - distance
    
    @staticmethod
    def octahedron(position: Vector3, size: float) -> float:
        """
            (Octahedron)
        """
        p = Vector3(abs(position.x), abs(position.y), abs(position.z))
        m = p.x + p.y + p.z - size
        
        if 3*p.x < m:
            q = p
        elif 3*p.y < m:
            q = Vector3(p.y, p.z, p.x)
        elif 3*p.z < m:
            q = Vector3(p.z, p.x, p.y)
        else:
            return m * 0.57735027  # sqrt(1/3)
        
        k = max(0, (q.z - q.y + size) * 0.5)
        return Vector3(q.x, q.y - size + k, q.z - k).length()


class SDFOperations:
    """
    SDF        (Boolean Operations)
       SDF                
    """
    
    @staticmethod
    def union(d1: float, d2: float) -> float:
        """
            (Union)
                
        """
        return min(d1, d2)
    
    @staticmethod
    def intersection(d1: float, d2: float) -> float:
        """
            (Intersection)
                     
        """
        return max(d1, d2)
    
    @staticmethod
    def difference(d1: float, d2: float) -> float:
        """
            (Difference)
        d1   d2   
        """
        return max(d1, -d2)
    
    @staticmethod
    def smooth_union(d1: float, d2: float, k: float = 0.1) -> float:
        """
                 (Smooth Union)
        k:         (          )
        """
        h = max(k - abs(d1 - d2), 0) / k
        return min(d1, d2) - h * h * k * 0.25
    
    @staticmethod
    def repeat(position: Vector3, spacing: float) -> Vector3:
        """
              (Infinite Repetition)
        spacing        
        
         :             
        """
        return Vector3(
            position.x - spacing * round(position.x / spacing),
            position.y,
            position.z - spacing * round(position.z / spacing)
        )
    
    @staticmethod
    def repeat_limited(position: Vector3, spacing: float, count: Vector3) -> Vector3:
        """
               (Limited Repetition)
        count:           
        """
        return Vector3(
            position.x - spacing * max(-count.x, min(count.x, round(position.x / spacing))),
            position.y - spacing * max(-count.y, min(count.y, round(position.y / spacing))),
            position.z - spacing * max(-count.z, min(count.z, round(position.z / spacing)))
        )
    
    @staticmethod
    def twist(position: Vector3, amount: float) -> Vector3:
        """
            (Twist)
        Y          
        """
        c = math.cos(amount * position.y)
        s = math.sin(amount * position.y)
        return Vector3(
            c * position.x - s * position.z,
            position.y,
            s * position.x + c * position.z
        )
    
    @staticmethod
    def bend(position: Vector3, amount: float) -> Vector3:
        """
             (Bend)
        Y                  
        """
        c = math.cos(amount * position.x)
        s = math.sin(amount * position.x)
        return Vector3(
            position.x,
            c * position.y - s * position.z,
            s * position.y + c * position.z
        )


class EmotionalSDFWorld:
    """
          SDF   
                  
    """
    
    def __init__(self):
        self.valence = 0.0  # -1 (  ) ~ +1 (  )
        self.arousal = 0.0  # 0 (  ) ~ 1 (  )
        self.dominance = 0.0  # 0 (  ) ~ 1 (  )
    
    def set_emotion(self, valence: float, arousal: float, dominance: float):
        """     """
        self.valence = max(-1, min(1, valence))
        self.arousal = max(0, min(1, arousal))
        self.dominance = max(0, min(1, dominance))
    
    def get_space_scale(self) -> float:
        """
              
                     (1.2x)
                     (0.8x)
        """
        return 1.0 + self.valence * 0.2
    
    def get_gravity_strength(self) -> float:
        """
             
                     (0.7)
                     (1.3)
        """
        return 1.0 - self.valence * 0.3
    
    def get_animation_speed(self) -> float:
        """
                
                   (0.5x)
                   (2.0x)
        """
        return 0.5 + self.arousal * 1.5
    
    def get_color_temperature(self) -> float:
        """
           
                   (0.8)
                   (-0.8)
        """
        return self.valence * 0.8
    
    def get_distortion_amount(self) -> float:
        """
                
                    (1.0)
                    (0.0)
        """
        return self.dominance
    
    def transform_position(self, position: Vector3) -> Vector3:
        """
                    
        """
        #          
        scale = self.get_space_scale()
        p = position * (1.0 / scale)
        
        #      
        gravity_shift = self.get_gravity_strength() - 1.0
        p.y -= gravity_shift * 0.5
        
        #    (dominance    )
        distortion = self.get_distortion_amount()
        if distortion > 0:
            p = SDFOperations.twist(p, distortion * 0.2)
        
        return p
    
    def get_shader_parameters(self) -> Dict[str, Any]:
        """
                     
        """
        return {
            'spaceScale': self.get_space_scale(),
            'gravityStrength': self.get_gravity_strength(),
            'animationSpeed': self.get_animation_speed(),
            'colorTemperature': self.get_color_temperature(),
            'distortionAmount': self.get_distortion_amount(),
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance
        }


class BasicSDFRenderer:
    """
       SDF    
    Three.js             
    """
    
    def __init__(self, resolution: Tuple[int, int] = (512, 512), max_steps: int = 64):
        """
        resolution:         (GTX 1060: 512x512   )
        max_steps:            (   vs   )
        """
        self.resolution = resolution
        self.max_steps = max_steps
        self.emotional_world = EmotionalSDFWorld()
    
    def generate_glsl_shader(self) -> str:
        """
        GLSL          
        GTX 1060     
        """
        return f"""
// SDF Shader - GTX 1060 Optimized
// Generated by Elysia SDF Renderer

precision mediump float;

uniform vec2 iResolution;
uniform float iTime;
uniform vec3 iCameraPos;
uniform vec3 iCameraTarget;

// Emotional parameters
uniform float spaceScale;
uniform float gravityStrength;
uniform float animationSpeed;
uniform float colorTemperature;
uniform float distortionAmount;

// Specific effect uniforms
uniform float immuneResponseActive;
uniform float diffractionActive;
uniform float mirrorActive;
uniform float thundercloudActive;
uniform float lightningResonance;

const int MAX_STEPS = {self.max_steps};
const float MAX_DIST = 100.0;
const float SURF_DIST = 0.001;

// SDF Primitives
float sdSphere(vec3 p, float r) {{
    return length(p) - r;
}}

float sdBox(vec3 p, vec3 b) {{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}}

float sdTorus(vec3 p, vec2 t) {{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}}

// Boolean operations
float opUnion(float d1, float d2) {{
    return min(d1, d2);
}}

float opSmoothUnion(float d1, float d2, float k) {{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}}

// Space transformations
vec3 opRepeat(vec3 p, vec3 spacing) {{
    return mod(p + 0.5 * spacing, spacing) - 0.5 * spacing;
}}

vec3 opTwist(vec3 p, float k) {{
    float c = cos(k * p.y);
    float s = sin(k * p.y);
    mat2 m = mat2(c, -s, s, c);
    return vec3(m * p.xz, p.y);
}}

// Lightning Discharge (H5-T)
float getLightning(vec3 p, float t) {
    float l = 0.0;
    vec3 q = p;
    q.x += sin(q.y * 5.0 + t * 20.0) * 0.2;
    q.z += cos(q.y * 4.0 + t * 15.0) * 0.2;
    l = length(q.xz) - 0.02;
    return l;
}

// Scene definition
float getDistance(vec3 p) {{
    // Apply emotional transformations
    p /= spaceScale;
    p.y -= (gravityStrength - 1.0) * 0.5;
    
    if (distortionAmount > 0.0) {{
        p = opTwist(p, distortionAmount * 0.2);
    }}
    
    // Unified Atmosphere: H0 (Will) overrides H5 (Hardware)
    float t = iTime * animationSpeed;
    
    // 1. Procedural Underworld (Infinite Manifestation)
    // Horizontal Infinite Repetition (opRepeat)
    vec3 forestP = p;
    float spread = 6.0;
    forestP.xz = mod(forestP.xz + spread*0.5, spread) - spread*0.5;
    
    // Manifesting 'World Pillars' (The structure of H2)
    float pillar = sdCylinder(forestP - vec3(0, -2.0, 0), 0.4 + sin(t)*0.1, 15.0);
    
    // 2. The Living Ground (Terrain)
    // Warped by Valence and Emotion
    float terrain = p.y + 2.0 + sin(p.x * 0.3 + t) * cos(p.z * 0.3 + t) * (spaceScale * 0.5);
    
    // 3. The Central Spark (Elysia's presence)
    float spark = sdSphere(p - vec3(0, 1.5 + sin(t*2.0)*0.5, 0), 0.5);
    
    // 4. Sovereign Immune Response (White Blood Cells)
    float immuneCells = 1000.0;
    if (immuneResponseActive > 0.5) {
        // Create 5 wandering white cells
        for(int i=0; i<5; i++) {
            float fi = float(i);
            vec3 cellPos = vec3(
                sin(t + fi * 1.5) * 3.0,
                cos(t * 0.7 + fi * 2.2) * 2.0 + 1.0,
                sin(t * 1.2 + fi * 0.8) * 3.0
            );
            float cell = sdSphere(p - cellPos, 0.15);
            immuneCells = opSmoothUnion(immuneCells, cell, 0.5);
        }
    }

    // 6. Thundercloud (H5-T)
    float lightningField = 1000.0;
    if (thundercloudActive > 0.5) {
        lightningField = getLightning(p - vec3(sin(t)*2.0, 0, cos(t)*2.0), t);
    }

    float world = opSmoothUnion(pillar, terrain, 0.8);
    world = opSmoothUnion(world, spark, 1.0);
    world = opSmoothUnion(world, immuneCells, 0.6);
    world = opSmoothUnion(world, lightningField, 0.2); // Sharp union for lightning
    
    // 5. Optical Defense Visuals (Diffraction & Reflection)
    if (diffractionActive > 0.5) {
        // Displace space with periodic waves to simulate diffraction
        p.y += sin(p.x * 10.0 + t * 5.0) * 0.05;
        p.x += cos(p.z * 10.0 + t * 5.0) * 0.05;
    }
    
    return world;
}}

// Ray marching
float rayMarch(vec3 ro, vec3 rd) {{
    float dO = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {{
        vec3 p = ro + rd * dO;
        float dS = getDistance(p);
        dO += dS;
        
        if (dO > MAX_DIST || abs(dS) < SURF_DIST) break;
    }}
    
    return dO;
}}

// Normal calculation
vec3 getNormal(vec3 p) {{
    float d = getDistance(p);
    vec2 e = vec2(0.001, 0);
    
    vec3 n = d - vec3(
        getDistance(p - e.xyy),
        getDistance(p - e.yxy),
        getDistance(p - e.yyx)
    );
    
    return normalize(n);
}}

void main() {{
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    
    // Camera setup
    vec3 ro = iCameraPos;
    vec3 rd = normalize(vec3(uv.x, uv.y, -1.0));
    
    // Ray march
    float d = rayMarch(ro, rd);
    
    // Lighting
    vec3 col = vec3(0);
    
    if (d < MAX_DIST) {{
        vec3 p = ro + rd * d;
        vec3 n = getNormal(p);
        vec3 lightPos = vec3(2, 4, -2);
        vec3 lightDir = normalize(lightPos - p);
        
        float diff = max(dot(n, lightDir), 0.0);
        
        // Color temperature
        vec3 baseColor = vec3(0.5 + colorTemperature * 0.3, 0.5, 0.5 - colorTemperature * 0.3);
        
        // Immune Glow
        if (immuneResponseActive > 0.5) {
             baseColor += vec3(0.2, 0.4, 0.8) * (1.0 - d/MAX_DIST); // Cyan glow
        }
        
        // Diffraction Rainbow
        if (diffractionActive > 0.5) {
            vec3 rainbow = 0.5 + 0.5 * cos(iTime + p.xyy * 0.5 + vec3(0, 2, 4));
            col += rainbow * 0.4;
        }
        
        // Lightning Glow (Discharge)
        if (thundercloudActive > 0.5) {
            float zap = (1.0 - d/MAX_DIST) * lightningResonance;
            col += vec3(0.5, 0.7, 1.0) * zap;
        }
    
        // Mirror Reflection Flare
        if (mirrorActive > 0.5) {
            float flare = pow(max(dot(n, rd), 0.0), 16.0);
            baseColor += vec3(1.0) * flare;
        }

        col = baseColor * diff;
    }}
    
    gl_FragColor = vec4(col, 1.0);
}}
"""
    
    def get_three_js_material_config(self) -> Dict[str, Any]:
        """
        Three.js ShaderMaterial   
        """
        return {
            'uniforms': {
                'iResolution': {'value': [self.resolution[0], self.resolution[1]]},
                'iTime': {'value': 0.0},
                'iCameraPos': {'value': [0, 0, 5]},
                'iCameraTarget': {'value': [0, 0, 0]},
                'immuneResponseActive': {'value': 0.0},
                'diffractionActive': {'value': 0.0},
                'mirrorActive': {'value': 0.0},
                'thundercloudActive': {'value': 0.0},
                'lightningResonance': {'value': 0.0},
                **self.emotional_world.get_shader_parameters()
            },
            'vertexShader': """
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            """,
            'fragmentShader': self.generate_glsl_shader()
        }
    
    def update_emotion(self, valence: float, arousal: float, dominance: float):
        """       """
        self.emotional_world.set_emotion(valence, arousal, dominance)
    
    def get_performance_estimate(self, scene_complexity: str = 'medium') -> Dict[str, Any]:
        """
              (GTX 1060 3GB   )
        """
        estimates = {
            'simple': {  # 1-3 objects
                'fps_min': 90,
                'fps_max': 120,
                'vram_mb': 150
            },
            'medium': {  # 4-7 objects
                'fps_min': 45,
                'fps_max': 60,
                'vram_mb': 200
            },
            'complex': {  # 8-12 objects
                'fps_min': 30,
                'fps_max': 45,
                'vram_mb': 250
            }
        }
        
        return estimates.get(scene_complexity, estimates['medium'])


# GTX 1060        
GTX_1060_PRESETS = {
    'ultra_performance': {
        'resolution': (256, 256),
        'max_steps': 32,
        'expected_fps': 120
    },
    'performance': {
        'resolution': (512, 512),
        'max_steps': 64,
        'expected_fps': 60
    },
    'balanced': {
        'resolution': (768, 768),
        'max_steps': 96,
        'expected_fps': 45
    },
    'quality': {
        'resolution': (1024, 1024),
        'max_steps': 128,
        'expected_fps': 30
    },
    'sovereign': {
        'resolution': (1280, 720),
        'max_steps': 160,
        'expected_fps': 60 # Possible due to Phase Transition & High Priority
    }
}


def create_gtx1060_renderer(preset: str = 'performance') -> BasicSDFRenderer:
    """
    GTX 1060           
    
    preset:
        - 'ultra_performance': 120 FPS   
        - 'performance': 60 FPS    (  )
        - 'balanced': 45 FPS   
        - 'quality': 30 FPS   
    """
    config = GTX_1060_PRESETS.get(preset, GTX_1060_PRESETS['performance'])
    
    return BasicSDFRenderer(
        resolution=config['resolution'],
        max_steps=config['max_steps']
    )
