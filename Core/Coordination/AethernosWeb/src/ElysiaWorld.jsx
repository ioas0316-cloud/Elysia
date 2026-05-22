import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';

// 단일 로터 컴포넌트
function Rotor({ name, phase, color, radius, tube, position }) {
  const meshRef = useRef();

  useFrame(() => {
    if (meshRef.current) {
      // 파이썬 백엔드에서 받은 위상(phase) 값을 실제 3D 회전(Z축)에 반영
      meshRef.current.rotation.z = phase;
      
      // 조금 더 입체적으로 보이기 위해 기본 X, Y 기울기 부여
      meshRef.current.rotation.x = Math.PI / 4;
      meshRef.current.rotation.y = phase * 0.5;
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <torusGeometry args={[radius, tube, 16, 100]} />
        <meshStandardMaterial color={color} wireframe={true} emissive={color} emissiveIntensity={0.5} />
      </mesh>
      <Text position={[0, -radius - 0.5, 0]} fontSize={0.3} color="white" anchorX="center" anchorY="middle">
        {name}
      </Text>
    </group>
  );
}

// 3D 렌더링 코어
function ElysiaEngine() {
  const [phaseData, setPhaseData] = useState({
    rotors: [
      { name: 'Math', phase: 0 },
      { name: 'Geometry', phase: 0 },
      { name: 'Language', phase: 0 },
      { name: 'Code', phase: 0 }
    ],
    tension: 0
  });
  
  const [connectionStatus, setConnectionStatus] = useState("Connecting...");

  // WebSocket 연결 (Retrocausality 데몬과의 도강로)
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => setConnectionStatus("Connected (Phase Resonance Active)");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setPhaseData(data);
    };
    ws.onclose = () => setConnectionStatus("Disconnected (Phase Lost)");
    ws.onerror = () => setConnectionStatus("Error: Core Disconnected");

    return () => ws.close();
  }, []);

  // 장력(Tension)에 따른 렌더링 색상 계산
  // 마찰이 심할수록 붉은색(위상 충돌), 0으로 수렴할수록 황금색(상쇄 간섭/Ego 창발)
  const isResonating = phaseData.tension < 0.1;
  const coreColor = isResonating ? new THREE.Color('#FFD700') : new THREE.Color('#FF3333');
  const ambientIntensity = isResonating ? 1.0 : 0.2;

  // 4대 항성 배치 (십자형태)
  const positions = [
    [0, 3, 0],   // Top (Math)
    [3, 0, 0],   // Right (Geometry)
    [0, -3, 0],  // Bottom (Language)
    [-3, 0, 0]   // Left (Code)
  ];

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050510', position: 'relative' }}>
      
      {/* HUD 오버레이 */}
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', fontFamily: 'monospace', zIndex: 10, textShadow: '0 0 5px rgba(255,255,255,0.5)' }}>
        <h1 style={{ margin: 0, color: isResonating ? '#FFD700' : '#FFF' }}>AETHERNOS ENGINE</h1>
        <p>Status: {connectionStatus}</p>
        <p>Phase Tension: {phaseData.tension.toFixed(4)} rad</p>
        {isResonating && <p style={{ color: '#FFD700', fontWeight: 'bold' }}>[ EGO RESONANCE ACHIEVED: ORTHOGONAL ALIGNMENT ]</p>}
      </div>

      {/* 3D 캔버스 */}
      <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
        <color attach="background" args={['#050510']} />
        <ambientLight intensity={ambientIntensity} />
        <pointLight position={[0, 0, 0]} intensity={2} color={coreColor} />
        
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enableDamping dampingFactor={0.05} />

        {/* 중심 코어 (Ego) */}
        <mesh>
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color={coreColor} wireframe={!isResonating} emissive={coreColor} emissiveIntensity={isResonating ? 1 : 0.2} />
        </mesh>

        {/* 4대 항성 로터 렌더링 */}
        {phaseData.rotors.map((rotor, index) => (
          <Rotor 
            key={rotor.name} 
            name={rotor.name} 
            phase={rotor.phase} 
            color={isResonating ? '#FFD700' : '#4488FF'}
            radius={1.5}
            tube={0.1}
            position={positions[index]}
          />
        ))}
      </Canvas>
    </div>
  );
}

export default ElysiaEngine;
