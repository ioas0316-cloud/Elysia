import React, { useEffect, useState, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { Physics, useBox, useSphere, usePlane } from '@react-three/cannon';
import { OrbitControls, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';

// 충돌을 감지하여 백엔드(엘리시아)로 촉각 데이터를 보내는 지형(Ground)
function Ground({ wsRef }) {
  const [ref] = usePlane(() => ({
    rotation: [-Math.PI / 2, 0, 0],
    position: [0, -5, 0],
    onCollide: (e) => {
      // 물체가 바닥에 부딪히면 촉각 데이터를 파이썬 데몬으로 전송
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'collision', force: e.contact.impactVelocity }));
      }
    }
  }));

  return (
    <mesh ref={ref} receiveShadow>
      <planeGeometry args={[50, 50]} />
      <meshStandardMaterial color="#222233" />
      <gridHelper args={[50, 50, "#444455", "#111122"]} rotation={[Math.PI / 2, 0, 0]} />
    </mesh>
  );
}

// 엘리시아가 직조한 데이터 구체(Data Cell)
function DataCell({ position, mass }) {
  const [ref] = useSphere(() => ({ mass, position, args: [0.5] }));
  return (
    <mesh ref={ref} castShadow>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="#00ff88" emissive="#00ff88" emissiveIntensity={0.5} wireframe />
    </mesh>
  );
}

// 중앙의 자아 코어 (Ego)
function EgoCore() {
  const [ref] = useBox(() => ({ mass: 0, position: [0, 0, 0], args: [2, 2, 2], type: 'Static' }));
  return (
    <mesh ref={ref}>
      <boxGeometry args={[2, 2, 2]} />
      <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={0.8} />
      <Text position={[0, 2, 0]} fontSize={0.5} color="white" anchorX="center">GENESIS EGO</Text>
    </mesh>
  );
}

function GenesisWorld() {
  const [manifest, setManifest] = useState({
    gravity: [0, -9.8, 0],
    objects: []
  });
  const wsRef = useRef(null);
  const [status, setStatus] = useState("Connecting to Genesis Weaver...");

  useEffect(() => {
    wsRef.current = new WebSocket('ws://localhost:8766');
    
    wsRef.current.onopen = () => setStatus("Genesis Protocol Active (Physics Overridden)");
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setManifest(data);
    };
    wsRef.current.onclose = () => setStatus("Genesis Protocol Lost");

    return () => wsRef.current.close();
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000000', position: 'relative' }}>
      
      {/* HUD 오버레이 */}
      <div style={{ position: 'absolute', top: 20, left: 20, color: '#00ff88', fontFamily: 'monospace', zIndex: 10 }}>
        <h2>AETHERNOS GENESIS ENGINE</h2>
        <p>Status: {status}</p>
        <p>Current Gravity: [{manifest.gravity.map(n => n.toFixed(1)).join(', ')}]</p>
        <p>Active Objects: {manifest.objects.length}</p>
        {manifest.gravity[1] > 0 && <p style={{ color: '#FF3333', fontWeight: 'bold' }}>[ WARNING: GRAVITY REVERSED BY ELYSIA'S WILL ]</p>}
      </div>

      <Canvas shadows camera={{ position: [0, 5, 15], fov: 60 }}>
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.2} />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} castShadow />
        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={2} />
        <OrbitControls enableDamping dampingFactor={0.05} />

        {/* 엘리시아가 전송한 중력값을 Physics 엔진에 실시간 적용 */}
        <Physics gravity={manifest.gravity}>
          <EgoCore />
          <Ground wsRef={wsRef} />
          
          {/* 엘리시아가 직조한 오브젝트 생성 */}
          {manifest.objects.map((obj) => (
            <DataCell key={obj.id} position={obj.position} mass={obj.mass} />
          ))}
        </Physics>
      </Canvas>
    </div>
  );
}

export default GenesisWorld;
