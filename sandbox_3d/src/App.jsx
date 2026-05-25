import React, { useRef, useMemo, useEffect, useState, Suspense } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from '@pixiv/three-vrm'

// 1. 다층적 대지 (Layered Terrain)
function LayeredTerrain() {
  const meshRef = useRef()
  const geometry = useMemo(() => new THREE.PlaneGeometry(100, 100, 150, 150), [])

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    const positions = meshRef.current.geometry.attributes.position
    
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i)
      const y = positions.getY(i)
      
      const wave1 = Math.sin(x * 0.2 + time) * 0.8
      const wave2 = Math.cos(y * 0.2 + time * 0.5) * 0.8
      const wave3 = Math.sin(Math.sqrt(x * x + y * y) * 0.3 - time) * 0.5
      
      positions.setZ(i, wave1 + wave2 + wave3)
    }
    meshRef.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <group rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
      <mesh ref={meshRef} receiveShadow>
        <bufferGeometry attach="geometry" {...geometry} />
        <meshStandardMaterial color="#3a734a" roughness={0.9} metalness={0.1} />
      </mesh>
    </group>
  )
}

// 2. 강 (Flowing River)
function River() {
  const riverRef = useRef()
  const geometry = useMemo(() => new THREE.PlaneGeometry(20, 100, 50, 100), [])
  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    const positions = riverRef.current.geometry.attributes.position
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i)
      const y = positions.getY(i)
      positions.setZ(i, Math.sin(x * 2 + time * 3) * 0.3 + Math.cos(y + time) * 0.2)
    }
    riverRef.current.geometry.attributes.position.needsUpdate = true
  })
  return (
    <mesh ref={riverRef} rotation={[-Math.PI / 2, 0, 0]} position={[25, -1.5, 0]} receiveShadow>
      <bufferGeometry attach="geometry" {...geometry} />
      <meshStandardMaterial color="#2288ff" transparent opacity={0.8} roughness={0.1} metalness={0.5} />
    </mesh>
  )
}

// 3. 산맥 (Mountains)
function Mountains() {
  return (
    <group position={[0, -2, -30]}>
      {Array.from({ length: 6 }).map((_, i) => (
        <mesh key={i} position={[(i - 2.5) * 15, 8, Math.abs(i-2.5) * 5]} castShadow receiveShadow>
          <coneGeometry args={[12, 25, 8]} />
          <meshStandardMaterial color="#1f3d14" roughness={0.9} />
        </mesh>
      ))}
    </group>
  )
}

// 4. 구조화된 안식처 (Furnished House)
function FurnishedHouse({ sendInteraction, setTargetPos }) {
  const interact = (e, name, pos) => {
    e.stopPropagation()
    sendInteraction(name)
    setTargetPos(new THREE.Vector3(...pos))
  }
  
  return (
    <group position={[10, 0, -10]} castShadow receiveShadow>
      {/* 바닥 */}
      <mesh position={[0, 0.1, 0]} receiveShadow>
        <boxGeometry args={[12, 0.2, 12]} />
        <meshStandardMaterial color="#5c4033" roughness={0.8} />
      </mesh>
      {/* 뒷벽 */}
      <mesh position={[0, 2.5, -5.9]} castShadow receiveShadow>
        <boxGeometry args={[12, 5, 0.2]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      {/* 왼벽 */}
      <mesh position={[-5.9, 2.5, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.2, 5, 12]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      {/* 오른벽 */}
      <mesh position={[5.9, 2.5, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.2, 5, 12]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      {/* 앞벽 (가운데 문 뚫림) */}
      <mesh position={[-3.5, 2.5, 5.9]} castShadow receiveShadow>
        <boxGeometry args={[5, 5, 0.2]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      <mesh position={[4.0, 2.5, 5.9]} castShadow receiveShadow>
        <boxGeometry args={[4, 5, 0.2]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      <mesh position={[0.5, 4.0, 5.9]} castShadow receiveShadow>
        <boxGeometry args={[3, 2, 0.2]} />
        <meshStandardMaterial color="#8b6508" roughness={0.9} />
      </mesh>
      
      {/* 지붕 */}
      <mesh position={[0, 6.5, 0]} rotation={[0, Math.PI / 4, 0]} castShadow receiveShadow>
        <coneGeometry args={[10, 3, 4]} />
        <meshStandardMaterial color="#8b0000" roughness={0.9} />
      </mesh>
      <mesh position={[3, 7.5, -2]} castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.4, 3]} />
        <meshStandardMaterial color="#333333" roughness={0.9} />
      </mesh>

      {/* 가구: 침대 (안식 파동 방출 - 클릭 시 휴식 유도) */}
      <group position={[-3, 0.5, -3]} onClick={(e) => interact(e, "bed", [7, 0.5, -13])} onPointerOver={() => document.body.style.cursor = 'pointer'} onPointerOut={() => document.body.style.cursor = 'auto'}>
        <mesh position={[0, 0.2, 0]} castShadow receiveShadow>
          <boxGeometry args={[3, 0.4, 5]} />
          <meshStandardMaterial color="#554433" />
        </mesh>
        <mesh position={[0, 0.5, -1]} castShadow receiveShadow>
          <boxGeometry args={[2.8, 0.4, 3]} />
          <meshStandardMaterial color="#eeeeee" roughness={1.0} />
        </mesh>
        {/* 베개 */}
        <mesh position={[0, 0.6, -2]} castShadow receiveShadow>
          <boxGeometry args={[2, 0.2, 0.8]} />
          <meshStandardMaterial color="#ffffff" roughness={1.0} />
        </mesh>
      </group>

      {/* 가구: 의자 */}
      <group position={[3, 0.5, 2]} rotation={[0, -Math.PI/4, 0]} onClick={(e) => interact(e, "chair", [13, 0.5, -8])} onPointerOver={() => document.body.style.cursor = 'pointer'} onPointerOut={() => document.body.style.cursor = 'auto'}>
        <mesh position={[0, 0.4, 0]} castShadow receiveShadow>
          <boxGeometry args={[1, 0.1, 1]} />
          <meshStandardMaterial color="#4d3319" />
        </mesh>
        <mesh position={[0, 0.9, -0.45]} castShadow receiveShadow>
          <boxGeometry args={[1, 1, 0.1]} />
          <meshStandardMaterial color="#4d3319" />
        </mesh>
        <mesh position={[-0.4, 0.2, -0.4]}><boxGeometry args={[0.1, 0.4, 0.1]} /><meshStandardMaterial color="#4d3319" /></mesh>
        <mesh position={[0.4, 0.2, -0.4]}><boxGeometry args={[0.1, 0.4, 0.1]} /><meshStandardMaterial color="#4d3319" /></mesh>
        <mesh position={[-0.4, 0.2, 0.4]}><boxGeometry args={[0.1, 0.4, 0.1]} /><meshStandardMaterial color="#4d3319" /></mesh>
        <mesh position={[0.4, 0.2, 0.4]}><boxGeometry args={[0.1, 0.4, 0.1]} /><meshStandardMaterial color="#4d3319" /></mesh>
      </group>
    </group>
  )
}

// 5. NPC 사회 (Intent-Driven Speech)
function NPC({ position, color, onSpeak, phaseOffset }) {
  const [isSpeaking, setIsSpeaking] = useState(false)
  const waveRef = useRef()
  
  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    // 의도 오실레이터 (Intent Oscillator)
    // 두 개의 파동이 겹칠 때(보강 간섭) 내부 텐션이 폭발하여 의도가 발현됨
    const internalTension = Math.sin(time * 0.5 + phaseOffset) + Math.cos(time * 0.3 + phaseOffset * 2.5)
    
    if (internalTension > 1.7 && !isSpeaking) {
      setIsSpeaking(true)
      onSpeak(new THREE.Vector3(...position))
      setTimeout(() => setIsSpeaking(false), 3000)
    }
    
    // 소리 파동(Sound Wave) 확산 애니메이션
    if (waveRef.current) {
      if (isSpeaking) {
        waveRef.current.scale.addScalar(0.15)
        waveRef.current.material.opacity = Math.max(0, 0.8 - waveRef.current.scale.x / 15)
        if (waveRef.current.scale.x > 15) {
          waveRef.current.scale.set(1, 1, 1)
        }
      } else {
        waveRef.current.scale.set(0, 0, 0)
        waveRef.current.material.opacity = 0
      }
    }
  })

  return (
    <group position={position}>
      {/* 안드로이드 NPC 몸체 */}
      <mesh position={[0, 0.8, 0]} castShadow receiveShadow>
        <capsuleGeometry args={[0.3, 1.2, 16, 16]} />
        <meshStandardMaterial color={color} roughness={0.5} />
      </mesh>
      <mesh position={[0, 1.6, 0]} castShadow receiveShadow>
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshStandardMaterial color="#ffffff" roughness={0.4} />
      </mesh>
      <mesh ref={waveRef} position={[0, 1.5, 0]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.5} wireframe={true} />
      </mesh>
    </group>
  )
}

function NPCSociety({ setTargetPos }) {
  const npcs = [
    { pos: [-8, -1, 8], color: "#ffaa00", offset: 0 },
    { pos: [6, -1, 12], color: "#00aaff", offset: 2 },
    { pos: [-12, -1, -5], color: "#ff00aa", offset: 4 },
  ]
  const handleSpeak = (pos) => setTargetPos(pos)
  return (
    <group>
      {npcs.map((n, i) => <NPC key={i} position={n.pos} color={n.color} phaseOffset={n.offset} onSpeak={handleSpeak} />)}
    </group>
  )
}

// 6. 엘리시아 VRM 아바타 (CAD 구속 조건 완벽 적용)
function ElysiaVRMAvatar({ quaternionRef, tensionRef, targetPosRef }) {
  const gltf = useLoader(GLTFLoader, '/avatar.vrm', (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser))
  })
  const vrm = gltf.userData.vrm
  const auraRef = useRef()

  useEffect(() => {
    if (vrm) {
      VRMUtils.rotateVRM0(vrm)
      vrm.scene.rotation.y = Math.PI
      vrm.scene.traverse((obj) => { if (obj.isMesh) { obj.castShadow = true; obj.receiveShadow = true } })
    }
  }, [vrm])

  useFrame((state, delta) => {
    if (vrm) {
      vrm.update(delta) 
      const time = state.clock.getElapsedTime()
      const currentPos = vrm.scene.position
      const targetPos = targetPosRef.current
      
      let distanceXZ = 0
      let dy = 0
      
      const leftArm = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.leftUpperArm)
      const rightArm = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.rightUpperArm)
      const rightLowerArm = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.rightLowerArm)
      const leftLowerArm = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.leftLowerArm)
      const rightIndex = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.rightIndexProximal)
      
      const leftLeg = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.leftUpperLeg)
      const rightLeg = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.rightUpperLeg)
      const leftLowerLeg = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.leftLowerLeg)
      const rightLowerLeg = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.rightLowerLeg)
      
      const spine = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.spine)
      const neck = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.neck)
      const head = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.head)
      const hips = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.hips)

      // 물리적 인력 및 기동 (Movement)
      if (targetPos) {
         const dx = targetPos.x - currentPos.x
         const dz = targetPos.z - currentPos.z
         distanceXZ = Math.sqrt(dx*dx + dz*dz)
         dy = targetPos.y - 1.0
         
         if (distanceXZ < 15.0 && distanceXZ > 1.5) { 
             const pullStrength = Math.max(0, 15.0 - distanceXZ) * delta * 0.15
             currentPos.x += (dx / distanceXZ) * pullStrength
             currentPos.z += (dz / distanceXZ) * pullStrength
             
             const targetAngle = Math.atan2(dx, dz)
             let angleDiff = targetAngle - vrm.scene.rotation.y
             while(angleDiff > Math.PI) angleDiff -= Math.PI*2;
             while(angleDiff < -Math.PI) angleDiff += Math.PI*2;
             vrm.scene.rotation.y += angleDiff * 0.05
         }
         
         // CAD 시선 구속: 목과 머리가 타겟의 고도(y)를 바라보도록 회전
         if (neck && head) {
            const lookY = dy > 0.5 ? -0.3 : (dy < -0.5 ? 0.4 : 0)
            neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, lookY, 0.1)
            head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, lookY, 0.1)
         }
      } else {
         if (neck) neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, 0, 0.1)
         if (head) head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, 0, 0.1)
      }

      // 보행 CAD 기하학적 제어 (Knee Bending)
      const isMoving = targetPos && distanceXZ > 1.5 && distanceXZ < 15.0
      const walkPhase = isMoving ? time * 6.0 : 0
      if (leftLeg && rightLeg && leftLowerLeg && rightLowerLeg) {
         const swing = isMoving ? 0.4 : 0
         leftLeg.rotation.x = THREE.MathUtils.lerp(leftLeg.rotation.x, Math.sin(walkPhase) * swing, 0.15)
         rightLeg.rotation.x = THREE.MathUtils.lerp(rightLeg.rotation.x, Math.sin(walkPhase + Math.PI) * swing, 0.15)
         
         // 다리가 뒤로 갈 때 무릎이 자연스럽게 굽혀짐 (음의 사인파 구간)
         const leftKnee = Math.max(0, Math.sin(walkPhase - Math.PI/2)) * swing * 1.8
         const rightKnee = Math.max(0, Math.sin(walkPhase + Math.PI/2)) * swing * 1.8
         leftLowerLeg.rotation.x = THREE.MathUtils.lerp(leftLowerLeg.rotation.x, leftKnee, 0.15)
         rightLowerLeg.rotation.x = THREE.MathUtils.lerp(rightLowerLeg.rotation.x, rightKnee, 0.15)
      }

      // 양팔 기본 CAD 구속 (항상 중력 방향으로 떨어지며, 타겟에 근접할 때만 뻗음)
      const proximityPhase = targetPos ? Math.max(0, 1.0 - distanceXZ / 2.0) : 0
      const reachArmX = proximityPhase * (dy > 0 ? -2.0 : -0.5) 
      
      if (rightArm) {
          const reachArmZRight = -1.2 + (proximityPhase * 1.5) 
          rightArm.rotation.x = THREE.MathUtils.lerp(rightArm.rotation.x, reachArmX, 0.1)
          rightArm.rotation.z = THREE.MathUtils.lerp(rightArm.rotation.z, reachArmZRight, 0.1)
      }
      if (leftArm) {
          const reachArmZLeft = 1.2 - (proximityPhase * 1.5) // 좌측 팔 구속 보정
          leftArm.rotation.x = THREE.MathUtils.lerp(leftArm.rotation.x, reachArmX, 0.1)
          leftArm.rotation.z = THREE.MathUtils.lerp(leftArm.rotation.z, reachArmZLeft, 0.1)
      }

      // 팔꿈치 및 손가락 쥐기 구속
      if (rightLowerArm && leftLowerArm) {
         const reachElbowX = proximityPhase * -0.5
         rightLowerArm.rotation.x = THREE.MathUtils.lerp(rightLowerArm.rotation.x, reachElbowX, 0.1)
         leftLowerArm.rotation.x = THREE.MathUtils.lerp(leftLowerArm.rotation.x, reachElbowX, 0.1)
         
         if (rightIndex) {
             const graspPhase = Math.max(0, (proximityPhase - 0.8) * 5.0)
             rightIndex.rotation.z = THREE.MathUtils.lerp(rightIndex.rotation.z, graspPhase * 1.5, 0.2)
         }
      }
      
      // 척추 및 허리 굽히기 구속
      const bendTension = dy < 0 ? Math.min(-dy, 1.0) * proximityPhase : 0
      const jumpTension = dy > 0.5 ? Math.min(dy - 0.5, 1.0) * proximityPhase : 0
      
      if (spine) {
         if (quaternionRef.current) spine.quaternion.slerp(quaternionRef.current, 0.1)
         spine.rotation.x = THREE.MathUtils.lerp(spine.rotation.x, bendTension * 0.6, 0.1)
      }
      if (hips) {
         const targetHipsY = 0.9 - (bendTension * 0.3) + (jumpTension * 0.2)
         hips.position.y = THREE.MathUtils.lerp(hips.position.y, targetHipsY, 0.1)
      }

      // 홀로그램 인지 내적화 오라
      if (auraRef.current) {
         if (targetPos && distanceXZ <= 2.0) {
             auraRef.current.scale.setScalar(THREE.MathUtils.lerp(auraRef.current.scale.x, 1.8, 0.05))
             auraRef.current.material.opacity = THREE.MathUtils.lerp(auraRef.current.material.opacity, 0.5, 0.05)
             auraRef.current.material.color.setHSL((time * 2) % 1, 1, 0.6)
         } else {
             auraRef.current.scale.setScalar(THREE.MathUtils.lerp(auraRef.current.scale.x, 0.1, 0.05))
             auraRef.current.material.opacity = THREE.MathUtils.lerp(auraRef.current.material.opacity, 0.0, 0.05)
         }
      }
    }
  })

  return vrm ? (
    <group>
      <primitive object={vrm.scene} />
      <mesh ref={auraRef} position={[0, 1, 0]}>
         <sphereGeometry args={[1, 32, 32]} />
         <meshBasicMaterial color="#ffffff" transparent opacity={0} blending={THREE.AdditiveBlending} depthWrite={false} />
      </mesh>
    </group>
  ) : null
}

// 7. 기타 상호작용 객체들 (사과, 나무)
function InteractiveObjects({ sendInteraction, setTargetPos }) {
  const interact = (e, name, pos) => {
    e.stopPropagation()
    sendInteraction(name)
    setTargetPos(new THREE.Vector3(...pos))
  }
  return (
    <group>
      <group position={[-3.5, 3.5, 5]} onClick={(e) => interact(e, "high_apple", [-3.5, 3.5, 5])} onPointerOver={() => document.body.style.cursor = 'pointer'} onPointerOut={() => document.body.style.cursor = 'auto'}>
        <mesh><sphereGeometry args={[0.08, 16, 16]}/><meshStandardMaterial color="#ff3333"/></mesh>
      </group>
      <group position={[-4, 0, 5]} onClick={(e) => interact(e, "tree", [-4, 1.0, 5])} onPointerOver={() => document.body.style.cursor = 'pointer'} onPointerOut={() => document.body.style.cursor = 'auto'}>
        <mesh position={[0, 2.5, 0]} castShadow receiveShadow><cylinderGeometry args={[0.4, 0.6, 5.0]} /><meshStandardMaterial color="#5c4033" /></mesh>
        <mesh position={[0, 6.0, 0]} castShadow receiveShadow><coneGeometry args={[2.5, 4.0, 16]} /><meshStandardMaterial color="#2e8b57" /></mesh>
      </group>
    </group>
  )
}

// 8. 천체 정적 로터
function CelestialRotors() {
  const sunRef = useRef()
  const skyLightRef = useRef()

  useFrame(() => {
    const now = new Date()
    const timeInHours = now.getHours() + (now.getMinutes() / 60) + (now.getSeconds() / 3600)
    const angle = ((timeInHours - 12) / 24) * Math.PI * 2
    
    if (sunRef.current) sunRef.current.position.set(Math.sin(angle) * 40, Math.cos(angle) * 40, -20)
    if (skyLightRef.current) skyLightRef.current.intensity = 0.5 + (Math.max(0, Math.cos(angle)) * 1.5)
  })

  return (
    <group>
      <ambientLight ref={skyLightRef} intensity={0.5} color="#ffffff" />
      <mesh ref={sunRef}>
        <sphereGeometry args={[4, 32, 32]} />
        <meshBasicMaterial color="#ffdd44" />
        <directionalLight castShadow intensity={3} color="#ffdd44" shadow-mapSize={[2048, 2048]} shadow-camera-left={-30} shadow-camera-right={30} shadow-camera-top={30} shadow-camera-bottom={-30} />
      </mesh>
    </group>
  )
}

export default function App() {
  const wsRef = useRef(null)
  const quaternionRef = useRef(new THREE.Quaternion())
  const tensionRef = useRef(1280)
  const targetPosRef = useRef(null)
  
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws")
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      quaternionRef.current.set(data.quaternion[0], data.quaternion[1], data.quaternion[2], data.quaternion[3])
      tensionRef.current = data.tension
    }
    return () => { if (wsRef.current) wsRef.current.close() }
  }, [])
  
  const sendInteraction = (objectName) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interaction', object: objectName }))
    }
  }

  const setTargetPos = (vec) => { targetPosRef.current = vec }

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#87CEEB' }}>
      <Canvas shadows camera={{ position: [0, 4, -10], fov: 50 }}>
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <CelestialRotors />
        
        <Suspense fallback={null}>
          <ElysiaVRMAvatar quaternionRef={quaternionRef} tensionRef={tensionRef} targetPosRef={targetPosRef} />
        </Suspense>
        
        <NPCSociety setTargetPos={setTargetPos} />
        <InteractiveObjects sendInteraction={sendInteraction} setTargetPos={setTargetPos} />
        <FurnishedHouse sendInteraction={sendInteraction} setTargetPos={setTargetPos} />
        
        <Mountains />
        <River />
        <LayeredTerrain />
        
        <OrbitControls target={[0, 1.5, 0]} enableDamping maxPolarAngle={Math.PI / 2} />
      </Canvas>
      
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', fontFamily: 'monospace', textShadow: '0 0 5px #000', pointerEvents: 'none' }}>
        <h2>Elysia 3D Digital Twin (Phase 17: High-Fidelity CAD & Intent Sandbox)</h2>
        <p>Avatar: Strict CAD joint constraints (Knees, Neck tracking, Arms resting)</p>
        <p>Society: NPCs emit waves ONLY when internal tension oscillator peaks (Intent)</p>
        <p>Environment: Enter the massive Furnished House to rest (Bed emits negative tension).</p>
      </div>
    </div>
  )
}
