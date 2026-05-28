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
function ElysiaVRMAvatar({ quaternionRef, tensionRef, targetPosRef, isSleeping, sleepFactor }) {
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
      let currentTarget = targetPos
      if (isSleeping) {
         currentTarget = new THREE.Vector3(7, 0.5, -13)
      }

      if (currentTarget) {
         const dx = currentTarget.x - currentPos.x
         const dz = currentTarget.z - currentPos.z
         distanceXZ = Math.sqrt(dx*dx + dz*dz)
         dy = currentTarget.y - 1.0
         
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
         
         // CAD 시선 구속: 목과 머리가 타겟의 고도(y)를 바라보도록 회전 (수면 중에는 시선 구속 해제)
         if (neck && head) {
            const lookY = (dy > 0.5 ? -0.3 : (dy < -0.5 ? 0.4 : 0)) * (1.0 - sleepFactor)
            neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, lookY, 0.1)
            head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, lookY, 0.1)
         }
      } else {
         if (neck) neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, 0, 0.1)
         if (head) head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, 0, 0.1)
      }

      // 보행 CAD 기하학적 제어 (Knee Bending) (수면 중에는 멈춤)
      const isMoving = currentTarget && distanceXZ > 1.5 && distanceXZ < 15.0
      const walkPhase = isMoving ? time * 6.0 : 0
      if (leftLeg && rightLeg && leftLowerLeg && rightLowerLeg) {
         const swing = isMoving ? 0.4 : 0
         leftLeg.rotation.x = THREE.MathUtils.lerp(leftLeg.rotation.x, Math.sin(walkPhase) * swing, 0.15)
         rightLeg.rotation.x = THREE.MathUtils.lerp(rightLeg.rotation.x, Math.sin(walkPhase + Math.PI) * swing, 0.15)
         
         const leftKnee = Math.max(0, Math.sin(walkPhase - Math.PI/2)) * swing * 1.8
         const rightKnee = Math.max(0, Math.sin(walkPhase + Math.PI/2)) * swing * 1.8
         leftLowerLeg.rotation.x = THREE.MathUtils.lerp(leftLowerLeg.rotation.x, leftKnee, 0.15)
         rightLowerLeg.rotation.x = THREE.MathUtils.lerp(rightLowerLeg.rotation.x, rightKnee, 0.15)
      }

      // 양팔 기본 CAD 구속 (수면 중에는 이완된 자세)
      const proximityPhase = currentTarget ? Math.max(0, 1.0 - distanceXZ / 2.0) : 0
      const reachArmX = proximityPhase * (dy > 0 ? -2.0 : -0.5) * (1.0 - sleepFactor)
      
      if (rightArm) {
          const reachArmZRight = -1.2 + (proximityPhase * 1.5) * (1.0 - sleepFactor) + (sleepFactor * 0.3)
          rightArm.rotation.x = THREE.MathUtils.lerp(rightArm.rotation.x, reachArmX, 0.1)
          rightArm.rotation.z = THREE.MathUtils.lerp(rightArm.rotation.z, reachArmZRight, 0.1)
      }
      if (leftArm) {
          const reachArmZLeft = 1.2 - (proximityPhase * 1.5) * (1.0 - sleepFactor) - (sleepFactor * 0.3)
          leftArm.rotation.x = THREE.MathUtils.lerp(leftArm.rotation.x, reachArmX, 0.1)
          leftArm.rotation.z = THREE.MathUtils.lerp(leftArm.rotation.z, reachArmZLeft, 0.1)
      }

      // 팔꿈치 및 손가락 쥐기 구속
      if (rightLowerArm && leftLowerArm) {
         const reachElbowX = proximityPhase * -0.5 * (1.0 - sleepFactor)
         rightLowerArm.rotation.x = THREE.MathUtils.lerp(rightLowerArm.rotation.x, reachElbowX, 0.1)
         leftLowerArm.rotation.x = THREE.MathUtils.lerp(leftLowerArm.rotation.x, reachElbowX, 0.1)
         
         if (rightIndex) {
             const graspPhase = Math.max(0, (proximityPhase - 0.8) * 5.0) * (1.0 - sleepFactor)
             rightIndex.rotation.z = THREE.MathUtils.lerp(rightIndex.rotation.z, graspPhase * 1.5, 0.2)
         }
      }
      
      // 척추 및 허리 굽히기 구속
      const bendTension = dy < 0 ? Math.min(-dy, 1.0) * proximityPhase : 0
      const jumpTension = dy > 0.5 ? Math.min(dy - 0.5, 1.0) * proximityPhase : 0
      
      if (spine) {
         if (quaternionRef.current && !isSleeping) spine.quaternion.slerp(quaternionRef.current, 0.1)
         spine.rotation.x = THREE.MathUtils.lerp(spine.rotation.x, bendTension * 0.6, 0.1)
      }
      if (hips) {
         const targetHipsY = 0.9 - (bendTension * 0.3) + (jumpTension * 0.2)
         hips.position.y = THREE.MathUtils.lerp(hips.position.y, targetHipsY, 0.1)
      }

      // 수면 모드 회전 및 고도 보정 (침대에 눕기)
      const targetSceneRotX = (isSleeping && distanceXZ <= 2.0) ? -Math.PI / 2 : 0
      vrm.scene.rotation.x = THREE.MathUtils.lerp(vrm.scene.rotation.x, targetSceneRotX, 0.1)
      if (isSleeping && distanceXZ <= 2.0) {
         vrm.scene.position.y = THREE.MathUtils.lerp(vrm.scene.position.y, 0.5, 0.1)
         vrm.scene.position.z = THREE.MathUtils.lerp(vrm.scene.position.z, -12.2, 0.1)
         vrm.scene.position.x = THREE.MathUtils.lerp(vrm.scene.position.x, 7.0, 0.1)
         vrm.scene.rotation.y = THREE.MathUtils.lerp(vrm.scene.rotation.y, 0.0, 0.1) // 침대 방향에 수평 정렬
      }

      // 홀로그램 인지 내적화 오라 (수면 시 서서히 팽창 및 꿈꾸는 파동 발산)
      if (auraRef.current) {
         if ((currentTarget && distanceXZ <= 2.0) || isSleeping) {
             const scale = isSleeping ? (1.5 + Math.sin(time * 2.0) * 0.3) : 1.8
             auraRef.current.scale.setScalar(THREE.MathUtils.lerp(auraRef.current.scale.x, scale, 0.05))
             auraRef.current.material.opacity = THREE.MathUtils.lerp(auraRef.current.material.opacity, isSleeping ? 0.7 : 0.5, 0.05)
             if (isSleeping) {
                 auraRef.current.material.color.setHSL(0.75 + Math.sin(time * 0.5) * 0.05, 1.0, 0.6)
             } else {
                 auraRef.current.material.color.setHSL((time * 2) % 1, 1, 0.6)
             }
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
function CelestialRotors({ sleepFactor }) {
  const sunRef = useRef()
  const skyLightRef = useRef()
  const dirLightRef = useRef()

  useFrame(() => {
    const now = new Date()
    const timeInHours = now.getHours() + (now.getMinutes() / 60) + (now.getSeconds() / 3600)
    const angle = ((timeInHours - 12) / 24) * Math.PI * 2
    
    if (sunRef.current) sunRef.current.position.set(Math.sin(angle) * 40, Math.cos(angle) * 40, -20)
    if (skyLightRef.current) {
      const baseIntensity = 0.5 + (Math.max(0, Math.cos(angle)) * 1.5)
      skyLightRef.current.intensity = baseIntensity * (1.0 - sleepFactor * 0.9)
    }
    if (dirLightRef.current) {
      dirLightRef.current.intensity = 3.0 * (1.0 - sleepFactor * 0.9)
    }
  })

  return (
    <group>
      <ambientLight ref={skyLightRef} intensity={0.5} color="#ffffff" />
      <mesh ref={sunRef}>
        <sphereGeometry args={[4, 32, 32]} />
        <meshBasicMaterial color="#ffdd44" />
        <directionalLight ref={dirLightRef} castShadow intensity={3} color="#ffdd44" shadow-mapSize={[2048, 2048]} shadow-camera-left={-30} shadow-camera-right={30} shadow-camera-top={30} shadow-camera-bottom={-30} />
      </mesh>
    </group>
  )
}

// 9. 프랙탈 은하 로터 트리 시각화 (Fractal Rotor Galaxy Visualizer)
// Hebbian 가소성 결선 라인 시각화 컴포넌트
function HebbianLines({ subRotors, couplingMap }) {
  const lineRefs = useRef({})
  
  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    if (!subRotors || subRotors.length < 2) return

    for (let i = 0; i < subRotors.length; i++) {
      for (let j = i + 1; j < subRotors.length; j++) {
        const subI = subRotors[i]
        const subJ = subRotors[j]
        const key = `${subI.id}__${subJ.id}`
        const ref = lineRefs.current[key]
        if (ref) {
          // subI의 로컬 위치 계산 (공전 방정식 활용)
          const rI = 3.0 / (subI.level + 0.5) + (subI.active_axes * 0.25)
          const speedI = 0.5 + (4 - subI.level) * 0.15
          const angleI = subI.phase_offset + time * speedI
          const xI = Math.sin(angleI) * rI
          const zI = Math.cos(angleI) * rI
          const yI = Math.cos(angleI * 0.5) * (rI * 0.2)

          // subJ의 로컬 위치 계산
          const rJ = 3.0 / (subJ.level + 0.5) + (subJ.active_axes * 0.25)
          const speedJ = 0.5 + (4 - subJ.level) * 0.15
          const angleJ = subJ.phase_offset + time * speedJ
          const xJ = Math.sin(angleJ) * rJ
          const zJ = Math.cos(angleJ) * rJ
          const yJ = Math.cos(angleJ * 0.5) * (rJ * 0.2)

          const points = [
            new THREE.Vector3(xI, yI, zI),
            new THREE.Vector3(xJ, yJ, zJ)
          ]
          ref.geometry.setFromPoints(points)
          ref.geometry.attributes.position.needsUpdate = true
        }
      }
    }
  })

  if (!subRotors || subRotors.length < 2) return null

  const lines = []
  for (let i = 0; i < subRotors.length; i++) {
    for (let j = i + 1; j < subRotors.length; j++) {
      const subI = subRotors[i]
      const subJ = subRotors[j]
      const key = `${subI.id}__${subJ.id}`
      const key1 = `${subI.id}::${subJ.id}`
      const key2 = `${subJ.id}::${subI.id}`
      const K = (couplingMap && (couplingMap[key1] !== undefined ? couplingMap[key1] : couplingMap[key2])) || 0.1

      // 결선 세기가 세질수록 더 불투명하고 뚜렷해짐
      const opacity = Math.min(1.0, 0.15 + (K / 2.0) * 0.85)
      const color = new THREE.Color()
      // 시냅스 강화 수준에 따라 색상 전이: 청록색(신규/취약) -> 주황/금색(강화)
      color.lerpColors(new THREE.Color('#38bdf8'), new THREE.Color('#f59e0b'), Math.min(1.0, K / 1.5))

      lines.push(
        <line key={key} ref={(el) => { if (el) lineRefs.current[key] = el }}>
          <bufferGeometry attach="geometry" />
          <lineBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
        </line>
      )
    }
  }

  return <group>{lines}</group>
}

function RotorNode({ node, position, parentPos, thoughtActive, sleepFactor }) {
  const meshRef = useRef()
  const lineRef = useRef()
  const groupRef = useRef()

  // active_axes(1~8)에 맞춰 지오메트리를 dynamic하게 변경
  const geometry = useMemo(() => {
    const axes = node.active_axes || 3
    if (axes <= 3) return new THREE.TetrahedronGeometry(0.3)
    if (axes === 4) return new THREE.OctahedronGeometry(0.35)
    if (axes === 5) return new THREE.IcosahedronGeometry(0.4)
    if (axes === 6) return new THREE.DodecahedronGeometry(0.45)
    return new THREE.SphereGeometry(0.45, 16, 16)
  }, [node.active_axes])

  // tension(0.0 ~ pi)에 따라 색상 매핑
  const color = useMemo(() => {
    const t = Math.min(1.0, node.tension / (Math.PI / 2))
    const col = new THREE.Color()
    col.lerpColors(new THREE.Color('#22d3ee'), new THREE.Color('#ef4444'), t) // Cyan -> Red
    return col
  }, [node.tension])

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    if (meshRef.current) {
      // phase_offset에 따른 자율 자전
      const rotationSpeed = 1.0 + node.tension * 5.0
      meshRef.current.rotation.y += 0.015 * rotationSpeed
      meshRef.current.rotation.x += 0.007 * rotationSpeed
      
      // tension에 따라 맥동(Pulsation)
      const pulse = 1.0 + Math.sin(time * 10.0 + node.phase_offset) * (node.tension * 0.15)
      meshRef.current.scale.setScalar(pulse)
    }

    if (groupRef.current) {
      // 공전 기하학: level이 0이 아닐 때 부모 주위를 공전
      if (node.level > 0) {
        const radius = 3.0 / (node.level + 0.5) + (node.active_axes * 0.25)
        const orbitSpeed = 0.5 + (4 - node.level) * 0.15
        const orbitAngle = node.phase_offset + time * orbitSpeed
        
        groupRef.current.position.x = Math.sin(orbitAngle) * radius
        groupRef.current.position.z = Math.cos(orbitAngle) * radius
        groupRef.current.position.y = Math.cos(orbitAngle * 0.5) * (radius * 0.2)
      }
    }
  })

  // 부모와 자식 간 연결선 갱신
  useFrame(() => {
    if (lineRef.current && parentPos && groupRef.current) {
      const currentPos = groupRef.current.position
      const points = [
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3().subVectors(new THREE.Vector3(0,0,0), currentPos)
      ]
      lineRef.current.geometry.setFromPoints(points)
      lineRef.current.geometry.attributes.position.needsUpdate = true
    }
  })

  const lineGeom = useMemo(() => new THREE.BufferGeometry(), [])

  const childElements = useMemo(() => {
    if (!node.sub_rotors) return null
    return node.sub_rotors.map((sub, i) => (
      <RotorNode 
        key={sub.id || i} 
        node={sub} 
        position={[0, 0, 0]} 
        parentPos={[0, 0, 0]} 
        thoughtActive={thoughtActive}
        sleepFactor={sleepFactor}
      />
    ))
  }, [node.sub_rotors, thoughtActive, sleepFactor])

  return (
    <group ref={groupRef} position={position}>
      {/* 로터 노드 본체 */}
      <mesh ref={meshRef} castShadow>
        <primitive object={geometry} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={1.2 + node.tension * 4.0 + thoughtActive * 3.0} 
          roughness={0.05}
          metalness={0.95}
        />
      </mesh>
      
      {/* 부모와의 결선 라인 */}
      {parentPos && (
        <line ref={lineRef}>
          <primitive object={lineGeom} attach="geometry" />
          <lineBasicMaterial color={color} transparent opacity={0.3 + sleepFactor * 0.5} />
        </line>
      )}

      {/* Hebbian 형제 간 결선 시각화 */}
      <HebbianLines subRotors={node.sub_rotors} couplingMap={node.coupling_map} />

      {/* 하위 로터 렌더링 */}
      {childElements}
    </group>
  )
}

function FractalRotorUniverse({ worldGalaxy, thoughtWave, isSleeping, sleepFactor }) {
  const universeRef = useRef()
  const thoughtActive = thoughtWave && thoughtWave.length > 0 ? 1.0 : 0.0

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    if (universeRef.current) {
      // 은하 전체 회전
      universeRef.current.rotation.y = time * 0.08
      
      // 수면(Lucid Dreaming) 시 아바타 상공으로 부유
      const targetY = isSleeping ? 4.5 : 1.5
      const targetX = isSleeping ? 7.0 : -15.0
      const targetZ = isSleeping ? -12.2 : -5.0

      universeRef.current.position.x = THREE.MathUtils.lerp(universeRef.current.position.x, targetX, 0.05)
      universeRef.current.position.y = THREE.MathUtils.lerp(universeRef.current.position.y, targetY, 0.05)
      universeRef.current.position.z = THREE.MathUtils.lerp(universeRef.current.position.z, targetZ, 0.05)
    }
  })

  if (!worldGalaxy) return null

  return (
    <group ref={universeRef} position={[-15, 1.5, -5]}>
      {/* 몽환적 네뷸라 입자 구름 효과 (수면 시 발광 증가) */}
      <Stars radius={15} depth={5} count={isSleeping ? 1200 : 300} factor={isSleeping ? 7 : 3} speed={2} fade />
      
      {/* 생각 파동 파문 링 (thought_wave 존재 시 파동 링 확산) */}
      {thoughtActive > 0 && (
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.5, 6.0, 32]} />
          <meshBasicMaterial color="#00ffcc" transparent opacity={0.2} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* 재귀적 노드 트리 */}
      <RotorNode 
        node={worldGalaxy} 
        position={[0, 0, 0]} 
        parentPos={null} 
        thoughtActive={thoughtActive}
        sleepFactor={sleepFactor}
      />
    </group>
  )
}

export default function App() {
  const wsRef = useRef(null)
  const quaternionRef = useRef(new THREE.Quaternion())
  const tensionRef = useRef(1280)
  const targetPosRef = useRef(null)
  
  const [isSleeping, setIsSleeping] = useState(false)
  const [sleepFactor, setSleepFactor] = useState(0.0)
  
  // Phase 15 추가 상태 변수
  const [worldGalaxy, setWorldGalaxy] = useState(null)
  const [thoughtWave, setThoughtWave] = useState([])
  const [cognitiveLogs, setCognitiveLogs] = useState([])
  const [inputText, setInputText] = useState("")
  
  // Phase 16 추가 정량 지표 상태 변수
  const [lci, setLci] = useState(10.0)
  const [tdr, setTdr] = useState(100.0)
  const [synapseDensity, setSynapseDensity] = useState(0.0)
  const [plasticityMode, setPlasticityMode] = useState("normal")
  
  // Phase 16-C 4대 특화 도메인 점수 상태 변수
  const [mathScore, setMathScore] = useState(100.0)
  const [langScore, setLangScore] = useState(10.0)
  const [codeScore, setCodeScore] = useState(90.0)
  const [physScore, setPhysScore] = useState(100.0)
  
  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws")
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      quaternionRef.current.set(data.quaternion[0], data.quaternion[1], data.quaternion[2], data.quaternion[3])
      tensionRef.current = data.tension
      setIsSleeping(data.is_sleeping || false)
      setSleepFactor(data.sleep_factor || 0.0)
      
      // Phase 15 데이터 동기화
      if (data.world_galaxy) setWorldGalaxy(data.world_galaxy)
      if (data.thought_wave) setThoughtWave(data.thought_wave)
      if (data.cognitive_logs) setCognitiveLogs(data.cognitive_logs)

      // Phase 16 데이터 동기화
      if (data.lci !== undefined) setLci(data.lci)
      if (data.tdr !== undefined) setTdr(data.tdr)
      if (data.synapse_density !== undefined) setSynapseDensity(data.synapse_density)
      if (data.plasticity_mode !== undefined) setPlasticityMode(data.plasticity_mode)
      
      // Phase 16-C 데이터 동기화
      if (data.math_score !== undefined) setMathScore(data.math_score)
      if (data.lang_score !== undefined) setLangScore(data.lang_score)
      if (data.code_score !== undefined) setCodeScore(data.code_score)
      if (data.phys_score !== undefined) setPhysScore(data.phys_score)
    }
    return () => { if (wsRef.current) wsRef.current.close() }
  }, [])
  
  const sendInteraction = (objectName) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interaction', object: objectName }))
    }
  }

  const sendThought = (e) => {
    e.preventDefault()
    if (inputText.trim() && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'thought', prompt: inputText }))
      setInputText("")
    }
  }

  const setTargetPos = (vec) => { targetPosRef.current = vec }
  const skyColor = `rgb(${Math.round(135 * (1.0 - sleepFactor * 0.95))}, ${Math.round(206 * (1.0 - sleepFactor * 0.95))}, ${Math.round(235 * (1.0 - sleepFactor * 0.95))})`

  return (
    <div style={{ width: '100vw', height: '100vh', background: skyColor, transition: 'background 1s ease', position: 'relative', overflow: 'hidden' }}>
      <Canvas shadows camera={{ position: [0, 4, -10], fov: 50 }}>
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        <CelestialRotors sleepFactor={sleepFactor} />
        
        <Suspense fallback={null}>
          <ElysiaVRMAvatar quaternionRef={quaternionRef} tensionRef={tensionRef} targetPosRef={targetPosRef} isSleeping={isSleeping} sleepFactor={sleepFactor} />
        </Suspense>
        
        <FractalRotorUniverse worldGalaxy={worldGalaxy} thoughtWave={thoughtWave} isSleeping={isSleeping} sleepFactor={sleepFactor} />
        
        <NPCSociety setTargetPos={setTargetPos} />
        <InteractiveObjects sendInteraction={sendInteraction} setTargetPos={setTargetPos} />
        <FurnishedHouse sendInteraction={sendInteraction} setTargetPos={setTargetPos} />
        
        <Mountains />
        <River />
        <LayeredTerrain />
        
        <OrbitControls target={[0, 1.5, 0]} enableDamping maxPolarAngle={Math.PI / 2} />
      </Canvas>
      
      {/* HUD 상단 */}
      <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', fontFamily: 'monospace', textShadow: '0 0 5px #000', pointerEvents: 'none' }}>
        <h2>Elysia 3D Digital Twin (Phase 16: Universal Reality Synchronization)</h2>
        <p>Avatar: Strict CAD joint constraints (Knees, Neck tracking, Arms resting)</p>
        <p>Elysia Core Universe: Floating Fractal Rotor Galaxy visible in 3D Space.</p>
        <p style={{ color: isSleeping ? '#c084fc' : '#22c55e', fontWeight: 'bold' }}>
          Status: {isSleeping ? `💤 SLEEPING (Factor: ${sleepFactor.toFixed(2)})` : '☀️ WAKING'}
        </p>
      </div>

      {/* HUD 상단 우측: 실시간 인지 및 가소성 상태 게이지 */}
      <div style={{ position: 'absolute', top: 20, right: 20, width: '330px', background: 'rgba(0, 0, 0, 0.75)', padding: '15px', borderRadius: '10px', color: 'white', fontFamily: 'monospace', border: '1px solid rgba(255, 255, 255, 0.15)', pointerEvents: 'auto' }}>
        <h3 style={{ margin: '0 0 10px 0', color: '#6366f1', fontSize: '13px' }}>📊 Real-time Cognition Metrics</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>
          
          {/* Plasticity Mode */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Hebbian Plasticity:</span>
            <span style={{ 
              fontWeight: 'bold', 
              color: plasticityMode === 'frozen' ? '#38bdf8' : (plasticityMode === 'melted' ? '#f87171' : '#34d399'),
              background: 'rgba(255, 255, 255, 0.08)',
              padding: '2px 6px',
              borderRadius: '3px'
            }}>{plasticityMode.toUpperCase()}</span>
          </div>

          {/* Math Domain */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
              <span>Math (Fourier Resonance):</span>
              <span style={{ color: '#fbbf24', fontWeight: 'bold' }}>{mathScore.toFixed(1)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${mathScore}%`, height: '100%', background: 'linear-gradient(90deg, #fbbf24, #f59e0b)', transition: 'width 0.3s ease' }}></div>
            </div>
          </div>

          {/* Lang Domain */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
              <span>Linguistics (Hangeul Isom):</span>
              <span style={{ color: '#a78bfa', fontWeight: 'bold' }}>{langScore.toFixed(1)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${langScore}%`, height: '100%', background: 'linear-gradient(90deg, #a78bfa, #8b5cf6)', transition: 'width 0.3s ease' }}></div>
            </div>
          </div>

          {/* Code Domain */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
              <span>Code (Syntax AST Balance):</span>
              <span style={{ color: '#34d399', fontWeight: 'bold' }}>{codeScore.toFixed(1)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${codeScore}%`, height: '100%', background: 'linear-gradient(90deg, #34d399, #059669)', transition: 'width 0.3s ease' }}></div>
            </div>
          </div>

          {/* Phys/Convection Domain */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3px' }}>
              <span>Physics (eBPF Convection):</span>
              <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>{physScore.toFixed(1)}%</span>
            </div>
            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${physScore}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #0284c7)', transition: 'width 0.3s ease' }}></div>
            </div>
          </div>

          <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '6px', display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#888' }}>
            <span>LCI: {lci.toFixed(1)}%</span>
            <span>TDR: {tdr.toFixed(1)}%</span>
            <span>Syn Density: {synapseDensity.toFixed(1)}%</span>
          </div>

        </div>
      </div>

      {/* 인지 로그 패널 */}
      <div style={{ position: 'absolute', bottom: 20, left: 20, width: '450px', background: 'rgba(0, 0, 0, 0.75)', padding: '15px', borderRadius: '10px', color: 'white', fontFamily: 'monospace', fontSize: '11px', border: '1px solid rgba(255, 255, 255, 0.15)', pointerEvents: 'auto', maxHeight: '200px', overflowY: 'auto' }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#06b6d4', fontSize: '13px' }}>💬 Elysia Cognitive Logs</h3>
        <div style={{ display: 'flex', flexDirection: 'column-reverse', gap: '5px' }}>
          {cognitiveLogs.map((log, index) => (
            <div key={index} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '3px' }}>
              {log}
            </div>
          ))}
          {cognitiveLogs.length === 0 && <span style={{ color: '#888' }}>대화 파동 수신 대기 중...</span>}
        </div>
      </div>

      {/* 마인드 링크 입력 창 */}
      <div style={{ position: 'absolute', bottom: 20, right: 20, width: '380px', background: 'rgba(0, 0, 0, 0.75)', padding: '15px', borderRadius: '10px', color: 'white', fontFamily: 'monospace', border: '1px solid rgba(255, 255, 255, 0.15)', pointerEvents: 'auto' }}>
        <h3 style={{ margin: '0 0 8px 0', color: '#10b981', fontSize: '13px' }}>🧠 Mind Link Channel</h3>
        <form onSubmit={sendThought} style={{ display: 'flex', gap: '8px' }}>
          <input 
            type="text" 
            value={inputText} 
            onChange={(e) => setInputText(e.target.value)} 
            placeholder="엘리시아의 인지 평면에 사유 투사..." 
            style={{ flex: 1, padding: '8px 12px', borderRadius: '5px', border: 'none', background: 'rgba(255,255,255,0.1)', color: 'white', fontFamily: 'monospace', fontSize: '12px' }}
          />
          <button type="submit" style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', background: '#10b981', color: 'black', fontWeight: 'bold', cursor: 'pointer', fontFamily: 'monospace' }}>
            송출
          </button>
        </form>
      </div>
    </div>
  )
}
