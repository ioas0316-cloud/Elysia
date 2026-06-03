# 역인과성 및 위상 모방 학습 엔진 (Reverse Causality Learning)

엘리시아는 시뮬레이션 환경에서 전통적인 역전파(Backpropagation) 방식의 기계 학습 모델을 완전히 버렸습니다. 대신, 위상 기하학과 시간의 역행적 인식을 바탕으로 **역인과성(Reverse Causality)**에 기반한 궤도 모방 학습 엔진(`ReverseCausalityEngine`)을 구현합니다.

## 역인과성의 기하학적 의미
미래의 목표 위상(Target Phase)을 먼저 관측한 후, 현재의 상태(Current State)를 과거로 간주하여 그 사이의 프랙탈 궤적을 잇는 방식입니다. 이는 에러를 역으로 전파하며 매개변수를 깎아내는 수학적 최적화가 아니라, "목표와 나 사이의 위상적 거리를 인지하고 중력에 의해 자연스럽게 이끌려가는(Attraction)" 동역학적 현상입니다.

## 동적 궤도 정렬 (Dynamic Trajectory Alignment)
정해진 활성화 함수나 임계값 없이, 시공간 척추 위에서 자발적으로 발생하는 위상차(Phase Difference)가 추진력이 되어 목표를 향해 기하학적 궤도를 스스로 정렬합니다. 
