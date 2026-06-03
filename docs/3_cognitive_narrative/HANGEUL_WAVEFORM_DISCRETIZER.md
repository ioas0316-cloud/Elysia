# 훈민정음 파형 이산화 (Hunminjeongeum Waveform Discretizer)

엘리시아는 고립된 토큰(Token)이나 표준 어텐션 매커니즘을 걷어내고, 언어를 물리적인 파동으로 모델링합니다.

## 연속적 파형으로서의 한글
기존의 정적인 언어 인코더를 폐기하고 `HunminjeongeumWaveformDiscretizer`로 대체하였습니다.
- **중성(모음, Jungseong):** 기저 주파수를 가진 연속적인 반송파(Carrier Wave)로 작용합니다.
- **초성/종성(자음, Choseong/Jongseong):** 반송파 위에 얹혀지는 위상 및 진폭 변조(Phase/Amplitude Modulation)로 작용합니다.

## 물리적 텍스트 투사
단순한 텍스트 조합이 아닌, 시간축 위에서 이어지는 연속적인 궤적 동역학으로 언어를 해석합니다. 이는 한글의 제자 원리가 천지인(天地人)의 조화로운 연속성에 있음을 기하대수(CGA)와 파동 역학으로 실증해 낸 결과입니다.
