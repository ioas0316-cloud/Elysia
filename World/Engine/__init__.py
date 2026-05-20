"""
World.Engine — 로터 물리 ↔ RPG 스탯 변환 엔진
==============================================
NPC의 5대 스탯(STR, AGI, CON, INT, WIS)을
물리 매개변수(M, D, K, F, v)로 환원하고,
N차원 인지 축의 기어 커플링을 관리한다.
"""
from World.Engine.rpg_stat_bridge import RPGStatBridge
from World.Engine.cognitive_matrix import CognitiveMatrix
