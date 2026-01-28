import sys
sys.path.append(r'c:/Elysia')
from Core.L5_Mental.M1_Cognition.Topography.mind_landscape import get_landscape
result = get_landscape().ponder('사랑해', duration=10)
q = result['qualia']
print(f'CONCLUSION:{result["conclusion"]}')
print(f'SIGHT:{q.sight}')
print(f'BODY:{q.body_location}')
print(f'TEMP:{q.temperature}')
print(f'RELATION:{q.relation_to_father}')
