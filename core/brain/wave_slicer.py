from core.utils.math_utils import Quaternion

class WaveSlicer:
    """
    [연속 파장 재단기 (Wave Slicer)]
    
    연속적으로 흘러들어오는 파동(데이터)을 관측하다가, 
    위상각(텐션)의 미분값이 급격히 튀는 변곡점을 스스로 감지하여
    파동을 '테트리스 블록' 단위로 잘라냅니다(Discretization).
    """
    def __init__(self):
        self.previous_wave = None
        self.current_block_buffer = []
        
    def stream_wave(self, current_wave: Quaternion, wave_name: str, logs: list) -> list:
        """
        파동이 1프레임 들어올 때마다 호출. 
        만약 잘라낸 블록(Tetris Block)이 완성되면 반환함.
        """
        cut_blocks = []
        
        # 1. 초기 상태
        if self.previous_wave is None:
            self.previous_wave = current_wave
            self.current_block_buffer.append(wave_name)
            return cut_blocks
            
        # 2. 위상 변화율(텐션 미분값) 측정
        dot_product = abs(current_wave.dot(self.previous_wave))
        dissonance = 1.0 - dot_product
        
        # 3. 변곡점(슬라이싱) 감지
        # 텐션이 0.1 이상 튀면(위상이 확 꺾이면), 새로운 조각의 시작으로 간주함
        if dissonance > 0.1:
            logs.append(f"   [Cut] [Wave Slicer] 파장의 급격한 꺾임(텐션 {dissonance:.2f}) 감지. 블록을 절단합니다.")
            
            # 버퍼에 있던 파장들을 하나의 테트리스 블록으로 뭉침
            completed_block = "".join(self.current_block_buffer)
            cut_blocks.append(completed_block)
            
            # 버퍼 초기화 및 새 조각 시작
            self.current_block_buffer = [wave_name]
        else:
            # 부드럽게 이어지는 연속 파장이면 계속 버퍼에 담음
            self.current_block_buffer.append(wave_name)
            
        self.previous_wave = current_wave
        return cut_blocks

    def flush_buffer(self) -> str:
        if self.current_block_buffer:
            return "".join(self.current_block_buffer)
        return None
