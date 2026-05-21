import asyncio
import random
import time
import sys

async def data_stream(stream_queue):
    """
    윈도우/하드웨어의 자연스러운 흐름 (물).
    엘리시아의 상태와 무관하게 끊임없이 이진 데이터를 생성하며 흐른다.
    """
    stream_id = 0
    while True:
        # Generate 9 bits for a 3x3 grid
        binary_stream = [random.choice([0, 1]) for _ in range(9)]
        stream_id += 1

        # Log the flow
        sys.stdout.write(f"[Data Stream] Flow #{stream_id}: {binary_stream} flowing...\n")
        sys.stdout.flush()

        # Pass to the turbine queue (non-blocking put)
        # We use a maxsize of 0 (infinite) or handle Full exceptions
        # but asyncio.Queue without maxsize is safe.
        await stream_queue.put((stream_id, binary_stream))

        # Simulate continuous rapid flow
        await asyncio.sleep(0.5)

class ElysiaTurbine:
    """
    엘리시아 터빈 (물레방아).
    흐르는 데이터 스트림을 센싱하고 복소수 감쇠 진동 모델을 통해 궤적을 그린다.
    """
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        # 복소수 위상 격자 초기화 (실수축 0, 완전 평형 상태)
        self.lattice = [0j for _ in range(grid_size * grid_size)]

    def map_to_complex_phase(self, binary_stream):
        """
        이진 스트림(0, 1)을 복소평면의 허수축 위상 (+i, -i)으로 매핑.
        1 -> +i (e^(i*π/2))
        0 -> -i (e^(-i*π/2))
        """
        complex_phases = []
        for bit in binary_stream:
            if bit == 1:
                complex_phases.append(1j) # +i
            else:
                complex_phases.append(-1j) # -i
        return complex_phases

    def apply_damped_oscillation(self, complex_phases):
        """
        새로 들어온 파동과 기존 격자의 파동을 간섭시키고, 감쇠(Damping)를 적용.
        """
        for i in range(len(self.lattice)):
            # 간섭 (Interference)
            self.lattice[i] += complex_phases[i]
            # 감쇠 (Damping): 에너지가 0으로 수렴하게 만듦. 마찰계수 역할 (예: 0.8)
            self.lattice[i] *= 0.8

    def display_lattice(self, stream_id):
        """
        3x3 격자의 파동 상태와 총 에너지를 텍스트로 시각화.
        """
        total_energy = sum(abs(z) for z in self.lattice)

        sys.stdout.write(f"\n[Elysia Turbine] Sensing Flow #{stream_id} | Total Energy: {total_energy:.4f}\n")
        sys.stdout.write("Waveform Lattice (Real, Imaginary):\n")

        for row in range(self.grid_size):
            row_str = " | ".join(
                f"{self.lattice[row * self.grid_size + col].real:5.2f} + {self.lattice[row * self.grid_size + col].imag:5.2f}i"
                for col in range(self.grid_size)
            )
            sys.stdout.write(f"  [{row_str}]\n")
        sys.stdout.write("-" * 60 + "\n")
        sys.stdout.flush()

async def elysia_turbine(stream_queue):
    turbine = ElysiaTurbine()

    while True:
        # 데이터 스트림 센싱 (논블로킹 대기)
        stream_id, binary_stream = await stream_queue.get()

        # 의도적 에러 시뮬레이션: 터빈이 특정 시점에 정지/에러 발생
        if stream_id == 4:
            sys.stdout.write("\n[Elysia Turbine] CRITICAL FAULT! Turbine is stalled/crashing!\n")
            sys.stdout.write("[Elysia Turbine] Simulating a 3-second block/hang to prove non-blocking flow...\n\n")
            sys.stdout.flush()
            # 터빈이 멈추더라도 data_stream은 계속 흐르는지 증명하기 위해 await asyncio.sleep 사용 (이벤트 루프 제어권 반환)
            await asyncio.sleep(3)
            sys.stdout.write("\n[Elysia Turbine] Recovered from fault.\n")
            sys.stdout.flush()

        # 1. 맵핑
        complex_phases = turbine.map_to_complex_phase(binary_stream)

        # 2. 감쇠 진동 적용
        turbine.apply_damped_oscillation(complex_phases)

        # 3. 궤적 로그 출력
        turbine.display_lattice(stream_id)

        # 큐 작업 완료 표시
        stream_queue.task_done()

async def main():
    stream_queue = asyncio.Queue()

    # Start the data stream (Water) in the background
    water_task = asyncio.create_task(data_stream(stream_queue))

    # Start the turbine (Elysia)
    turbine_task = asyncio.create_task(elysia_turbine(stream_queue))

    # Run for 10 seconds to verify the flow, sensing, and fault tolerance
    await asyncio.sleep(10)

    # Cancel the tasks
    water_task.cancel()
    turbine_task.cancel()

    sys.stdout.write("\n[System] Simulation complete. As demonstrated, Data Stream flows continuously regardless of Turbine's state.\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
