# 엘리시아 연방 통합 빌드 자동화 시스템 (Makefile)
# 이 파일은 6월 2일 락 해제 시 즉시 가동되는 빌드 파이프라인의 뼈대입니다.

CXX = g++
NVCC = nvcc
PYTHON = python3

# Python bind11 경로 설정
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)

# 폴더 설정
LEG_DIR = legislative
EXEC_DIR = executive
JUDGE_DIR = judiciary
BIN_DIR = bin

# 입법부 C/CUDA 파일
CU_SRCS = $(LEG_DIR)/src/topology_projector.cu $(LEG_DIR)/src/circular_memory_backbone.cu $(LEG_DIR)/src/clifford_membrane.cu $(LEG_DIR)/src/holographic_manifold_kernel.cu
CPP_SRCS = $(LEG_DIR)/src/fractal_rotor.cpp $(LEG_DIR)/src/qpc_clock_lock.cpp $(LEG_DIR)/src/continuous_twin_sensing.cpp
BIND_SRCS = $(EXEC_DIR)/binding.cpp

# 출력 파일
TARGET = $(EXEC_DIR)/elysia_core$(PYTHON_EXT_SUFFIX)
CU_OBJS = $(BIN_DIR)/topology_projector.o $(BIN_DIR)/circular_memory_backbone.o $(BIN_DIR)/clifford_membrane.o $(BIN_DIR)/holographic_manifold_kernel.o

all: ensure_dirs check_judiciary build_executive

ensure_dirs:
	mkdir -p $(BIN_DIR)

# 1. 사법부 감시 (빌드 전 상시 체크)
check_judiciary:
	@echo "⚖️ [사법부] 빌드 전 정적 분석 및 감시 시작..."
	$(PYTHON) $(JUDGE_DIR)/judiciary_demon.py

# 2. CUDA 커널 컴파일 (실제 환경이 아니면 스킵하도록 하여 링커 에러 방지)
compile_cuda:
	@echo "🔨 [입법부] CUDA 커널 객체 파일 생성 여부 체크..."
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) -c -O3 -Xcompiler -fPIC $(LEG_DIR)/src/topology_projector.cu -o $(BIN_DIR)/topology_projector.o; \
		$(NVCC) -c -O3 -Xcompiler -fPIC $(LEG_DIR)/src/circular_memory_backbone.cu -o $(BIN_DIR)/circular_memory_backbone.o; \
		$(NVCC) -c -O3 -Xcompiler -fPIC $(LEG_DIR)/src/clifford_membrane.cu -o $(BIN_DIR)/clifford_membrane.o; \
		$(NVCC) -c -O3 -Xcompiler -fPIC $(LEG_DIR)/src/holographic_manifold_kernel.cu -o $(BIN_DIR)/holographic_manifold_kernel.o; \
	else \
		echo "⚠️ NVCC not found. Skipping CUDA compilation to avoid linker errors."; \
	fi

# 3. 행정부/입법부 통합 빌드 (pybind11)
build_executive: compile_cuda
	@echo "🎬 [행정부] pybind11 바인딩을 통한 C/CUDA 입법부 컴파일 시작..."
	@if [ -f $(BIN_DIR)/topology_projector.o ]; then \
		$(CXX) -O3 -shared -fPIC $(PYBIND_INCLUDES) $(CPP_SRCS) $(BIND_SRCS) $(CU_OBJS) -L/usr/local/cuda/lib64 -lcudart -o $(TARGET); \
	else \
		$(CXX) -O3 -shared -fPIC $(PYBIND_INCLUDES) $(CPP_SRCS) $(BIND_SRCS) -o $(TARGET); \
	fi
	@echo "✅ 빌드 완료: $(TARGET)"

clean:
	rm -rf $(BIN_DIR)/* $(TARGET)
	@echo "🧹 클린 완료."

# Phase Inverter Target
build_phase_inverter:
	@echo "🔨 [입법부/행정부] Phase Inverter Native Kernel (CUDA) 컴파일 시작..."
	@mkdir -p lib
	$(NVCC) -O3 --shared -Xcompiler -fPIC src/phase_kernel.cu -o lib/phase_kernel.so
	@echo "✅ Phase Inverter 빌드 완료: lib/phase_kernel.so"

all: ensure_dirs check_judiciary build_executive build_phase_inverter
