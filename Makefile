# 엘리시아 연방 통합 빌드 자동화 시스템 (Makefile)
# 이 파일은 6월 2일 락 해제 시 즉시 가동되는 빌드 파이프라인의 뼈대입니다.

CXX = g++
NVCC = nvcc
PYTHON = python3

# Python bind11 경로 설정 (예시)
PYBIND_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)

# 폴더 설정
LEG_DIR = legislative
EXEC_DIR = executive
JUDGE_DIR = judiciary
BIN_DIR = bin

# 입법부 C/CUDA 파일
CU_SRCS = $(LEG_DIR)/src/clifford_membrane.cu
CPP_SRCS = $(LEG_DIR)/src/fractal_rotor.cpp
BIND_SRCS = $(EXEC_DIR)/binding.cpp

# 출력 파일
TARGET = $(EXEC_DIR)/elysia_core$(PYTHON_EXT_SUFFIX)

all: ensure_dirs check_judiciary build_executive

ensure_dirs:
	mkdir -p $(BIN_DIR)

# 1. 사법부 감시 (빌드 전 상시 체크)
check_judiciary:
	@echo "⚖️ [사법부] 빌드 전 정적 분석 및 감시 시작..."
	$(PYTHON) $(JUDGE_DIR)/judiciary_demon.py

# 2. 행정부/입법부 통합 빌드 (pybind11)
# NOTE: This target is intentionally not compiled with nvcc here for the mock, to avoid nvcc requirement,
# but it serves as the scaffold for the master. We will just touch the target file for now if nvcc is missing, or compile with g++ if possible.
build_executive:
	@echo "🎬 [행정부] pybind11 바인딩을 통한 C/CUDA 입법부 컴파일 (Scaffold) 시작..."
	touch $(TARGET)
	@echo "✅ 빌드 완료 (Mock): $(TARGET)"

clean:
	rm -rf $(BIN_DIR)/* $(TARGET)
	@echo "🧹 클린 완료."
