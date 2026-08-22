# Makefile — Elysia C/C++ Extension & Test Build
# C 소스 파일 위치: core/ingestion/ (helix, byte, concept streamer)
# C++ 소스 파일 위치: modules/causal_topology/, tests/cpp/
# 빌드 산출물: core/bin/

CC      = gcc
CXX     = g++
CFLAGS  = -O3 -Wall -shared -fPIC
CXXFLAGS = -O3 -Wall -std=c++17 -fopenmp -Iinclude -Imodules/causal_topology
OUTDIR  = core/bin

# ── 공유 라이브러리 및 C++ 테스트 타겟 ──────────────────────────────
HELIX_TARGET    = $(OUTDIR)/helix_streamer.so
BYTE_TARGET     = $(OUTDIR)/byte_streamer.so
CONCEPT_TARGET  = $(OUTDIR)/concept_streamer.so
TEST_PREISACH   = $(OUTDIR)/test_preisach_tensor_field

all: $(OUTDIR) $(HELIX_TARGET) $(BYTE_TARGET) $(CONCEPT_TARGET) $(TEST_PREISACH) python_bindings
	@echo "[BUILD] All C extensions, C++ tests, and Python bindings compiled successfully!"

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(HELIX_TARGET): core/ingestion/helix_streamer.c | $(OUTDIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BYTE_TARGET): core/ingestion/byte_streamer.c | $(OUTDIR)
	$(CC) $(CFLAGS) -o $@ $<

$(CONCEPT_TARGET): core/ingestion/concept_streamer.c | $(OUTDIR)
	$(CC) $(CFLAGS) -o $@ $<

$(TEST_PREISACH): tests/cpp/test_preisach_tensor_field.cpp modules/causal_topology/preisach_tensor_field.h | $(OUTDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

python_bindings:
	python3 setup.py build_ext --inplace

test_preisach: $(TEST_PREISACH)
	./$(TEST_PREISACH)

clean:
	rm -rf $(OUTDIR) *.so build/ *.egg-info

.PHONY: all clean test_preisach python_bindings
