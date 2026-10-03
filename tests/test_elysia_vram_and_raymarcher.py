import os
import pytest
from core.elysia_hangul_kronecker_tokenizer import HangulKroneckerTokenizer

def test_hangul_kronecker_tokenizer():
    tokenizer = HangulKroneckerTokenizer()
    text = "엘리시아"
    wavevectors = tokenizer(text)
    assert len(wavevectors) == len(text)
    assert wavevectors[0].shape[0] == 64

def test_files_exist():
    assert os.path.exists("include/elysia/elysia_vram_ring_buffer.hpp")
    assert os.path.exists("kernels/elysia_raymarcher_kernel.cu")
    assert os.path.exists("kernels/elysia_clifford_rotor_kernel.cu")
    assert os.path.exists("src/elysia_core_loop.cu")
    assert os.path.exists("src/elysia_pinned_async_stream.cu")
    assert os.path.exists("src/elysia_integrated_bench.cu")
    assert os.path.exists("benchmarks/elysia_hangul_clifford_benchmark.py")
    assert os.path.exists("benchmarks/elysia_hangul_clifford_multi_stream_bench.py")
